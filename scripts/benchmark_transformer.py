from __future__ import annotations

import argparse
import contextlib
import inspect
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F


Mode = Literal["forward", "forward_backward", "train_step"]
BasicsImpl = Literal["staff", "user-adapters"]
Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class BenchmarkResult:
    model_size: str
    mode: str
    batch_size: int
    context_length: int
    warmup_steps: int
    measure_steps: int
    mean_ms: float
    std_ms: float
    device: str
    precision: str
    basics_impl: str
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    peak_allocated_mib: float | None = None
    peak_reserved_mib: float | None = None
    memory_snapshot_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cs336_basics Transformer steps.")
    parser.add_argument("--model-size", choices=["small", "medium", "large", "xl"], default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="forward")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--basics-impl", choices=["staff", "user-adapters"], default="staff")
    parser.add_argument(
        "--assignment1-path",
        type=Path,
        default=Path("/mnt/bn/ai-infra-aigc-my/mlx/users/zuotongbin.tobin/playground/assignment1-basics"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-format", choices=["table", "json"], default="table")
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges for Nsight Systems profiling.")
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Record PyTorch CUDA allocator peak stats for measured steps, after warmup.",
    )
    parser.add_argument(
        "--memory-snapshot-path",
        type=Path,
        default=None,
        help="Optional .pickle output for torch.cuda.memory._dump_snapshot, viewable in https://pytorch.org/memory_viz.",
    )
    parser.add_argument(
        "--memory-history-max-entries",
        type=int,
        default=200_000,
        help="Maximum allocator events kept in the PyTorch memory history snapshot.",
    )
    return parser.parse_args()


def get_model_config(model_size: str, context_length: int) -> ModelConfig:
    configs = {
        "small": dict(d_model=768, d_ff=3_072, num_layers=12, num_heads=12),
        "medium": dict(d_model=1_024, d_ff=4_096, num_layers=24, num_heads=16),
        "large": dict(d_model=1_280, d_ff=5_120, num_layers=36, num_heads=20),
        "xl": dict(d_model=2_560, d_ff=10_240, num_layers=32, num_heads=32),
    }
    return ModelConfig(
        vocab_size=10_000,
        context_length=context_length,
        **configs[model_size],
    )


class UserAdapterTransformerLM(torch.nn.Module):
    def __init__(self, config: ModelConfig, adapters, rope_theta: float = 10_000.0):
        super().__init__()
        self.config = config
        self.adapters = adapters
        self.rope_theta = rope_theta
        self.names = []
        self.params = torch.nn.ParameterList()
        self._add_weight("token_embeddings.weight", (config.vocab_size, config.d_model))
        for layer_idx in range(config.num_layers):
            prefix = f"layers.{layer_idx}."
            self._add_weight(prefix + "attn.q_proj.weight", (config.d_model, config.d_model))
            self._add_weight(prefix + "attn.k_proj.weight", (config.d_model, config.d_model))
            self._add_weight(prefix + "attn.v_proj.weight", (config.d_model, config.d_model))
            self._add_weight(prefix + "attn.output_proj.weight", (config.d_model, config.d_model))
            self._add_weight(prefix + "ln1.weight", (config.d_model,), init="ones")
            self._add_weight(prefix + "ffn.w1.weight", (config.d_ff, config.d_model))
            self._add_weight(prefix + "ffn.w2.weight", (config.d_model, config.d_ff))
            self._add_weight(prefix + "ffn.w3.weight", (config.d_ff, config.d_model))
            self._add_weight(prefix + "ln2.weight", (config.d_model,), init="ones")
        self._add_weight("ln_final.weight", (config.d_model,), init="ones")
        self._add_weight("lm_head.weight", (config.vocab_size, config.d_model))

    def _add_weight(self, name: str, shape: tuple[int, ...], init: str = "normal") -> None:
        if init == "ones":
            param = torch.nn.Parameter(torch.ones(shape))
        else:
            param = torch.nn.Parameter(torch.empty(shape))
            torch.nn.init.trunc_normal_(param, std=0.02, a=-0.06, b=0.06)
        self.names.append(name)
        self.params.append(param)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        weights = {name: param for name, param in zip(self.names, self.params)}
        return self.adapters.run_transformer_lm(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            rope_theta=self.rope_theta,
            weights=weights,
            in_indices=input_ids,
        )


def load_user_adapters(assignment1_path: Path):
    sys.path.insert(0, str(assignment1_path))
    adapters_path = assignment1_path / "tests" / "adapters.py"
    spec = importlib.util.spec_from_file_location("assignment1_user_adapters", adapters_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load adapters from {adapters_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(config: ModelConfig, device: torch.device, basics_impl: BasicsImpl, assignment1_path: Path) -> torch.nn.Module:
    if basics_impl == "staff":
        from cs336_basics.model import BasicsTransformerLM

        model = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
        )
    else:
        model = UserAdapterTransformerLM(config=config, adapters=load_user_adapters(assignment1_path))
    return model.to(device)


def make_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    return input_ids, targets


def build_optimizer(model: torch.nn.Module, basics_impl: BasicsImpl, assignment1_path: Path) -> torch.optim.Optimizer:
    if basics_impl == "staff":
        from cs336_basics.optimizer import AdamW
    else:
        AdamW = load_user_adapters(assignment1_path).get_adamw_cls()
    return AdamW(model.parameters(), lr=1e-3)


@contextlib.contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def compute_loss(model: torch.nn.Module, input_ids: torch.Tensor, targets: torch.Tensor, emit_nvtx: bool) -> torch.Tensor:
    with nvtx_range("model_forward", emit_nvtx):
        logits = model(input_ids)
    with nvtx_range("loss", emit_nvtx):
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def autocast_dtype(precision: Precision) -> torch.dtype | None:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


@contextlib.contextmanager
def precision_context(device: torch.device, precision: Precision):
    dtype = autocast_dtype(precision)
    if dtype is None:
        yield
        return
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"Autocast precision {precision} is only supported for CPU/CUDA devices, got {device}.")
    with torch.autocast(device_type=device.type, dtype=dtype):
        yield


def make_grad_scaler(device: torch.device, precision: Precision, mode: Mode):
    enabled = device.type == "cuda" and precision == "fp16" and mode in {"forward_backward", "train_step"}
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def run_one_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    mode: Mode,
    device: torch.device,
    precision: Precision,
    grad_scaler,
    emit_nvtx: bool,
) -> torch.Tensor | None:
    input_ids, targets = batch

    if mode == "forward":
        with torch.no_grad(), nvtx_range("forward", emit_nvtx):
            with precision_context(device, precision):
                return compute_loss(model, input_ids, targets, emit_nvtx)

    if mode == "forward_backward":
        with nvtx_range("zero_grad", emit_nvtx):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        with nvtx_range("forward", emit_nvtx):
            with precision_context(device, precision):
                loss = compute_loss(model, input_ids, targets, emit_nvtx)
        with nvtx_range("backward", emit_nvtx):
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss

    if mode == "train_step":
        if optimizer is None:
            raise ValueError("train_step mode requires an optimizer.")
        with nvtx_range("train_step", emit_nvtx):
            with nvtx_range("zero_grad", emit_nvtx):
                optimizer.zero_grad(set_to_none=True)
            with nvtx_range("forward", emit_nvtx):
                with precision_context(device, precision):
                    loss = compute_loss(model, input_ids, targets, emit_nvtx)
            with nvtx_range("backward", emit_nvtx):
                if grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
            with nvtx_range("optimizer_step", emit_nvtx):
                if grad_scaler.is_enabled():
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
        return loss

    raise ValueError(f"Unsupported mode: {mode}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


@contextlib.contextmanager
def cuda_memory_history(
    enabled: bool,
    device: torch.device,
    snapshot_path: Path | None,
    max_entries: int,
):
    if not enabled:
        yield None
        return
    if device.type != "cuda":
        raise ValueError("--profile-memory requires a CUDA device.")
    if not hasattr(torch.cuda.memory, "_record_memory_history") or not hasattr(torch.cuda.memory, "_dump_snapshot"):
        raise RuntimeError("This PyTorch build does not expose CUDA memory snapshot APIs.")

    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    record_kwargs = {
        "enabled": "all",
        "context": "all",
        "stacks": "all",
        "max_entries": max_entries,
        "device": device,
        "clear_history": True,
    }
    supported = set(inspect.signature(torch.cuda.memory._record_memory_history).parameters)
    record_kwargs = {key: value for key, value in record_kwargs.items() if key in supported}
    torch.cuda.memory._record_memory_history(**record_kwargs)
    try:
        yield snapshot_path
    finally:
        if snapshot_path is not None:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
        torch.cuda.memory._record_memory_history(enabled=None)


def time_one_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    mode: Mode,
    device: torch.device,
    precision: Precision,
    grad_scaler,
    emit_nvtx: bool,
    step_label: str,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    with nvtx_range(step_label, emit_nvtx):
        run_one_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            mode=mode,
            device=device,
            precision=precision,
            grad_scaler=grad_scaler,
            emit_nvtx=emit_nvtx,
        )
    synchronize(device)
    end = time.perf_counter()
    return (end - start) * 1_000


def run_warmup(
    warmup_steps: int,
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    mode: Mode,
    device: torch.device,
    precision: Precision,
    grad_scaler,
    emit_nvtx: bool,
) -> None:
    for step_idx in range(warmup_steps):
        time_one_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            mode=mode,
            device=device,
            precision=precision,
            grad_scaler=grad_scaler,
            emit_nvtx=emit_nvtx,
            step_label=f"warmup_step_{step_idx}",
        )


def benchmark(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    mode: Mode,
    warmup_steps: int,
    measure_steps: int,
    model_size: str,
    batch_size: int,
    context_length: int,
    device: torch.device,
    precision: Precision,
    basics_impl: str,
    emit_nvtx: bool,
    profile_memory: bool,
    memory_snapshot_path: Path | None,
    memory_history_max_entries: int,
) -> BenchmarkResult:
    grad_scaler = make_grad_scaler(device, precision, mode)
    run_warmup(
        warmup_steps=warmup_steps,
        model=model,
        batch=batch,
        optimizer=optimizer,
        mode=mode,
        device=device,
        precision=precision,
        grad_scaler=grad_scaler,
        emit_nvtx=emit_nvtx,
    )

    peak_allocated_bytes = None
    peak_reserved_bytes = None
    snapshot_output_path = str(memory_snapshot_path) if memory_snapshot_path is not None else None

    synchronize(device)
    if profile_memory:
        if device.type != "cuda":
            raise ValueError("--profile-memory requires a CUDA device.")
        torch.cuda.reset_peak_memory_stats(device)

    step_times_ms = []
    with cuda_memory_history(
        enabled=profile_memory and memory_snapshot_path is not None,
        device=device,
        snapshot_path=memory_snapshot_path,
        max_entries=memory_history_max_entries,
    ):
        for step_idx in range(measure_steps):
            step_times_ms.append(
                time_one_step(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    mode=mode,
                    device=device,
                    precision=precision,
                    grad_scaler=grad_scaler,
                    emit_nvtx=emit_nvtx,
                    step_label=f"measure_step_{step_idx}",
                )
            )

    synchronize(device)
    if profile_memory:
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

    return BenchmarkResult(
        model_size=model_size,
        mode=mode,
        batch_size=batch_size,
        context_length=context_length,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        mean_ms=statistics.fmean(step_times_ms),
        std_ms=statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0,
        device=str(device),
        precision=precision,
        basics_impl=basics_impl,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        peak_allocated_mib=mib(peak_allocated_bytes) if peak_allocated_bytes is not None else None,
        peak_reserved_mib=mib(peak_reserved_bytes) if peak_reserved_bytes is not None else None,
        memory_snapshot_path=snapshot_output_path,
    )


def print_result(result: BenchmarkResult, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    rows = [
        ("model_size", result.model_size),
        ("mode", result.mode),
        ("batch_size", result.batch_size),
        ("context_length", result.context_length),
        ("warmup_steps", result.warmup_steps),
        ("measure_steps", result.measure_steps),
        ("mean_ms", f"{result.mean_ms:.3f}"),
        ("std_ms", f"{result.std_ms:.3f}"),
        ("device", result.device),
        ("precision", result.precision),
        ("basics_impl", result.basics_impl),
    ]
    if result.peak_allocated_bytes is not None:
        rows.extend(
            [
                ("peak_allocated_mib", f"{result.peak_allocated_mib:.3f}"),
                ("peak_reserved_mib", f"{result.peak_reserved_mib:.3f}"),
            ]
        )
    if result.memory_snapshot_path is not None:
        rows.append(("memory_snapshot_path", result.memory_snapshot_path))
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")


def main() -> None:
    args = parse_args()
    if args.memory_snapshot_path is not None:
        args.profile_memory = True
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = get_model_config(args.model_size, args.context_length)
    model = build_model(config, device, args.basics_impl, args.assignment1_path)
    model.eval() if args.mode == "forward" else model.train()

    batch = make_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=config.vocab_size,
        device=device,
    )
    optimizer = build_optimizer(model, args.basics_impl, args.assignment1_path) if args.mode in {"forward_backward", "train_step"} else None

    result = benchmark(
        model=model,
        batch=batch,
        optimizer=optimizer,
        mode=args.mode,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
        precision=args.precision,
        basics_impl=args.basics_impl,
        emit_nvtx=args.nvtx,
        profile_memory=args.profile_memory,
        memory_snapshot_path=args.memory_snapshot_path,
        memory_history_max_entries=args.memory_history_max_entries,
    )
    print_result(result, args.output_format)


if __name__ == "__main__":
    main()
