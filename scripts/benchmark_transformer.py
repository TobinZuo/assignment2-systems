from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


Mode = Literal["forward", "forward_backward", "train_step"]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cs336_basics Transformer steps.")
    parser.add_argument("--model-size", choices=["small", "medium", "large", "xl"], default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="forward")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32"], default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-format", choices=["table", "json"], default="table")
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


def build_model(config: ModelConfig, device: torch.device) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    )
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


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return AdamW(model.parameters(), lr=1e-3)


def compute_loss(model: torch.nn.Module, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def run_one_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    mode: Mode,
) -> torch.Tensor | None:
    input_ids, targets = batch

    if mode == "forward":
        with torch.no_grad():
            return compute_loss(model, input_ids, targets)

    if mode == "forward_backward":
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, input_ids, targets)
        loss.backward()
        return loss

    if mode == "train_step":
        if optimizer is None:
            raise ValueError("train_step mode requires an optimizer.")
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, input_ids, targets)
        loss.backward()
        optimizer.step()
        return loss

    raise ValueError(f"Unsupported mode: {mode}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_one_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    mode: Mode,
    device: torch.device,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    run_one_step(model=model, batch=batch, optimizer=optimizer, mode=mode)
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
) -> None:
    for _ in range(warmup_steps):
        time_one_step(model=model, batch=batch, optimizer=optimizer, mode=mode, device=device)


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
    precision: str,
) -> BenchmarkResult:
    run_warmup(
        warmup_steps=warmup_steps,
        model=model,
        batch=batch,
        optimizer=optimizer,
        mode=mode,
        device=device,
    )

    step_times_ms = [
        time_one_step(model=model, batch=batch, optimizer=optimizer, mode=mode, device=device)
        for _ in range(measure_steps)
    ]

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
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")


def main() -> None:
    args = parse_args()
    if args.precision != "fp32":
        raise NotImplementedError("First version only supports --precision fp32.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = get_model_config(args.model_size, args.context_length)
    model = build_model(config, device)
    model.eval() if args.mode == "forward" else model.train()

    batch = make_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=config.vocab_size,
        device=device,
    )
    optimizer = build_optimizer(model) if args.mode in {"forward_backward", "train_step"} else None

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
    )
    print_result(result, args.output_format)


if __name__ == "__main__":
    main()
