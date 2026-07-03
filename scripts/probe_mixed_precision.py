from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DTypeRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]


class ToyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 64):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.linear1 = torch.nn.Linear(d_model, 4 * d_model)
        self.linear2 = torch.nn.Linear(4 * d_model, d_model)
        self.output = torch.nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        hidden = self.linear1(embedded)
        activated = F.gelu(hidden)
        projected = self.linear2(activated)
        attention_scores = projected @ projected.transpose(-1, -2)
        attention_probs = F.softmax(attention_scores, dim=-1)
        mixed = attention_probs @ projected
        logits = self.output(mixed)
        return {
            "embedding": embedded,
            "linear1": hidden,
            "gelu": activated,
            "linear2": projected,
            "matmul_scores": attention_scores,
            "softmax": attention_probs,
            "matmul_values": mixed,
            "logits": logits,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe autocast dtypes for a small mixed-precision ToyModel.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-format", choices=["table", "json"], default="table")
    return parser.parse_args()


def autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


@contextlib.contextmanager
def precision_context(device: torch.device, precision: str):
    dtype = autocast_dtype(precision)
    if dtype is None:
        yield
    else:
        with torch.autocast(device_type=device.type, dtype=dtype):
            yield


def collect_records(args: argparse.Namespace) -> list[DTypeRecord]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = ToyModel(vocab_size=args.vocab_size, d_model=args.d_model).to(device)
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    with precision_context(device, args.precision):
        outputs = model(input_ids)
        loss = F.cross_entropy(outputs["logits"].flatten(0, 1), targets.flatten())

    records = [
        DTypeRecord("parameter.embedding.weight", str(model.embedding.weight.dtype), tuple(model.embedding.weight.shape)),
        DTypeRecord("parameter.linear1.weight", str(model.linear1.weight.dtype), tuple(model.linear1.weight.shape)),
    ]
    records.extend(DTypeRecord(name, str(tensor.dtype), tuple(tensor.shape)) for name, tensor in outputs.items())
    records.append(DTypeRecord("loss", str(loss.dtype), tuple(loss.shape)))
    return records


def print_records(records: list[DTypeRecord], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps([asdict(record) for record in records], indent=2, sort_keys=True))
        return

    name_width = max(len(record.name) for record in records)
    dtype_width = max(len(record.dtype) for record in records)
    print(f"{'name':<{name_width}}  {'dtype':<{dtype_width}}  shape")
    print(f"{'-' * name_width}  {'-' * dtype_width}  -----")
    for record in records:
        print(f"{record.name:<{name_width}}  {record.dtype:<{dtype_width}}  {record.shape}")


def main() -> None:
    args = parse_args()
    records = collect_records(args)
    print_records(records, args.output_format)


if __name__ == "__main__":
    main()
