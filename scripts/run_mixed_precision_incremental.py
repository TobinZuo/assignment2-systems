from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


FIELDNAMES = [
    "model_size",
    "mode",
    "batch_size",
    "context_length",
    "warmup_steps",
    "measure_steps",
    "mean_ms",
    "std_ms",
    "device",
    "precision",
    "basics_impl",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "memory_snapshot_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mixed precision benchmark combinations and persist after every row.")
    parser.add_argument("--model-sizes", nargs="+", default=["small", "medium", "large", "xl"])
    parser.add_argument("--modes", nargs="+", default=["forward", "forward_backward", "train_step"])
    parser.add_argument("--precisions", nargs="+", default=["fp32", "bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--basics-impl", choices=["staff", "user-adapters"], default="user-adapters")
    parser.add_argument("--assignment1-path", default="/mlx/users/zuotongbin.tobin/playground/assignment1-basics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("benchmark_mixed_incremental.csv"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("benchmark_mixed_incremental.jsonl"))
    return parser.parse_args()


def run_one(args: argparse.Namespace, precision: str, model_size: str, mode: str) -> dict:
    command = [
        sys.executable,
        "scripts/benchmark_transformer.py",
        "--model-size",
        model_size,
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(args.context_length),
        "--mode",
        mode,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
        "--device",
        args.device,
        "--precision",
        precision,
        "--basics-impl",
        args.basics_impl,
        "--assignment1-path",
        args.assignment1_path,
        "--seed",
        str(args.seed),
        "--output-format",
        "json",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode == 0:
        return json.loads(completed.stdout)
    return {
        "model_size": model_size,
        "mode": mode,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "mean_ms": "",
        "std_ms": "",
        "device": args.device,
        "precision": precision,
        "basics_impl": args.basics_impl,
        "error": (completed.stderr or completed.stdout).strip().splitlines()[-1],
    }


def append_row(csv_path: Path, jsonl_path: Path, row: dict) -> None:
    csv_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()

    with jsonl_path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def main() -> None:
    args = parse_args()
    args.output_csv.unlink(missing_ok=True)
    args.output_jsonl.unlink(missing_ok=True)

    for precision in args.precisions:
        for model_size in args.model_sizes:
            for mode in args.modes:
                print(f"Running {precision} {model_size} {mode}...", flush=True)
                row = run_one(args, precision, model_size, mode)
                append_row(args.output_csv, args.output_jsonl, row)


if __name__ == "__main__":
    main()
