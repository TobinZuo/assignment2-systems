from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


MODEL_SIZES = ["small", "medium", "large", "xl"]
MODES = ["forward", "forward_backward", "train_step"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 2 benchmark timing sweep.")
    parser.add_argument("--model-sizes", nargs="+", choices=MODEL_SIZES, default=MODEL_SIZES)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32"], default="fp32")
    parser.add_argument("--basics-impl", choices=["staff", "user-adapters"], default="staff")
    parser.add_argument(
        "--assignment1-path",
        default="/mnt/bn/ai-infra-aigc-my/mlx/users/zuotongbin.tobin/playground/assignment1-basics",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("benchmark_results.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmark_results.md"))
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def run_one(args: argparse.Namespace, model_size: str, mode: str) -> dict:
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
        args.precision,
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
    if not args.continue_on_error:
        completed.check_returncode()
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
        "precision": args.precision,
        "basics_impl": args.basics_impl,
        "error": (completed.stderr or completed.stdout).strip().splitlines()[-1],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
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
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    by_key = {(row["model_size"], row["mode"]): row for row in rows}
    lines = [
        "| Model | Forward Mean | Forward Std | Fwd+Bwd Mean | Fwd+Bwd Std | Train Step Mean | Train Step Std |",
        "|-|-|-|-|-|-|-|",
    ]
    for model_size in MODEL_SIZES:
        forward = by_key.get((model_size, "forward"))
        fwd_bwd = by_key.get((model_size, "forward_backward"))
        train_step = by_key.get((model_size, "train_step"))

        def mean(row):
            return f"{row['mean_ms']:.3f}" if row and row.get("mean_ms") != "" else "ERROR" if row else "-"

        def std(row):
            return f"{row['std_ms']:.3f}" if row and row.get("std_ms") != "" else "ERROR" if row else "-"

        lines.append(
            "| "
            + " | ".join(
                [
                    model_size,
                    mean(forward),
                    std(forward),
                    mean(fwd_bwd),
                    std(fwd_bwd),
                    mean(train_step),
                    std(train_step),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = []
    for model_size in args.model_sizes:
        for mode in args.modes:
            print(f"Running {model_size} {mode}...", flush=True)
            rows.append(run_one(args, model_size, mode))

    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
