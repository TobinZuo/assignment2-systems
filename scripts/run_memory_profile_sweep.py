from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


MODES = ["forward", "train_step"]
PRECISIONS = ["fp32", "bf16", "fp16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 2 xl memory profiling sweep.")
    parser.add_argument("--model-size", choices=["xl"], default="xl")
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 2048])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=["fp32", "bf16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--basics-impl", choices=["staff", "user-adapters"], default="user-adapters")
    parser.add_argument(
        "--assignment1-path",
        default="/mnt/bn/ecomcommonnas/mlx/users/zuotongbin.tobin/playground/assignment1-basics",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot-dir", type=Path, default=Path("memory_profiles"))
    parser.add_argument("--output-csv", type=Path, default=Path("memory_profile_results.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("memory_profile_results.md"))
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def run_one(args: argparse.Namespace, context_length: int, mode: str, precision: str) -> dict:
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.snapshot_dir / f"{args.model_size}_ctx{context_length}_{mode}_{precision}.pickle"
    command = [
        sys.executable,
        "scripts/benchmark_transformer.py",
        "--model-size",
        args.model_size,
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(context_length),
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
        "--profile-memory",
        "--memory-snapshot-path",
        str(snapshot_path),
        "--output-format",
        "json",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode == 0:
        row = json.loads(completed.stdout)
        row["error"] = ""
        return row
    if not args.continue_on_error:
        print(completed.stdout, end="", file=sys.stdout)
        print(completed.stderr, end="", file=sys.stderr)
        completed.check_returncode()
    return {
        "model_size": args.model_size,
        "mode": mode,
        "batch_size": args.batch_size,
        "context_length": context_length,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "mean_ms": "",
        "std_ms": "",
        "device": args.device,
        "precision": precision,
        "basics_impl": args.basics_impl,
        "peak_allocated_bytes": "",
        "peak_reserved_bytes": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "memory_snapshot_path": str(snapshot_path),
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
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "memory_snapshot_path",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_peak(row: dict | None) -> str:
    if row is None:
        return "-"
    if row.get("error"):
        return "OOM/error"
    peak = row.get("peak_allocated_mib")
    return f"{float(peak):.1f} MiB" if peak not in {"", None} else "-"


def write_markdown(path: Path, rows: list[dict]) -> None:
    by_key = {(row["context_length"], row["mode"], row["precision"]): row for row in rows}
    lines = [
        "# Memory Profiling Results",
        "",
        "Peak values are `torch.cuda.max_memory_allocated()` after warmup, measured over the profiled step(s). Snapshot pickle files can be opened at https://pytorch.org/memory_viz.",
        "",
        "## fp32 peak memory",
        "",
        "| Context Length | Forward Peak Memory | Train Step Peak Memory |",
        "|-|-|-|",
    ]
    contexts = sorted({int(row["context_length"]) for row in rows})
    for context_length in contexts:
        lines.append(
            f"| {context_length} | "
            f"{format_peak(by_key.get((context_length, 'forward', 'fp32')))} | "
            f"{format_peak(by_key.get((context_length, 'train_step', 'fp32')))} |"
        )

    lines.extend(
        [
            "",
            "## Mixed precision comparison",
            "",
            "| Context Length | Mode | fp32 Peak | bf16 Peak | fp16 Peak |",
            "|-|-|-|-|-|",
        ]
    )
    for context_length in contexts:
        for mode in MODES:
            lines.append(
                f"| {context_length} | {mode} | "
                f"{format_peak(by_key.get((context_length, mode, 'fp32')))} | "
                f"{format_peak(by_key.get((context_length, mode, 'bf16')))} | "
                f"{format_peak(by_key.get((context_length, mode, 'fp16')))} |"
            )

    activation_mib = 4 * 2048 * 2560 * 4 / (1024 * 1024)
    lines.extend(
        [
            "",
            "## Residual stream activation size",
            "",
            "`batch_size * context_length * d_model * bytes_per_element` for xl at batch 4, context 2048, d_model 2560, fp32:",
            "",
            f"`4 * 2048 * 2560 * 4 / 1024^2 = {activation_mib:.1f} MiB`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = []
    for context_length in args.contexts:
        for mode in args.modes:
            for precision in args.precisions:
                print(f"Running memory profile context={context_length} mode={mode} precision={precision}...", flush=True)
                rows.append(run_one(args, context_length, mode, precision))

    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote snapshots under {args.snapshot_dir}")


if __name__ == "__main__":
    main()
