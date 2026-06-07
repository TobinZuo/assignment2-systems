#!/usr/bin/env bash
set -euo pipefail

cd /mnt/bn/ai-infra-aigc-my/mlx/users/zuotongbin.tobin/playground/assignment2-systems

export UV_LINK_MODE=copy

uv venv /tmp/cs336-systems-cu121-venv --seed --clear
source /tmp/cs336-systems-cu121-venv/bin/activate

uv pip install --no-progress "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-progress einops einx jaxtyping numpy
uv pip install --no-progress --no-deps -e ./cs336-basics

python scripts/check_gpu_env.py

python scripts/run_benchmark_sweep.py \
  --device cuda \
  --batch-size 4 \
  --context-length 512 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --output-csv benchmark_results.csv \
  --output-md benchmark_results.md \
  --continue-on-error

python scripts/run_benchmark_sweep.py \
  --device cuda \
  --model-sizes small \
  --batch-size 4 \
  --context-length 512 \
  --warmup-steps 0 \
  --measure-steps 10 \
  --output-csv benchmark_warmup0.csv \
  --output-md benchmark_warmup0.md \
  --continue-on-error

python scripts/run_benchmark_sweep.py \
  --device cuda \
  --model-sizes small \
  --batch-size 4 \
  --context-length 512 \
  --warmup-steps 1 \
  --measure-steps 10 \
  --output-csv benchmark_warmup1.csv \
  --output-md benchmark_warmup1.md \
  --continue-on-error

python scripts/run_benchmark_sweep.py \
  --device cuda \
  --model-sizes small \
  --batch-size 4 \
  --context-length 512 \
  --warmup-steps 2 \
  --measure-steps 10 \
  --output-csv benchmark_warmup2.csv \
  --output-md benchmark_warmup2.md \
  --continue-on-error
