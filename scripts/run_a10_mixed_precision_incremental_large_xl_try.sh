#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mlx/users/zuotongbin.tobin/playground/assignment2-systems"
ASSIGNMENT1_PATH="/mlx/users/zuotongbin.tobin/playground/assignment1-basics"

cd "${PROJECT_DIR}"

export UV_LINK_MODE=copy

uv venv /tmp/cs336-systems-a10-mixed-cu121-venv --seed --clear
source /tmp/cs336-systems-a10-mixed-cu121-venv/bin/activate

uv pip install --no-progress "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-progress einops einx jaxtyping numpy pandas regex tiktoken
uv pip install --no-progress --no-deps -e ./cs336-basics
uv pip install --no-progress --no-deps -e "${ASSIGNMENT1_PATH}"

python scripts/check_worker_cuda.py

python scripts/run_mixed_precision_incremental.py \
  --device cuda \
  --basics-impl user-adapters \
  --assignment1-path "${ASSIGNMENT1_PATH}" \
  --model-sizes large xl \
  --modes forward forward_backward train_step \
  --precisions fp32 bf16 fp16 \
  --batch-size 4 \
  --context-length 512 \
  --warmup-steps 1 \
  --measure-steps 3 \
  --output-csv benchmark_mixed_a10_large_xl_try_user-adapters.csv \
  --output-jsonl benchmark_mixed_a10_large_xl_try_user-adapters.jsonl
