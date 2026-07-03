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

for precision in fp32 bf16 fp16; do
  python scripts/probe_mixed_precision.py \
    --device cuda \
    --precision "${precision}" \
    --output-format json \
    > "mixed_precision_probe_a10_${precision}.json"
done

python scripts/run_mixed_precision_incremental.py \
  --device cuda \
  --basics-impl user-adapters \
  --assignment1-path "${ASSIGNMENT1_PATH}" \
  --model-sizes small medium \
  --modes forward forward_backward train_step \
  --precisions fp32 bf16 fp16 \
  --batch-size 4 \
  --context-length 512 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --output-csv benchmark_mixed_a10_small_medium_user-adapters.csv \
  --output-jsonl benchmark_mixed_a10_small_medium_user-adapters.jsonl
