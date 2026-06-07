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

for batch_size in 1 2 3 4; do
  python scripts/run_benchmark_sweep.py \
    --device cuda \
    --model-sizes xl \
    --modes train_step \
    --batch-size "${batch_size}" \
    --context-length 512 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --output-csv "benchmark_xl_train_step_bs${batch_size}.csv" \
    --output-md "benchmark_xl_train_step_bs${batch_size}.md" \
    --continue-on-error
done
