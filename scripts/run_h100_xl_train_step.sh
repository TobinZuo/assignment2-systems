#!/usr/bin/env bash
set -euo pipefail

cd /mnt/bn/ecomcommonnas/mlx/users/zuotongbin.tobin/playground/assignment2-systems

export UV_LINK_MODE=copy

uv venv /tmp/cs336-systems-h100-user-a1-cu121-venv --seed --clear
source /tmp/cs336-systems-h100-user-a1-cu121-venv/bin/activate

uv pip install --no-progress "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-progress einops einx jaxtyping numpy regex tiktoken
uv pip install --no-progress --no-deps -e /mnt/bn/ecomcommonnas/mlx/users/zuotongbin.tobin/playground/assignment1-basics

nvidia-smi || true
python scripts/check_gpu_env.py

python scripts/run_benchmark_sweep.py \
  --device cuda \
  --basics-impl user-adapters \
  --assignment1-path /mnt/bn/ecomcommonnas/mlx/users/zuotongbin.tobin/playground/assignment1-basics \
  --model-sizes xl \
  --modes train_step \
  --batch-size 4 \
  --context-length 512 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --output-csv benchmark_h100_xl_train_step_user_assignment1.csv \
  --output-md benchmark_h100_xl_train_step_user_assignment1.md \
  --continue-on-error
