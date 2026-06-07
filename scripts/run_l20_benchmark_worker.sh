#!/usr/bin/env bash
set -euo pipefail

cd /mnt/bn/ai-infra-aigc-my/mlx/users/zuotongbin.tobin/playground/assignment2-systems

export UV_LINK_MODE=copy

uv venv /tmp/bench-cu121-venv --seed --clear
source /tmp/bench-cu121-venv/bin/activate

uv pip install --no-progress "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-progress einops einx jaxtyping numpy
uv pip install --no-progress --no-deps -e ./cs336-basics

python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

python scripts/benchmark_transformer.py \
  --model-size small \
  --batch-size 1 \
  --context-length 8 \
  --mode forward \
  --warmup-steps 0 \
  --measure-steps 1 \
  --device cuda \
  --precision fp32 \
  --seed 0 \
  --output-format json
