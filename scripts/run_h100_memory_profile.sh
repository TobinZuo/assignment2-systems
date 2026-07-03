#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/bn/ecomcommonnas/mlx/users/zuotongbin.tobin/playground/assignment2-systems
ASSIGNMENT1_PATH=/mnt/bn/ecomcommonnas/mlx/users/zuotongbin.tobin/playground/assignment1-basics
LOG="${ROOT}/memory_profiles/h100_memory_profile.log"

mkdir -p "${ROOT}/memory_profiles"
exec > >(tee -a "${LOG}") 2>&1

echo "[start] $(date -Is)"
echo "[host] $(hostname)"
nvidia-smi || true

export UV_LINK_MODE=copy
export PATH="${HOME}/.local/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"

uv venv /tmp/cs336-systems-memory-cu121-venv --seed --clear
source /tmp/cs336-systems-memory-cu121-venv/bin/activate

uv pip install --no-progress "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-progress einops einx jaxtyping numpy regex tiktoken
uv pip install --no-progress --no-deps -e "${ASSIGNMENT1_PATH}"

python scripts/check_gpu_env.py

python scripts/run_memory_profile_sweep.py \
  --device cuda \
  --basics-impl user-adapters \
  --assignment1-path "${ASSIGNMENT1_PATH}" \
  --contexts 128 2048 \
  --modes forward train_step \
  --precisions fp32 bf16 \
  --batch-size 4 \
  --warmup-steps 1 \
  --measure-steps 1 \
  --snapshot-dir memory_profiles \
  --output-csv memory_profile_results.csv \
  --output-md memory_profile_results.md \
  --continue-on-error

ls -lh memory_profile_results.csv memory_profile_results.md memory_profiles || true
echo "[done] $(date -Is)"
