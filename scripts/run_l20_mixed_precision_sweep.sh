#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mlx/users/zuotongbin.tobin/playground/assignment2-systems}"
ASSIGNMENT1_PATH="${ASSIGNMENT1_PATH:-/mlx/users/zuotongbin.tobin/playground/assignment1-basics}"
MIXED_MODEL_SIZES="${MIXED_MODEL_SIZES:-small medium large xl}"
MIXED_MODES="${MIXED_MODES:-forward forward_backward train_step}"
MIXED_PRECISIONS="${MIXED_PRECISIONS:-fp32 bf16 fp16}"
MIXED_BATCH_SIZE="${MIXED_BATCH_SIZE:-4}"
MIXED_CONTEXT_LENGTH="${MIXED_CONTEXT_LENGTH:-512}"
MIXED_WARMUP_STEPS="${MIXED_WARMUP_STEPS:-5}"
MIXED_MEASURE_STEPS="${MIXED_MEASURE_STEPS:-10}"
BASICS_IMPL="${BASICS_IMPL:-user-adapters}"

cd "${PROJECT_DIR}"

export UV_LINK_MODE=copy

uv venv /tmp/cs336-systems-mixed-cu121-venv --seed --clear
source /tmp/cs336-systems-mixed-cu121-venv/bin/activate

uv pip install --no-progress "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-progress einops einx jaxtyping numpy pandas regex tiktoken
uv pip install --no-progress --no-deps -e ./cs336-basics
if [[ "${BASICS_IMPL}" == "user-adapters" ]]; then
  uv pip install --no-progress --no-deps -e "${ASSIGNMENT1_PATH}"
fi

python scripts/check_gpu_env.py

for precision in ${MIXED_PRECISIONS}; do
  python scripts/probe_mixed_precision.py \
    --device cuda \
    --precision "${precision}" \
    --output-format json \
    > "mixed_precision_probe_${precision}.json"

  python scripts/run_benchmark_sweep.py \
    --device cuda \
    --basics-impl "${BASICS_IMPL}" \
    --assignment1-path "${ASSIGNMENT1_PATH}" \
    --model-sizes ${MIXED_MODEL_SIZES} \
    --modes ${MIXED_MODES} \
    --batch-size "${MIXED_BATCH_SIZE}" \
    --context-length "${MIXED_CONTEXT_LENGTH}" \
    --warmup-steps "${MIXED_WARMUP_STEPS}" \
    --measure-steps "${MIXED_MEASURE_STEPS}" \
    --precision "${precision}" \
    --output-csv "benchmark_mixed_${precision}_${BASICS_IMPL}.csv" \
    --output-md "benchmark_mixed_${precision}_${BASICS_IMPL}.md" \
    --continue-on-error
done
