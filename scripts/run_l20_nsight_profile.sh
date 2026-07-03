#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/bn/ecomcommonnas/mlx/users/zuotongbin.tobin/playground/assignment2-systems}
ASSIGNMENT1_PATH=${ASSIGNMENT1_PATH:-/mnt/bn/ecomcommonnas/mlx/users/zuotongbin.tobin/playground/assignment1-basics}
PROFILE_MODE=${NSIGHT_PROFILE_MODE:-smoke}
OUTPUT_DIR=${NSIGHT_OUTPUT_DIR:-nsight_profiles}
BATCH_SIZE=${BATCH_SIZE:-4}
WARMUP_STEPS=${WARMUP_STEPS:-2}
MEASURE_STEPS=${MEASURE_STEPS:-3}
NSIGHT_MODELS=${NSIGHT_MODELS:-"small medium"}
NSIGHT_CONTEXTS=${NSIGHT_CONTEXTS:-"512 1024 2048"}
NSIGHT_BENCHMARK_MODES=${NSIGHT_BENCHMARK_MODES:-"forward forward_backward train_step"}

cd "$REPO_ROOT"

export UV_LINK_MODE=copy
export PATH="${HOME}/.local/bin:${PATH}"

uv venv /tmp/cs336-systems-nsight-user-a1-cu121-venv --seed --clear
source /tmp/cs336-systems-nsight-user-a1-cu121-venv/bin/activate

uv pip install --no-progress "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-progress einops einx jaxtyping numpy regex tiktoken
uv pip install --no-progress --no-deps -e "$ASSIGNMENT1_PATH"

nvidia-smi || true
python scripts/check_gpu_env.py

bash scripts/install_nsight_systems.sh
NSIGHT_VERSION=${NSIGHT_VERSION:-2025.6.3}
NSIGHT_ROOT=${NSIGHT_INSTALL_ROOT:-${HOME}/.local/nsight-systems-${NSIGHT_VERSION}}/opt/nvidia/nsight-systems/${NSIGHT_VERSION}
export LD_LIBRARY_PATH="${NSIGHT_ROOT}/host-linux-x64:${NSIGHT_ROOT}/target-linux-x64:${LD_LIBRARY_PATH:-}"
nsys --version || true

mkdir -p "$OUTPUT_DIR"
echo "Nsight profile config: mode=${PROFILE_MODE} models=${NSIGHT_MODELS} contexts=${NSIGHT_CONTEXTS} benchmark_modes=${NSIGHT_BENCHMARK_MODES} batch_size=${BATCH_SIZE} warmup_steps=${WARMUP_STEPS} measure_steps=${MEASURE_STEPS}"

run_profile() {
  local model_size=$1
  local context_length=$2
  local mode=$3
  local prefix="${OUTPUT_DIR}/${model_size}_ctx${context_length}_${mode}"
  local raw_log="${prefix}.log"

  echo "Profiling model=${model_size} context=${context_length} mode=${mode}"
  rm -f "${prefix}.json" "${raw_log}" "${prefix}.nsys-rep" "${prefix}.qdrep" "${prefix}.qdstrm" "${prefix}.sqlite" "${prefix}"_stats_*.csv
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --stats=true \
    --output "$prefix" \
    python scripts/benchmark_transformer.py \
      --device cuda \
      --basics-impl user-adapters \
      --assignment1-path "$ASSIGNMENT1_PATH" \
      --model-size "$model_size" \
      --context-length "$context_length" \
      --batch-size "$BATCH_SIZE" \
      --mode "$mode" \
      --warmup-steps "$WARMUP_STEPS" \
      --measure-steps "$MEASURE_STEPS" \
      --nvtx \
      --output-format json | tee "${raw_log}"

  python - "$raw_log" "${prefix}.json" <<'PY'
import json
import sys

raw_path, json_path = sys.argv[1], sys.argv[2]
text = open(raw_path, encoding="utf-8").read()
start = text.find("{")
if start < 0:
    raise SystemExit(f"No JSON object found in {raw_path}")

decoder = json.JSONDecoder()
obj, _ = decoder.raw_decode(text[start:])
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
    f.write("\n")
PY

  local report=""
  if [[ -f "${prefix}.nsys-rep" ]]; then
    report="${prefix}.nsys-rep"
  elif [[ -f "${prefix}.qdrep" ]]; then
    report="${prefix}.qdrep"
  fi

  if [[ -n "$report" ]]; then
    nsys stats --report cuda_gpu_kern_sum,nvtx_sum --format csv --force-overwrite true --output "${prefix}_stats" "$report" || true
    nsys export --type sqlite --force-overwrite true --output "${prefix}.sqlite" "$report" || true
  fi
}

if [[ "$PROFILE_MODE" == "smoke" ]]; then
  run_profile small 512 train_step
elif [[ "$PROFILE_MODE" == "full" ]]; then
  for model_size in $NSIGHT_MODELS; do
    for context_length in $NSIGHT_CONTEXTS; do
      for mode in $NSIGHT_BENCHMARK_MODES; do
        run_profile "$model_size" "$context_length" "$mode"
      done
    done
  done
else
  echo "Unsupported NSIGHT_PROFILE_MODE=${PROFILE_MODE}; expected smoke or full." >&2
  exit 2
fi

find "$OUTPUT_DIR" -maxdepth 1 -type f | sort
