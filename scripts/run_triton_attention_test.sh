#!/usr/bin/env bash
set -euo pipefail

cd /mlx/users/zuotongbin.tobin/playground/assignment2-systems

LOG=triton_attention_test.log
RESULTS=triton_attention_test_results.xml

exec > >(tee "$LOG") 2>&1

export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600

echo "[start] $(date -Is)"
echo "[host] $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.12)"
else
  PYTHON_BIN="3.11"
fi
echo "[python] ${PYTHON_BIN}"

uv venv /tmp/cs336-systems-flash-triton-cu121-venv --seed --clear --python "$PYTHON_BIN"
source /tmp/cs336-systems-flash-triton-cu121-venv/bin/activate

uv pip install --no-progress --index-url https://bytedpypi.byted.org/simple/ \
  pytest einops numpy "triton==3.1.0"
uv pip install --no-progress --index-url https://bytedpypi.byted.org/simple/ \
  "torch==2.5.1"

python - <<'PY'
import torch
import triton

print(
    "torch",
    torch.__version__,
    "cuda",
    torch.cuda.is_available(),
    "device",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
)
print("triton", triton.__version__)
PY

python -m pytest -q tests/test_attention.py --tb=short --junitxml="$RESULTS"
status=$?

echo "[done] $(date -Is) status=${status}"
exit "$status"
