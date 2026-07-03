#!/usr/bin/env bash
set -euo pipefail

if [[ "${SKIP_TESTS:-0}" == "1" ]]; then
    echo "Skipping local pytest because SKIP_TESTS=1"
else
    uv run pytest -v ./tests --junitxml=test_results.xml || true
    echo "Done running tests"
fi

# Set the name of the output tar.gz file
output_file="cs336-spring2024-assignment-2-submission.zip"
rm "$output_file" || true

# Compress all files in the current directory into a single zip file.
if command -v zip >/dev/null 2>&1; then
    zip -r "$output_file" . \
        -x "$output_file" \
        -x '*egg-info*' \
        -x '*mypy_cache*' \
        -x '*pytest_cache*' \
        -x '*build*' \
        -x '*ipynb_checkpoints*' \
        -x '*__pycache__*' \
        -x '*.pkl' \
        -x '*.pickle' \
        -x '*.txt' \
        -x '*.log' \
        -x '*.json' \
        -x '*.out' \
        -x '*.err' \
        -x '.git*' \
        -x '.venv/*' \
        -x 'memory_profiles/*' \
        -x 'nsight_profiles/*' \
        -x 'nsight_profiles_b2/*' \
        -x 'diagrams/*' \
        -x '*.nsys-rep' \
        -x '*.sqlite' \
        -x '*.jsonl' \
        -x '*.bin' \
        -x '*.pt' \
        -x '*.pth'
else
    python - "$output_file" <<'PY'
from __future__ import annotations

import fnmatch
import os
import stat
import sys
import zipfile
from pathlib import Path

output_file = sys.argv[1]
exclude_patterns = [
    output_file,
    "*egg-info*",
    "*mypy_cache*",
    "*pytest_cache*",
    "*build*",
    "*ipynb_checkpoints*",
    "*__pycache__*",
    "*.pkl",
    "*.pickle",
    "*.txt",
    "*.log",
    "*.json",
    "*.out",
    "*.err",
    ".git*",
    ".venv/*",
    "memory_profiles/*",
    "nsight_profiles/*",
    "nsight_profiles_b2/*",
    "diagrams/*",
    "*.nsys-rep",
    "*.sqlite",
    "*.jsonl",
    "*.bin",
    "*.pt",
    "*.pth",
]


def should_exclude(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in exclude_patterns)


with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk("."):
        rel_root = Path(root).as_posix().removeprefix("./")
        dirs[:] = [
            d
            for d in dirs
            if not should_exclude(f"{rel_root}/{d}/*".lstrip("./"))
        ]
        for name in files:
            path = Path(root) / name
            arcname = path.as_posix().removeprefix("./")
            if should_exclude(arcname):
                continue
            info = zipfile.ZipInfo.from_file(path, arcname)
            if os.access(path, os.X_OK):
                info.external_attr = (stat.S_IFREG | 0o755) << 16
            with open(path, "rb") as f:
                zf.writestr(info, f.read(), compress_type=zipfile.ZIP_DEFLATED)
PY
fi

echo "All files have been compressed into $output_file"
