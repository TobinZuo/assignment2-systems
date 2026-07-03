#!/usr/bin/env bash
set -euo pipefail

echo "PATH=$PATH"
command -v nsys || true
find /usr /opt /usr/local -maxdepth 6 -type f -name nsys 2>/dev/null | sed -n '1,80p' || true
dpkg -l 2>/dev/null | grep -i -E 'nsight|nsys|cuda' | sed -n '1,120p' || true
apt-cache policy nsight-systems 2>/dev/null | sed -n '1,120p' || true
nvidia-smi -L || true
