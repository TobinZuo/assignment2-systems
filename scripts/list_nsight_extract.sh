#!/usr/bin/env bash
set -euo pipefail

find "${HOME}/.local/nsight-systems" -maxdepth 8 -type f | sort | sed -n '1,300p'
