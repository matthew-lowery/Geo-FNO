#!/bin/bash
set -euo pipefail
ME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TRAIN_PYTHON:-/u/mlowery/.conda/envs/gnot/bin/python}"
exec "$PYTHON" "$ME_DIR/train_turb_remaining.py" "$@"
