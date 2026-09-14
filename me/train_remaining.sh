#!/bin/bash
set -euo pipefail

ME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ME_DIR/train_div.sh" --remaining "$@"
