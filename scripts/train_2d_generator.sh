#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
export PYTHON_BIN="${PYTHON_BIN:-${PANOWORLD_PYTHON}}"

exec bash panoworld_2d_generator/scripts/train.sh "$@"
