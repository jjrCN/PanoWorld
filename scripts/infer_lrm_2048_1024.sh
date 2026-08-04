#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PANOWORLD_PYTHON}" -m panoworld_lrm.inference --config "${CONFIG:-configs/inference_2048_1024.yaml}" "$@"
