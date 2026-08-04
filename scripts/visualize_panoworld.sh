#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

VIEWPOINTS_DIR="${VIEWPOINTS_DIR:-}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  VIEWPOINTS_DIR="$1"
  shift
fi
VIEWPOINTS_DIR="${VIEWPOINTS_DIR:-examples/full_pipeline_demo_datas/scene0000/viewpoints}"

PYTHON="${PYTHON:-${PANOWORLD_PYTHON}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"
IMAGE_NAME="${IMAGE_NAME:-panoImage_2048.png}"
START_IMAGE_NAME="${START_IMAGE_NAME:-panoImage_2048_franch.png}"
PUBLIC_HOST="${PUBLIC_HOST:-}"

ARGS=(
  --viewpoints "${VIEWPOINTS_DIR}"
  --host "${HOST}"
  --port "${PORT}"
  --image-name "${IMAGE_NAME}"
  --start-image-name "${START_IMAGE_NAME}"
)

if [[ -n "${PUBLIC_HOST}" ]]; then
  ARGS+=(--public-host "${PUBLIC_HOST}")
fi

"${PYTHON}" tools/panoworld_viewer_server.py "${ARGS[@]}" "$@"
