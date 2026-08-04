#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="${CONFIG:-configs/train_lrm_1024_512.yaml}"

"${PANOWORLD_PYTHON}" -m torch.distributed.run \
  --nproc-per-node="${NUM_GPUS}" \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  -m panoworld_lrm.train_lrm --config "${CONFIG}" "$@"
