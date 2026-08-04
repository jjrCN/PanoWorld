#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${CODE_DIR}/scripts/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CODE_DIR}/scripts/env.sh"
fi
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-${CODE_DIR}/data_list/data_front3d/train_2d_generator.jsonl}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${DATA_ROOT:-}}"

BASE_MODEL="${BASE_MODEL:-${CODE_DIR}/model_ckpt/Qwen-Image-Edit-2509}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/panoworld_2d_generator}"
PYTHON_BIN="${PYTHON_BIN:-${PANOWORLD_PYTHON:-python}}"
NUM_GPUS="${NUM_GPUS:-1}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-14545}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-150}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
REPORT_TO="${REPORT_TO:-tensorboard}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

echo "Launching PanoWorld training with PYTHON_BIN=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"
if [[ -n "${TRAIN_DATA_ROOT}" ]]; then
  echo "TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT}"
fi
echo "MASTER_ADDR=${MASTER_ADDR} NNODES=${NNODES} NODE_RANK=${NODE_RANK} NUM_GPUS=${NUM_GPUS}"

DATA_ARGS=()
if [[ -n "${TRAIN_DATA_ROOT}" ]]; then
  DATA_ARGS+=(--data_root "${TRAIN_DATA_ROOT}")
fi

CHECKPOINT_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING}" != "0" && "${GRADIENT_CHECKPOINTING}" != "false" && "${GRADIENT_CHECKPOINTING}" != "False" ]]; then
  CHECKPOINT_ARGS+=(--gradient_checkpointing)
fi

# Keep the released recipe explicit while avoiding internal experiment knobs.
RECIPE_ARGS=(
  --inputs_unpadded
  --curriculum_stage1_steps 0
  --stage2_style_end_prob 1.0
  --stage2_style_warmup_steps 0
  --control_latent_dropout_prob 0
  --style_image_flip_prob 0
  --style_image_rotation_prob 0
  --style_latent_dropout_prob 0
  --style_latent_spatial_mask_prob 0
)

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc-per-node="${NUM_GPUS}" \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --module panoworld_2d_generator.train \
  --pretrained_model_name_or_path "${BASE_MODEL}" \
  --train_manifest "${TRAIN_MANIFEST}" \
  "${DATA_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --report_to "${REPORT_TO}" \
  --lr_scheduler cosine \
  --lr_warmup_steps "${LR_WARMUP_STEPS}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --dataloader_num_workers "${NUM_WORKERS}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS}" \
  --rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${RECIPE_ARGS[@]}" \
  "$@"
