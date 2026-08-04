#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${CODE_DIR}/scripts/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CODE_DIR}/scripts/env.sh"
fi
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MANIFEST="${MANIFEST:-${CODE_DIR}/data_list/data_demo_data/inference_2d_generator.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/outputs/2d_generator_demo}"

PYTHON_BIN="${PYTHON_BIN:-${PANOWORLD_PYTHON:-python}}"
BASE_MODEL="${BASE_MODEL:-${CODE_DIR}/model_ckpt/Qwen-Image-Edit-2509}"
PANOWORLD_LORA="${PANOWORLD_LORA:-${CODE_DIR}/model_ckpt/pytorch_lora_weights.safetensors}"
LIGHTNING_LORA="${LIGHTNING_LORA:-${CODE_DIR}/model_ckpt/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors}"
CONTROL_MODEL_ROOT="${CONTROL_MODEL_ROOT:-${CODE_DIR}/panoworld_2d_generator/models/control_models}"
INPUTS_UNPADDED="${INPUTS_UNPADDED:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}"
PRECISION="${PRECISION:-bf16}"

EXTRA_ARGS=()
if [[ "${INPUTS_UNPADDED}" != "0" && "${INPUTS_UNPADDED}" != "false" && "${INPUTS_UNPADDED}" != "False" ]]; then
  EXTRA_ARGS+=(--inputs-unpadded)
fi

"${PYTHON_BIN}" -m panoworld_2d_generator.infer \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-model "${BASE_MODEL}" \
  --panoworld-lora "${PANOWORLD_LORA}" \
  --lightning-lora "${LIGHTNING_LORA}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --precision "${PRECISION}" \
  --control-model-root "${CONTROL_MODEL_ROOT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
