#!/usr/bin/env bash
set -Eeuo pipefail

CONTROL_MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for command_name in git curl; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command is missing: ${command_name}" >&2
    exit 1
  fi
done

clone_repo() {
  local url="$1"
  local destination="$2"
  local branch="${3:-}"

  if [ -d "${destination}/.git" ]; then
    echo "[code] exists: ${destination}"
    return
  fi
  if [ -e "${destination}" ] && [ -n "$(find "${destination}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "Path exists but is not an empty Git repository directory: ${destination}" >&2
    exit 1
  fi

  if [ -n "${branch}" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "${branch}" "${url}" "${destination}"
  else
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${url}" "${destination}"
  fi
}

download_file() {
  local url="$1"
  local destination="$2"
  local partial="${destination}.part"

  if [ -s "${destination}" ]; then
    echo "[weight] exists: ${destination}"
    return
  fi

  mkdir -p "$(dirname "${destination}")"
  curl -L --fail --show-error --retry 8 --retry-delay 5 --continue-at - \
    --output "${partial}" "${url}"
  mv "${partial}" "${destination}"
}

mkdir -p "${CONTROL_MODELS_DIR}/panosamic" "${CONTROL_MODELS_DIR}/moge"

clone_repo \
  "https://github.com/dfki-av/PanoSAMic.git" \
  "${CONTROL_MODELS_DIR}/panosamic/PanoSAMic"

clone_repo \
  "https://github.com/microsoft/MoGe.git" \
  "${CONTROL_MODELS_DIR}/moge/MoGe"

clone_repo \
  "https://github.com/open-mmlab/mmdetection.git" \
  "${CONTROL_MODELS_DIR}/mmdetection" \
  "v3.3.0"

clone_repo \
  "https://github.com/cocodataset/panopticapi.git" \
  "${CONTROL_MODELS_DIR}/mmdetection/panopticapi"

cp "${CONTROL_MODELS_DIR}/panosamic/PanoSAMic/config/config_stanford2d3ds_dv.json" \
  "${CONTROL_MODELS_DIR}/panosamic/config_stanford2d3ds_dv.json"

download_file \
  "https://huggingface.co/dfki-av/PanoSAMic/resolve/main/stanford2d3ds-vith-rgb-fold1/model.safetensors?download=true" \
  "${CONTROL_MODELS_DIR}/panosamic/stanford2d3ds-vith-rgb-fold1/model.safetensors"

download_file \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
  "${CONTROL_MODELS_DIR}/panosamic/sam_vit_h_4b8939.pth"

download_file \
  "https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt?download=true" \
  "${CONTROL_MODELS_DIR}/moge/moge-2-vitl-normal/model.pt"

MASK2FORMER_FILE="mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic_20220407_104949-82f8d28d.pth"
download_file \
  "https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic/${MASK2FORMER_FILE}" \
  "${CONTROL_MODELS_DIR}/mmdetection/${MASK2FORMER_FILE}"

ln -sfn "${MASK2FORMER_FILE}" \
  "${CONTROL_MODELS_DIR}/mmdetection/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.pth"

echo "Control models are ready: ${CONTROL_MODELS_DIR}"
