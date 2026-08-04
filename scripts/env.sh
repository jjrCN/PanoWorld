#!/usr/bin/env bash
# Shared runtime selection for every PanoWorld entry point.

if [[ -z "${PANOWORLD_ROOT:-}" ]]; then
  PANOWORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PANOWORLD_ROOT

if [[ -z "${PANOWORLD_PYTHON:-}" ]]; then
  if [[ -x "${PANOWORLD_ROOT}/.venv/bin/python" ]]; then
    PANOWORLD_PYTHON="${PANOWORLD_ROOT}/.venv/bin/python"
  else
    PANOWORLD_PYTHON="$(command -v python)"
  fi
fi
export PANOWORLD_PYTHON

export PYTHONPATH="${PANOWORLD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
