#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${1:-place_empty_cup}"
TASK_CONFIG="${2:-franka_smoke}"
GPU_ID="${3:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f assets/embodiments/franka-panda/config.yml ]]; then
  echo "Franka Panda assets are missing. Run this first:"
  echo "  bash tools/prepare_franka_assets.sh"
  exit 1
fi

if [[ "${TASK_CONFIG}" == "franka_smoke" || ! -f "task_config/${TASK_CONFIG}.yml" ]]; then
  bash tools/make_franka_smoke_config.sh
fi

python tools/fix_embodiment_asset_paths.py

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py "${TASK_NAME}" "${TASK_CONFIG}"

python tools/check_franka_hdf5.py "data/${TASK_NAME}/${TASK_CONFIG}/data/episode0.hdf5"
