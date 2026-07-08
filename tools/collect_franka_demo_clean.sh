#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash tools/collect_franka_demo_clean.sh <task_name> [episode_num=300] [gpu_id=0] [task_config=franka_demo_clean_<episode_num>]"
  echo "Example: bash tools/collect_franka_demo_clean.sh stack_blocks_two 300 0"
  echo "Small test: bash tools/collect_franka_demo_clean.sh stack_blocks_two 3 0 franka_demo_clean_test"
  exit 2
fi

TASK_NAME="$1"
EPISODE_NUM="${2:-300}"
GPU_ID="${3:-0}"
TASK_CONFIG="${4:-franka_demo_clean_${EPISODE_NUM}}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f assets/embodiments/franka-panda/config.yml ]]; then
  echo "Franka Panda assets are missing. Run this first:"
  echo "  bash tools/prepare_franka_assets.sh"
  exit 1
fi

bash tools/make_franka_demo_clean_config.sh "${TASK_CONFIG}" "${EPISODE_NUM}"
python tools/fix_embodiment_asset_paths.py

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py "${TASK_NAME}" "${TASK_CONFIG}"

python tools/check_franka_dataset.py "data/${TASK_NAME}/${TASK_CONFIG}/data" "${EPISODE_NUM}"
