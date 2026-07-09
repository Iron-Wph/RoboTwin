#!/bin/bash
set -euo pipefail

data_dir=${1:?Usage: bash generate_franka.sh <processed_data_dir> <repo_id>}
repo_id=${2:?Usage: bash generate_franka.sh <processed_data_dir> <repo_id>}

uv run examples/aloha_real/convert_franka_data_to_lerobot_robotwin.py --raw_dir "$data_dir" --repo_id "$repo_id"
