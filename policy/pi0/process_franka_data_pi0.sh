#!/bin/bash
set -euo pipefail

task_name=${1:?Usage: bash process_franka_data_pi0.sh <task_name> <setting> <expert_data_num> [dataset_root]}
setting=${2:?Usage: bash process_franka_data_pi0.sh <task_name> <setting> <expert_data_num> [dataset_root]}
expert_data_num=${3:?Usage: bash process_franka_data_pi0.sh <task_name> <setting> <expert_data_num> [dataset_root]}
dataset_root=${4:-../../data}

python scripts/process_franka_data.py "$task_name" "$setting" "$expert_data_num" --dataset-root "$dataset_root"
