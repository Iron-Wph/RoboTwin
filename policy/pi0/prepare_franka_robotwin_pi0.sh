#!/bin/bash
set -euo pipefail

task_name=${1:?Usage: bash policy/pi0/prepare_franka_robotwin_pi0.sh <task_name> <task_config> <expert_data_num> [repo_id] [dataset_root] [config_name]}
task_config=${2:?Usage: bash policy/pi0/prepare_franka_robotwin_pi0.sh <task_name> <task_config> <expert_data_num> [repo_id] [dataset_root] [config_name]}
expert_data_num=${3:?Usage: bash policy/pi0/prepare_franka_robotwin_pi0.sh <task_name> <task_config> <expert_data_num> [repo_id] [dataset_root] [config_name]}
repo_id=${4:-robotwin/${task_name}_${task_config}}
dataset_root=${5:-/mnt/public2/wph/codes/develop_async/RoboTwin_main_official/data}
config_name=${6:-pi0_base_franka_robotwin_full}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/mnt/public2/wph/.cache/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/public2/wph/.cache/transformers_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/mnt/public2/wph/.cache/xdg_cache_home}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-"$dataset_root"}
export PYTHONPATH="$script_dir/src:${PYTHONPATH:-}"
export OPENPI_ROBOTWIN_FRANKA_REPO_ID="$repo_id"

if [ -n "${FFMPEG_CONDA:-}" ]; then
    export PATH="$FFMPEG_CONDA/bin:$PATH"
    export LD_LIBRARY_PATH="$FFMPEG_CONDA/lib:${LD_LIBRARY_PATH:-}"
elif [ -d /mnt/public2/wph/envs/miniconda3/envs/RoboTwin ]; then
    export FFMPEG_CONDA=/mnt/public2/wph/envs/miniconda3/envs/RoboTwin
    export PATH="$FFMPEG_CONDA/bin:$PATH"
    export LD_LIBRARY_PATH="$FFMPEG_CONDA/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p processed_data training_data

echo "[INFO] process single-arm Franka RoboTwin data ..."
bash process_franka_data_pi0.sh "$task_name" "$task_config" "$expert_data_num" "$dataset_root"

converted_hdf5_name="${task_name}-${task_config}-${expert_data_num}"
src_dir="./processed_data/${converted_hdf5_name}"
dst_dir="./training_data/${converted_hdf5_name}"

if [ ! -d "$src_dir" ]; then
    echo "[ERROR] missing processed data directory: $src_dir"
    exit 1
fi

rm -rf "$dst_dir"
cp -r "$src_dir" "$dst_dir"
echo "[INFO] copied processed data to: $dst_dir"

echo "[INFO] generate LeRobot dataset: $repo_id"
bash generate_franka.sh "$dst_dir" "$repo_id"

echo "[INFO] compute norm_stats with config: $config_name"
python scripts/compute_norm_stats.py --config-name "$config_name"

echo "[INFO] done. repo_id=$repo_id config=$config_name"
