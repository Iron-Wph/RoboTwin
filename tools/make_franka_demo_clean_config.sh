#!/usr/bin/env bash
set -euo pipefail

TASK_CONFIG="${1:-franka_demo_clean_300}"
EPISODE_NUM="${2:-300}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! "${TASK_CONFIG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid task config name: ${TASK_CONFIG}"
  exit 2
fi

if [[ ! "${EPISODE_NUM}" =~ ^[0-9]+$ ]] || [[ "${EPISODE_NUM}" -lt 1 ]]; then
  echo "EPISODE_NUM must be a positive integer, got: ${EPISODE_NUM}"
  exit 2
fi

mkdir -p task_config

cat > "task_config/${TASK_CONFIG}.yml" <<YAML
render_freq: 0
episode_num: ${EPISODE_NUM}
use_seed: false
save_freq: 15
embodiment: [franka-panda]
language_num: 100
domain_randomization:
  random_background: false
  cluttered_table: false
  clean_background_rate: 1
  random_head_camera_dis: 0
  random_table_height: 0
  random_light: false
  crazy_random_light_rate: 0
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
  static_camera_list:
    - name: head_camera
      type: D435
      position: [0.0, 0.85, 1.55]
      forward: [0.0, -0.82, -0.57]
      left: [1.0, 0.0, 0.0]
data_type:
  rgb: true
  third_view: false
  depth: false
  pointcloud: false
  observer: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
pcd_down_sample_num: 1024
pcd_crop: true
save_path: ./data
clear_cache_freq: 5
collect_data: true
eval_video_log: false
YAML

echo "Wrote task_config/${TASK_CONFIG}.yml with episode_num=${EPISODE_NUM}"
