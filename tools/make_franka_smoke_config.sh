#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p task_config

cat > task_config/franka_smoke.yml <<'YAML'
render_freq: 0
episode_num: 1
use_seed: false
save_freq: 15
embodiment: [franka-panda]
language_num: 1
domain_randomization:
  random_background: false
  cluttered_table: false
  clean_background_rate: 1
  random_head_camera_dis: 0
  random_table_height: 0
  random_light: false
  crazy_random_light_rate: 0
camera:
  collect_wrist_camera: false
  static_camera_list:
    - name: head_camera
      type: D435
      position: [0.0, 0.85, 1.55]
      forward: [0.0, -0.82, -0.57]
      left: [1.0, 0.0, 0.0]
    - name: third_view
      type: D435
      position: [0.85, -1.35, 1.35]
      forward: [-0.42, 0.74, -0.52]
      left: [-0.87, -0.49, 0.0]
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
clear_cache_freq: 1
collect_data: true
eval_video_log: false
YAML

echo "Wrote task_config/franka_smoke.yml"
