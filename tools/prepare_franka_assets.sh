#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f assets/_download.py ]]; then
  echo "Missing assets/_download.py. Restore tracked asset helper first:"
  echo "  git restore assets/_download.py assets/files/50_tasks.gif assets/files/domain_randomization.png"
  exit 1
fi

if [[ -f assets/embodiments/franka-panda/config.yml && -f assets/embodiments/franka-panda/curobo.yml ]]; then
  echo "Franka Panda assets already exist."
else
  echo "Downloading and unpacking RoboTwin assets..."
  bash script/_download_assets.sh
fi

python script/test_render.py

python - <<'PY'
import os
import yaml

cfg_path = "assets/embodiments/franka-panda/config.yml"
curobo_path = "assets/embodiments/franka-panda/curobo.yml"

assert os.path.isfile(cfg_path), f"Missing {cfg_path}"
assert os.path.isfile(curobo_path), f"Missing {curobo_path}"

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

assert cfg.get("dual_arm") is False, "franka-panda config should set dual_arm: false"
assert len(cfg["arm_joints_name"][0]) == 7, "Franka Panda should expose 7 arm joints"

print("OK: Franka Panda assets and config are ready")
PY
