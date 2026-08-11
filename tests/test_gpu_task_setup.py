"""Integration test: build ordinary RoboTwin task slots before final GPU init."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest

import yaml
import torch

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from robotwin.envs.gpu_runtime import SharedGpuRuntime
from robotwin.envs.vector_env import class_decorator


def load_task_args(rlinf_root: Path) -> dict:
    path = rlinf_root / "examples/embodiment/config/env/robotwin_place_empty_cup.yaml"
    with path.open(encoding="utf-8") as file:
        args = copy.deepcopy(yaml.safe_load(file)["task_config"])
    args.update(
        save_path=tempfile.mkdtemp(prefix="robotwin-gpu-task-"),
        save_freq=None,
        eval_video_log=False,
        render_freq=0,
        eval_mode=True,
    )
    assets_path = Path(os.environ["ASSETS_PATH"])
    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", encoding="utf-8") as file:
        embodiments = yaml.safe_load(file)
    embodiment = args["embodiment"]
    args["left_robot_file"] = str(assets_path / embodiments[embodiment[0]]["file_path"])
    args["right_robot_file"] = str(assets_path / embodiments[embodiment[1]]["file_path"])
    args["embodiment_dis"] = embodiment[2]
    args["dual_arm_embodied"] = False
    for side in ("left", "right"):
        with open(Path(args[f"{side}_robot_file"]) / "config.yml", encoding="utf-8") as file:
            args[f"{side}_embodiment_config"] = yaml.safe_load(file)
    return args


class SharedGpuTaskSetupTest(unittest.TestCase):
    def test_two_task_slots_construct_and_finalize(self) -> None:
        args = load_task_args(Path(os.environ["RLINF_ROOT"]))
        runtime = SharedGpuRuntime(num_envs=2, enable_render=True)
        tasks = []
        for env_id, seed in enumerate((10000, 10001)):
            task = class_decorator(args["task_name"])
            task.setup_demo(
                now_ep_num=seed,
                seed=seed,
                instruction="test instruction",
                _shared_scene=runtime.scenes[env_id],
                _gpu_runtime=runtime,
                **copy.deepcopy(args),
            )
            runtime.record_env_topology(env_id)
            tasks.append(task)

        runtime.initialize()
        for task in tasks:
            task.finalize_gpu_construction()
        states = [runtime.capture_env_state(env_id) for env_id in range(2)]
        self.assertTrue(all(state.rigid_dynamic_rows.numel() > 0 for state in states))
        self.assertTrue(all(state.articulation_rows.numel() == 2 for state in states))
        self.assertNotEqual(
            tuple(states[0].rigid_dynamic_rows.tolist()),
            tuple(states[1].rigid_dynamic_rows.tolist()),
        )
        rigid_data = runtime.physx.cuda_rigid_dynamic_data.torch()
        for task, state in zip(tasks, states):
            cup_position = torch.as_tensor(task.cup.get_pose().p, device=rigid_data.device)
            self.assertTrue(
                torch.any(torch.all(torch.isclose(rigid_data[state.rigid_dynamic_rows, :3], cup_position), dim=1))
            )


if __name__ == "__main__":
    unittest.main()
