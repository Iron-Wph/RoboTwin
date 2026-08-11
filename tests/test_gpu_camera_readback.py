"""Pixel-equivalence test for GPU camera readback across all rollout cameras."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np
import torch

from robotwin.envs.vector_env import VectorEnv
from test_vector_env_baseline import load_task_config


class GpuCameraReadbackTest(unittest.TestCase):
    def test_cuda_tensor_observations_stay_on_device(self) -> None:
        config = load_task_config(Path(os.environ["RLINF_ROOT"]))
        config["gpu_sim"] = True
        config["gpu_image_tensor_obs"] = True
        config["camera"]["collect_wrist_camera"] = True
        env = VectorEnv(config, n_envs=1, env_seeds=[10000])
        try:
            env.reset()
            observation = env.get_obs()[0]
            for key in ("full_image", "left_wrist_image", "right_wrist_image"):
                image = observation[key]
                self.assertIsInstance(image, torch.Tensor)
                self.assertTrue(image.is_cuda)
                self.assertEqual(image.dtype, torch.uint8)
                self.assertEqual(image.shape[-1], 3)
        finally:
            env.close()

    def test_cuda_readback_matches_cpu_for_head_and_wrist_cameras(self) -> None:
        config = load_task_config(Path(os.environ["RLINF_ROOT"]))
        config["gpu_sim"] = True
        config["camera"]["collect_wrist_camera"] = True
        env = VectorEnv(config, n_envs=1, env_seeds=[10000])
        try:
            env.reset()
            task = env.envs[0].active_slot.task
            # Read both paths from exactly the same SAPIEN camera frames.  This
            # checks conversion and all three VectorEnv image sources without
            # allowing a new random-light render between comparisons.
            task._update_render(force=True)
            task.cameras.update_picture()
            task.cameras.gpu_image_readback = False
            cpu_rgba = task.cameras.get_rgba()
            task.cameras.gpu_image_readback = True
            cuda_rgba = task.cameras.get_rgba()
            for camera_name in ("head_camera", "left_camera", "right_camera"):
                cpu = cpu_rgba[camera_name]["rgba"]
                cuda = cuda_rgba[camera_name]["rgba"]
                self.assertEqual(cpu.dtype, np.uint8)
                self.assertEqual(cuda.dtype, np.uint8)
                self.assertEqual(cpu.shape, cuda.shape)
                np.testing.assert_array_equal(cuda, cpu)
        finally:
            env.close()

    def test_cross_scene_camera_group_matches_cpu_for_head_and_wrists(self) -> None:
        config = load_task_config(Path(os.environ["RLINF_ROOT"]))
        config["gpu_sim"] = True
        config["camera"]["collect_wrist_camera"] = True
        # Keep the same light while comparing separate camera-capture APIs.
        config["domain_randomization"]["crazy_random_light_rate"] = 0.0
        env = VectorEnv(config, n_envs=2, env_seeds=[10000, 10001])
        try:
            env.reset()
            slots = [sub_env.active_slot for sub_env in env.envs]
            cpu_images = {}
            for slot in slots:
                slot.task._update_render(force=True)
                slot.task.cameras.update_picture()
                slot.task.cameras.gpu_image_readback = False
                cpu_images[slot.env_id] = slot.task.cameras.get_rgba()

            grouped_images = env.gpu_runtime.capture_batched_rgba(
                {
                    slot.env_id: slot.task.cameras.observation_camera_entries()
                    for slot in slots
                }
            )
            for slot in slots:
                for camera_name in ("head_camera", "left_camera", "right_camera"):
                    np.testing.assert_array_equal(
                        grouped_images[slot.env_id][camera_name],
                        cpu_images[slot.env_id][camera_name]["rgba"],
                    )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
