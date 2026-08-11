"""Integration test for state-isolated GPU partial reset through VectorEnv."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import unittest

import numpy as np

from robotwin.envs.vector_env import VectorEnv
from test_vector_env_baseline import load_task_config


class GpuVectorPartialResetTest(unittest.TestCase):
    def test_check_seeds_keeps_the_existing_gpu_environment_interface(self) -> None:
        config = load_task_config(Path(os.environ["RLINF_ROOT"]))
        config["gpu_sim"] = True
        config["gpu_validate_partial_reset"] = True
        env = VectorEnv(config, n_envs=1, env_seeds=[10000])
        try:
            physical_id = env.envs[0].runtime_env_id
            before = env.gpu_runtime.capture_env_state(physical_id)
            result = env.check_seeds([10000])[0]

            self.assertTrue(result["setup_demo_success"])
            self.assertIsInstance(result["play_once_success"], bool)
            self.assertGreaterEqual(result["cost_time"], 0)
            after = env.gpu_runtime.capture_env_state(physical_id)
            np.testing.assert_allclose(
                after.rigid_dynamic_data.cpu().numpy(),
                before.rigid_dynamic_data.cpu().numpy(),
                atol=1e-6,
            )
        finally:
            env.close()

    def test_resetting_one_slot_restores_only_that_slot(self) -> None:
        config = load_task_config(Path(os.environ["RLINF_ROOT"]))
        config["gpu_sim"] = True
        env = VectorEnv(config, n_envs=2, env_seeds=[10000, 10001])
        try:
            slot0_id = env.envs[0].runtime_env_id
            slot1_id = env.envs[1].runtime_env_id
            initial_slot0 = env.gpu_runtime.capture_env_state(slot0_id)
            initial_slot1 = env.gpu_runtime.capture_env_state(slot1_id)
            observations = env.get_obs()
            actions = np.stack(
                [np.asarray(observation["state"], dtype=np.float32) for observation in observations]
            )[:, None, :]
            actions[0, 0, 0] += 0.02
            scheduled_steps_before = env.gpu_runtime.batched_step_count
            env.step(actions)
            self.assertGreater(env.gpu_runtime.batched_step_count, scheduled_steps_before)
            slot1_before_reset = env.gpu_runtime.capture_env_state(slot1_id)

            env.reset(env_idx=[0], env_seeds=[10000])
            slot0_after_reset = env.gpu_runtime.capture_env_state(slot0_id)
            slot1_after_reset = env.gpu_runtime.capture_env_state(slot1_id)

            np.testing.assert_allclose(
                slot0_after_reset.rigid_dynamic_data.cpu().numpy(),
                initial_slot0.rigid_dynamic_data.cpu().numpy(),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                slot1_after_reset.rigid_dynamic_data.cpu().numpy(),
                slot1_before_reset.rigid_dynamic_data.cpu().numpy(),
                atol=1e-6,
            )
            self.assertEqual(
                slot1_after_reset.articulation_target_qpos.cpu().tolist(),
                slot1_before_reset.articulation_target_qpos.cpu().tolist(),
            )
            self.assertEqual(
                initial_slot1.rigid_dynamic_rows.cpu().tolist(),
                slot1_after_reset.rigid_dynamic_rows.cpu().tolist(),
            )
        finally:
            env.close()

    def test_reset_can_switch_to_a_prebuilt_seed_with_its_own_visual_scene(self) -> None:
        config = load_task_config(Path(os.environ["RLINF_ROOT"]))
        config["gpu_sim"] = True
        config["gpu_reset_seed_pool"] = [10001]
        env = VectorEnv(config, n_envs=1, env_seeds=[10000])
        try:
            original_physical_id = env.envs[0].runtime_env_id
            original_state = env.gpu_runtime.capture_env_state(original_physical_id)
            env.reset(env_idx=[0], env_seeds=[10001])

            self.assertEqual(env.envs[0].active_slot.env_seed, 10001)
            self.assertNotEqual(env.envs[0].runtime_env_id, original_physical_id)
            texture_info = env.envs[0].active_slot.task.info["texture_info"]
            self.assertIn("wall_texture", texture_info)
            self.assertIn("table_texture", texture_info)
            observation = env.get_obs()[0]
            self.assertEqual(observation["full_image"].ndim, 3)

            original_after_switch = env.gpu_runtime.capture_env_state(original_physical_id)
            np.testing.assert_allclose(
                original_after_switch.rigid_dynamic_data.cpu().numpy(),
                original_state.rigid_dynamic_data.cpu().numpy(),
                atol=1e-6,
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
