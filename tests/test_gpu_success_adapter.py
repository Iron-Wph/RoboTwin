"""Task-level validation for the first CUDA success-predicate adapter."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np
import sapien
import torch

from bench_real_pi0_rollout import load_task_config
from robotwin.envs.vector_env import VectorEnv


class AdjustBottleGpuSuccessAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        config = load_task_config(
            Path(os.environ["RLINF_ROOT"]),
            gpu_sim=True,
            clear_cache_freq=1,
        )
        config["gpu_validate_success_adapter"] = True
        self.env = VectorEnv(config, n_envs=1, env_seeds=[10_000])
        self.slot = self.env.envs[0].active_slot
        self.task = self.slot.task
        self.runtime = self.env.gpu_runtime

    def tearDown(self) -> None:
        self.env.close()

    def _set_bottle_pose(self, x: float, z: float) -> None:
        component = self.task.bottle.actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        self.assertIsNotNone(component)
        data = self.runtime.physx.cuda_rigid_dynamic_data.torch()
        data[component.gpu_pose_index, :3] = torch.as_tensor(
            [x, -0.1, z], device=data.device, dtype=data.dtype
        )
        data[component.gpu_pose_index, 3:7] = torch.as_tensor(
            [1.0, 0.0, 0.0, 0.0], device=data.device, dtype=data.dtype
        )
        self.runtime.physx.gpu_apply_rigid_dynamic_data()
        self.runtime.physx.sync_poses_gpu_to_cpu()

    def test_gpu_predicate_matches_cpu_for_success_and_failure(self) -> None:
        initial_state = self.runtime.capture_env_state(self.slot.env_id)
        try:
            success_x = -1.0 if self.task.qpose_tag == 0 else 1.0
            self._set_bottle_pose(success_x, 1.2)
            self.assertTrue(self.task.check_success())
            self.assertTrue(bool(self.task.check_success_gpu().item()))

            self._set_bottle_pose(0.0, 0.2)
            self.assertFalse(self.task.check_success())
            self.assertFalse(bool(self.task.check_success_gpu().item()))
        finally:
            self.runtime.restore_env_state(initial_state)

    def test_live_gpu_control_ticks_validate_the_adapter(self) -> None:
        # A single zero action chunk runs the normal TOPP/control path.  The
        # validation hook compares CPU and CUDA success on every local tick.
        actions = np.zeros((1, 1, 14), dtype=np.float32)
        self.env.step(actions)
        self.assertGreater(self.task._gpu_success_validation_ticks, 0)

    @staticmethod
    def _run_one_chunk(chunk_actions: np.ndarray, *, defer_cpu_sync: bool):
        config = load_task_config(
            Path(os.environ["RLINF_ROOT"]),
            gpu_sim=True,
            clear_cache_freq=1,
        )
        config["gpu_defer_cpu_sync"] = defer_cpu_sync
        env = VectorEnv(config, n_envs=1, env_seeds=[10_000])
        try:
            result = env.step(chunk_actions[None, ...])
            state = env.gpu_runtime.capture_env_state(env.envs[0].runtime_env_id)
            return result[1:4], state
        finally:
            env.close()

    def test_one_chunk_deferred_sync_matches_precise_physics_state(self) -> None:
        # Representative multi-point qpos/gripper chunk.  It exercises the
        # same MPLib TOPP, gripper interpolation, passive-force, and physics
        # paths as an action chunk emitted by a policy, without relying on a
        # stochastic model forward in a unit test.
        chunk_actions = np.asarray(
            [
                [0.05] * 6 + [0.85] + [-0.05] * 6 + [0.85],
                [0.10] * 6 + [0.55] + [-0.10] * 6 + [0.55],
                [0.15] * 6 + [0.20] + [-0.15] * 6 + [0.20],
                [0.10] * 6 + [0.75] + [-0.10] * 6 + [0.75],
            ],
            dtype=np.float32,
        )
        precise_result, precise_state = self._run_one_chunk(
            chunk_actions, defer_cpu_sync=False
        )
        deferred_result, deferred_state = self._run_one_chunk(
            chunk_actions, defer_cpu_sync=True
        )

        for precise_value, deferred_value in zip(precise_result, deferred_result):
            np.testing.assert_array_equal(
                np.asarray(precise_value), np.asarray(deferred_value)
            )
        torch.testing.assert_close(
            precise_state.rigid_dynamic_data,
            deferred_state.rigid_dynamic_data,
            rtol=2e-4,
            atol=2e-4,
        )
        torch.testing.assert_close(
            precise_state.articulation_qpos,
            deferred_state.articulation_qpos,
            rtol=2e-4,
            atol=2e-4,
        )
        torch.testing.assert_close(
            precise_state.articulation_qvel,
            deferred_state.articulation_qvel,
            rtol=2e-4,
            atol=2e-4,
        )
        torch.testing.assert_close(
            precise_state.articulation_target_qpos,
            deferred_state.articulation_target_qpos,
            rtol=2e-5,
            atol=2e-5,
        )


if __name__ == "__main__":
    unittest.main()
