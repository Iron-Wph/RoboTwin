"""Focused tests for the task-independent shared SAPIEN GPU primitive."""

from __future__ import annotations

import threading
import unittest

import sapien
import torch

from robotwin.envs.gpu_runtime import SharedGpuRuntime


class SharedGpuRuntimeTest(unittest.TestCase):
    def _build_slot(self, env_id: int) -> None:
        scene = self.runtime.scenes[env_id]
        actor_builder = scene.create_actor_builder()
        actor_builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
        actor_builder.set_initial_pose(sapien.Pose([0.0, 0.0, 1.0 + env_id]))
        actor_builder.build(name=f"box-{env_id}")

        articulation_builder = scene.create_articulation_builder()
        root = articulation_builder.create_link_builder()
        root.add_box_collision(half_size=[0.1, 0.1, 0.1])
        link = articulation_builder.create_link_builder(root)
        link.add_box_collision(half_size=[0.1, 0.1, 0.1])
        link.set_joint_name(f"joint-{env_id}")
        link.set_joint_properties("revolute", [[-1.0, 1.0]], sapien.Pose(), sapien.Pose(), 0, 0)
        articulation = articulation_builder.build()
        articulation.get_active_joints()[0].set_drive_property(1_000, 200)
        self.runtime.record_env_topology(env_id)

    def setUp(self) -> None:
        self.runtime = SharedGpuRuntime(num_envs=2, enable_render=False)
        for env_id in range(self.runtime.num_envs):
            self._build_slot(env_id)
        self.runtime.initialize()

    def test_scene_offsets_are_unique(self) -> None:
        offsets = [tuple(self.runtime.physx.get_scene_offset(scene)) for scene in self.runtime.scenes]
        self.assertEqual(len(set(offsets)), 2)

    def test_partial_restore_keeps_another_environment_changed(self) -> None:
        env0 = self.runtime.capture_env_state(0)
        env1 = self.runtime.capture_env_state(1)

        rigid = self.runtime.physx.cuda_rigid_dynamic_data.torch()
        qpos = self.runtime.physx.cuda_articulation_qpos.torch()
        targets = self.runtime.physx.cuda_articulation_target_qpos.torch()
        rigid[env0.rigid_dynamic_rows, 0] += 1.0
        rigid[env1.rigid_dynamic_rows, 0] += 2.0
        qpos[env0.articulation_rows, 0] += 0.2
        qpos[env1.articulation_rows, 0] += 0.4
        targets[env0.articulation_rows, 0] += 0.3
        targets[env1.articulation_rows, 0] += 0.6
        self.runtime.physx.gpu_apply_rigid_dynamic_data()
        self.runtime.physx.gpu_apply_articulation_qpos()
        self.runtime.physx.gpu_apply_articulation_target_position()

        env1_before_restore = self.runtime.capture_env_state(1)
        self.runtime.restore_env_state(env0)
        self.runtime.assert_partial_restore_isolated(env0, (env1_before_restore,))

        rigid_after = self.runtime.physx.cuda_rigid_dynamic_data.torch()
        qpos_after = self.runtime.physx.cuda_articulation_qpos.torch()
        targets_after = self.runtime.physx.cuda_articulation_target_qpos.torch()
        torch.testing.assert_close(rigid_after[env0.rigid_dynamic_rows], env0.rigid_dynamic_data)
        torch.testing.assert_close(qpos_after[env0.articulation_rows], env0.articulation_qpos)
        torch.testing.assert_close(targets_after[env0.articulation_rows], env0.articulation_target_qpos)
        self.assertFalse(torch.equal(rigid_after[env1.rigid_dynamic_rows], env1.rigid_dynamic_data))
        self.assertFalse(torch.equal(qpos_after[env1.articulation_rows], env1.articulation_qpos))
        self.assertFalse(torch.equal(targets_after[env1.articulation_rows], env1.articulation_target_qpos))

    def test_step_advances_the_shared_system_once(self) -> None:
        self.runtime.step()
        self.assertTrue(self.runtime.initialized)

    def test_gpu_drive_targets_update_only_the_selected_articulation_rows(self) -> None:
        env0 = self.runtime.capture_env_state(0)
        env1 = self.runtime.capture_env_state(1)
        target_qpos = self.runtime.physx.cuda_articulation_target_qpos.torch()
        target_qvel = self.runtime.physx.cuda_articulation_target_qvel.torch()
        target_qpos[env0.articulation_rows, 0] = 0.2
        target_qvel[env0.articulation_rows, 0] = 1.0
        self.runtime.apply_articulation_controls()

        target_after = self.runtime.physx.cuda_articulation_target_qpos.torch()
        velocity_after = self.runtime.physx.cuda_articulation_target_qvel.torch()
        self.assertAlmostEqual(target_after[env0.articulation_rows, 0].item(), 0.2)
        self.assertAlmostEqual(velocity_after[env0.articulation_rows, 0].item(), 1.0)
        self.assertAlmostEqual(target_after[env1.articulation_rows, 0].item(), 0.0)

    def test_queued_controls_flush_at_the_shared_physics_barrier(self) -> None:
        env0 = self.runtime.capture_env_state(0)
        env1 = self.runtime.capture_env_state(1)
        row0 = int(env0.articulation_rows[0].item())
        row1 = int(env1.articulation_rows[0].item())
        self.runtime.begin_batched_step([0, 1])

        def run_slot(env_id: int, row: int, target: float) -> None:
            self.assertTrue(
                self.runtime.queue_articulation_targets(row, (0,), (target,), (1.0,))
            )
            self.assertTrue(self.runtime.queue_articulation_force(row, (target * 2,)))
            self.runtime.step_slot(env_id)
            self.runtime.finish_batched_slot(env_id)

        threads = [
            threading.Thread(target=run_slot, args=(0, row0, 0.2)),
            threading.Thread(target=run_slot, args=(1, row1, 0.4)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive(), "GPU slot scheduler deadlocked")
        self.runtime.end_batched_step()

        target_qpos = self.runtime.physx.cuda_articulation_target_qpos.torch()
        target_qvel = self.runtime.physx.cuda_articulation_target_qvel.torch()
        qf = self.runtime.physx.cuda_articulation_qf.torch()
        self.assertAlmostEqual(target_qpos[row0, 0].item(), 0.2)
        self.assertAlmostEqual(target_qpos[row1, 0].item(), 0.4)
        self.assertAlmostEqual(target_qvel[row0, 0].item(), 1.0)
        self.assertAlmostEqual(target_qvel[row1, 0].item(), 1.0)
        self.assertAlmostEqual(qf[row0, 0].item(), 0.4)
        self.assertAlmostEqual(qf[row1, 0].item(), 0.8)

    def test_two_slots_synchronize_to_one_shared_physics_step(self) -> None:
        initial_count = self.runtime.batched_step_count
        self.runtime.begin_batched_step([0, 1])

        def run_slot(env_id: int) -> None:
            self.runtime.step_slot(env_id)
            self.runtime.finish_batched_slot(env_id)

        threads = [threading.Thread(target=run_slot, args=(env_id,)) for env_id in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive(), "GPU slot scheduler deadlocked")
        self.runtime.end_batched_step()
        self.assertEqual(self.runtime.batched_step_count, initial_count + 1)


class DeferredCpuSyncRuntimeTest(SharedGpuRuntimeTest):
    """The opt-in path has one pose copy at the action-chunk boundary."""

    def setUp(self) -> None:
        self.runtime = SharedGpuRuntime(
            num_envs=2,
            enable_render=False,
            profile=True,
            defer_cpu_sync=True,
        )
        for env_id in range(self.runtime.num_envs):
            self._build_slot(env_id)
        self.runtime.initialize()

    def test_deferred_sync_happens_once_after_multiple_ticks(self) -> None:
        self.runtime.begin_batched_step([0, 1])

        def run_slot(env_id: int) -> None:
            self.runtime.step_slot(env_id)
            self.runtime.step_slot(env_id)
            self.runtime.finish_batched_slot(env_id)
            self.runtime.wait_for_cpu_sync()

        threads = [threading.Thread(target=run_slot, args=(env_id,)) for env_id in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive(), "deferred GPU slot scheduler deadlocked")
        self.runtime.end_batched_step()

        profile = self.runtime.profile_snapshot()
        self.assertEqual(profile["physics_steps"], 2)
        self.assertEqual(profile["physics_sync_calls"], 1)

if __name__ == "__main__":
    unittest.main()
