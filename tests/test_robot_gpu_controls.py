"""Unit tests for RoboTwin's GPU joint-target adapter."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from envs.robot.robot import Robot


class FakeCudaArray:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def torch(self) -> torch.Tensor:
        return self.tensor


class FakeJoint:
    dof = 1

    def __init__(self, name: str, target: float = 0.0) -> None:
        self.name = name
        self.target = target
        self.velocity = 0.0

    def get_name(self) -> str:
        return self.name

    def get_drive_target(self):
        return np.array([self.target])

    def set_drive_target(self, value: float) -> None:
        self.target = value

    def set_drive_velocity_target(self, value: float) -> None:
        self.velocity = value


class FakeEntity:
    def __init__(self, gpu_index: int, joints: list[FakeJoint]) -> None:
        self.gpu_index = gpu_index
        self._joints = joints
        self.dof = len(joints)
        self.applied_qf = None
        self.passive_force_calls = 0

    def get_active_joints(self):
        return self._joints

    def compute_passive_force(self, **_kwargs):
        self.passive_force_calls += 1
        return np.arange(self.dof, dtype=np.float32) + 1

    def set_qf(self, qf) -> None:
        self.applied_qf = qf


def make_robot(gpu_active: bool) -> Robot:
    robot = Robot.__new__(Robot)
    left_joints = [FakeJoint("left_arm"), FakeJoint("left_gripper")]
    right_joints = [FakeJoint("right_arm"), FakeJoint("right_gripper")]
    robot.left_entity = FakeEntity(0, left_joints)
    robot.right_entity = FakeEntity(1, right_joints)
    robot.left_arm_joints = [left_joints[0]]
    robot.right_arm_joints = [right_joints[0]]
    robot.left_gripper = [(left_joints[1], 1.0, 0.0)]
    robot.right_gripper = [(right_joints[1], 1.0, 0.0)]
    robot.left_gripper_scale = [0.0, 1.0]
    robot.right_gripper_scale = [0.0, 1.0]
    robot.left_gripper_val = 0.0
    robot.right_gripper_val = 0.0
    robot.left_homestate = [0.15]
    robot.right_homestate = [-0.15]
    if gpu_active:
        physx = SimpleNamespace(
            cuda_articulation_qf=FakeCudaArray(torch.zeros((2, 2), device="cpu")),
            cuda_articulation_target_qpos=FakeCudaArray(torch.zeros((2, 2), device="cpu")),
            cuda_articulation_target_qvel=FakeCudaArray(torch.zeros((2, 2), device="cpu")),
        )
        robot._gpu_runtime = SimpleNamespace(initialized=True, physx=physx)
    else:
        robot._gpu_runtime = None
    return robot


class RobotGpuControlsTest(unittest.TestCase):
    def test_gpu_arm_and_gripper_writes_cuda_buffers(self) -> None:
        robot = make_robot(gpu_active=True)
        robot.set_arm_joints([0.3], [0.4], "left")
        robot.set_gripper(0.6, "right", gripper_eps=0)

        physx = robot._gpu_runtime.physx
        self.assertAlmostEqual(physx.cuda_articulation_target_qpos.torch()[0, 0].item(), 0.3)
        self.assertAlmostEqual(physx.cuda_articulation_target_qvel.torch()[0, 0].item(), 0.4)
        self.assertAlmostEqual(physx.cuda_articulation_target_qpos.torch()[1, 1].item(), 0.6)
        np.testing.assert_allclose(physx.cuda_articulation_qf.torch()[0].numpy(), [1.0, 2.0])
        np.testing.assert_allclose(physx.cuda_articulation_qf.torch()[1].numpy(), [1.0, 2.0])
        self.assertAlmostEqual(robot.get_left_arm_jointState()[0], 0.3)
        self.assertAlmostEqual(robot.get_left_arm_jointState()[1], 0.0)
        self.assertEqual(robot.left_entity.passive_force_calls, 1)
        self.assertEqual(robot.right_entity.passive_force_calls, 1)

    def test_gpu_target_cache_is_invalidated_after_partial_reset(self) -> None:
        robot = make_robot(gpu_active=True)
        robot.set_arm_joints([0.3], [0.4], "left")
        self.assertAlmostEqual(robot.get_left_arm_jointState()[0], 0.3)

        robot._gpu_runtime.physx.cuda_articulation_target_qpos.torch()[0, 0] = -0.2
        robot.invalidate_gpu_control_cache()

        self.assertAlmostEqual(robot.get_left_arm_jointState()[0], -0.2)

    def test_cpu_path_remains_unchanged_before_gpu_init(self) -> None:
        robot = make_robot(gpu_active=False)
        robot.move_to_homestate()
        self.assertAlmostEqual(robot.left_arm_joints[0].target, 0.15)
        self.assertAlmostEqual(robot.right_arm_joints[0].target, -0.15)


if __name__ == "__main__":
    unittest.main()
