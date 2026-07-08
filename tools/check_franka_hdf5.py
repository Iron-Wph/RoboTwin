import sys
import h5py


def require_missing(group, name):
    if name in group:
        raise AssertionError(f"Unexpected dual-arm field exists: {group.name}/{name}")


def main(path):
    with h5py.File(path, "r") as f:
        joint_action = f["joint_action"]
        observation = f["observation"]
        endpose = f["endpose"]

        vector_shape = joint_action["vector"].shape
        print("joint_action/vector:", vector_shape)

        if vector_shape[-1] != 8:
            raise AssertionError(f"Expected 8 qpos values for Franka Panda, got {vector_shape[-1]}")

        for name in ["left_arm", "left_gripper", "vector"]:
            if name not in joint_action:
                raise AssertionError(f"Missing joint_action/{name}")

        require_missing(joint_action, "right_arm")
        require_missing(joint_action, "right_gripper")

        for name in ["head_camera", "third_view"]:
            if name not in observation:
                raise AssertionError(f"Missing observation/{name}")

        require_missing(observation, "left_camera")
        require_missing(observation, "right_camera")

        for name in ["left_endpose", "left_gripper"]:
            if name not in endpose:
                raise AssertionError(f"Missing endpose/{name}")

        require_missing(endpose, "right_endpose")
        require_missing(endpose, "right_gripper")

    print("OK: single-arm Franka HDF5 format is correct")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/check_franka_hdf5.py <episode.hdf5>")
        raise SystemExit(2)
    main(sys.argv[1])
