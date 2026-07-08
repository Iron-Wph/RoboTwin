import re
import sys
from pathlib import Path

import h5py


def episode_id(path):
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        return None
    return int(match.group(1))


def require_missing(group, name):
    if name in group:
        raise AssertionError(f"Unexpected dual-arm field exists: {group.name}/{name}")


def check_episode(path):
    with h5py.File(path, "r") as f:
        joint_action = f["joint_action"]
        observation = f["observation"]
        endpose = f["endpose"]

        vector_shape = joint_action["vector"].shape
        if vector_shape[-1] != 8:
            raise AssertionError(f"{path}: expected 8 qpos values for Franka Panda, got {vector_shape[-1]}")

        for name in ["left_arm", "left_gripper", "vector"]:
            if name not in joint_action:
                raise AssertionError(f"{path}: missing joint_action/{name}")

        require_missing(joint_action, "right_arm")
        require_missing(joint_action, "right_gripper")

        for name in ["head_camera", "left_camera"]:
            if name not in observation:
                raise AssertionError(f"{path}: missing observation/{name}")

        require_missing(observation, "right_camera")

        for name in ["left_endpose", "left_gripper"]:
            if name not in endpose:
                raise AssertionError(f"{path}: missing endpose/{name}")

        require_missing(endpose, "right_endpose")
        require_missing(endpose, "right_gripper")

    return vector_shape[0]


def main(data_dir, expected_count):
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise SystemExit(f"Missing data directory: {data_path}")

    files = []
    for path in data_path.glob("episode*.hdf5"):
        idx = episode_id(path)
        if idx is not None:
            files.append((idx, path))
    files.sort()

    if len(files) != expected_count:
        raise AssertionError(f"Expected {expected_count} hdf5 files, found {len(files)} in {data_path}")

    expected_ids = list(range(expected_count))
    actual_ids = [idx for idx, _ in files]
    if actual_ids != expected_ids:
        raise AssertionError(f"Episode ids are not contiguous. Expected {expected_ids[:5]}...{expected_ids[-5:]}, got {actual_ids[:5]}...{actual_ids[-5:]}")

    frame_counts = [check_episode(path) for _, path in files]
    print(f"OK: {expected_count} single-arm Franka episodes in {data_path}")
    print(f"Frames per episode: min={min(frame_counts)}, max={max(frame_counts)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/check_franka_dataset.py <data_dir> <expected_episode_num>")
        raise SystemExit(2)
    main(sys.argv[1], int(sys.argv[2]))
