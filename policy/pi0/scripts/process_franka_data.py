import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np


def images_encoding(imgs):
    encode_data = []
    max_len = 0
    for img in imgs:
        success, encoded_image = cv2.imencode(".jpg", img)
        if not success:
            raise RuntimeError("Failed to encode camera frame")
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return [data.ljust(max_len, b"\0") for data in encode_data], max_len


def decode_image(frame):
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        img = frame
    else:
        raw = bytes(frame).rstrip(b"\0")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode camera frame")
    if img.shape[:2] != (480, 640):
        img = cv2.resize(img, (640, 480))
    return img


def load_instructions(path, desc_type):
    with open(path, "r", encoding="utf-8") as f_instr:
        instruction_dict = json.load(f_instr)

    instructions = instruction_dict.get(desc_type)
    if instructions is None:
        instructions = instruction_dict.get("instructions")
    if instructions is None:
        raise KeyError(f"Missing '{desc_type}' or 'instructions' in {path}")
    if isinstance(instructions, str):
        instructions = [instructions]
    return {"instructions": instructions}


def load_franka_hdf5(dataset_path):
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        if "/joint_action/vector" in root:
            states = root["/joint_action/vector"][()]
        else:
            left_arm = root["/joint_action/left_arm"][()]
            left_gripper = root["/joint_action/left_gripper"][()]
            states = np.concatenate([left_arm, left_gripper.reshape(-1, 1)], axis=-1)

        if states.shape[-1] != 8:
            raise ValueError(f"{dataset_path}: expected 8D Franka vector, got {states.shape[-1]}")

        image_dict = {
            "head_camera": root["/observation/head_camera/rgb"][()],
            "left_camera": root["/observation/left_camera/rgb"][()],
        }

    return states.astype(np.float32), image_dict


def data_transform(path, episode_num, save_path, desc_type="seen"):
    path = Path(path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for i in range(episode_num):
        instruction_data_path = path / "instructions" / f"episode{i}.json"
        save_episode_dir = save_path / f"episode_{i}"
        save_episode_dir.mkdir(parents=True, exist_ok=True)

        instructions = load_instructions(instruction_data_path, desc_type)
        with open(save_episode_dir / "instructions.json", "w", encoding="utf-8") as f:
            json.dump(instructions, f, indent=2)

        states, image_dict = load_franka_hdf5(path / "data" / f"episode{i}.hdf5")
        usable_len = min(states.shape[0], image_dict["head_camera"].shape[0], image_dict["left_camera"].shape[0])
        if usable_len < 2:
            raise ValueError(f"episode{i}: need at least 2 frames, got {usable_len}")

        qpos = states[:usable_len - 1]
        actions = states[1:usable_len]

        cam_high = [decode_image(frame) for frame in image_dict["head_camera"][:usable_len - 1]]
        cam_left_wrist = [decode_image(frame) for frame in image_dict["left_camera"][:usable_len - 1]]

        hdf5_path = save_episode_dir / f"episode_{i}.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("action", data=actions)
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=qpos)
            image = obs.create_group("images")

            cam_high_enc, len_high = images_encoding(cam_high)
            cam_left_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_left_wrist", data=cam_left_enc, dtype=f"S{len_left}")

        print(f"process franka episode {i} success!")

    return episode_num


def main():
    parser = argparse.ArgumentParser(description="Convert single-arm Franka RoboTwin episodes to pi0 raw HDF5.")
    parser.add_argument("task_name", type=str)
    parser.add_argument("setting", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--dataset-root", type=Path, default=Path("../../data"))
    parser.add_argument("--desc-type", type=str, default="seen")
    args = parser.parse_args()

    load_dir = args.dataset_root / args.task_name / args.setting
    if not load_dir.is_dir():
        raise FileNotFoundError(f"Missing RoboTwin dataset directory: {load_dir}")

    target_dir = Path("processed_data") / f"{args.task_name}-{args.setting}-{args.expert_data_num}"
    print(f"read single-arm Franka data from path: {load_dir}")
    data_transform(load_dir, args.expert_data_num, target_dir, desc_type=args.desc_type)


if __name__ == "__main__":
    main()
