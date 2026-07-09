"""Convert single-arm Franka RoboTwin HDF5 data to LeRobot format."""

import dataclasses
import fnmatch
import json
import os
from pathlib import Path
import re
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

FRANKA_MOTORS = [
    "franka_joint_1",
    "franka_joint_2",
    "franka_joint_3",
    "franka_joint_4",
    "franka_joint_5",
    "franka_joint_6",
    "franka_joint_7",
    "franka_gripper",
]

FRANKA_CAMERAS = [
    "cam_high",
    "cam_left_wrist",
]


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(FRANKA_MOTORS), ),
            "names": [FRANKA_MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(FRANKA_MOTORS), ),
            "names": [FRANKA_MOTORS],
        },
    }

    for cam in FRANKA_CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        if camera not in ep["/observations/images"]:
            raise KeyError(f"Missing camera {camera} in {ep.filename}")

        dataset = ep[f"/observations/images/{camera}"]
        if dataset.ndim == 4:
            imgs_array = dataset[:]
        else:
            imgs_array = []
            for data in dataset:
                data = bytes(data).rstrip(b"\0")
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to decode {camera} frame in {ep.filename}")
                imgs_array.append(img)
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(ep_path: Path) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        if state.shape[-1] != len(FRANKA_MOTORS):
            raise ValueError(f"{ep_path}: expected {len(FRANKA_MOTORS)}D state, got {state.shape[-1]}")
        if action.shape[-1] != len(FRANKA_MOTORS):
            raise ValueError(f"{ep_path}: expected {len(FRANKA_MOTORS)}D action, got {action.shape[-1]}")
        imgs_per_cam = load_raw_images_per_camera(ep, FRANKA_CAMERAS)

    return imgs_per_cam, state, action


def load_instruction(ep_path: Path) -> str:
    instruction_path = ep_path.parent / "instructions.json"
    with open(instruction_path, "r", encoding="utf-8") as f_instr:
        instruction_dict = json.load(f_instr)
    instructions = instruction_dict["instructions"]
    if isinstance(instructions, str):
        return instructions
    if not instructions:
        raise ValueError(f"No instructions in {instruction_path}")
    return instructions[0]


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        imgs_per_cam, state, action = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        instruction = load_instruction(ep_path) if (ep_path.parent / "instructions.json").exists() else task

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)\.hdf5$", path.name)
    if match:
        return int(match.group(1)), str(path)
    return 10**9, str(path)


def port_franka(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw_dir: {raw_dir}")

    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(Path(root) / filename)
    hdf5_files = sorted(hdf5_files, key=episode_sort_key)
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {raw_dir}")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="franka_panda",
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_franka)
