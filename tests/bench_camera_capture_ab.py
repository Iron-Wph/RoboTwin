#!/usr/bin/env python3
"""Compare RoboTwin's serial and grouped camera capture without changing its API."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from robotwin.envs.vector_env import VectorEnv


def load_task_config(rlinf_root: Path) -> dict:
    config_path = (
        rlinf_root
        / "examples"
        / "embodiment"
        / "config"
        / "env"
        / "robotwin_place_empty_cup.yaml"
    )
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    task_config = copy.deepcopy(config["task_config"])
    task_config.update(
        save_path=tempfile.mkdtemp(prefix="robotwin-camera-ab-"),
        save_freq=None,
        eval_video_log=False,
        render_freq=0,
        clear_cache_freq=1_000_000,
        gpu_sim=True,
    )
    return task_config


def grouped_get_obs(env: VectorEnv) -> list[dict]:
    """Execute VectorEnv's internal grouped RGB path unconditionally.

    This is deliberately a benchmark helper: production selection remains inside
    ``VectorEnv.get_obs`` and keeps its public signature unchanged.
    """
    slots = [sub_env.active_slot for sub_env in env.envs]
    entries_by_env = {
        slot.env_id: slot.task.cameras.observation_camera_entries() for slot in slots
    }
    for slot in slots:
        slot.task._update_render(force=True)
    rgba_by_env = env.gpu_runtime.capture_batched_rgba(entries_by_env)
    for slot in slots:
        slot.task.cameras.set_batched_rgba(rgba_by_env[slot.env_id])

    futures = {
        index: env.env_thread_pool.submit(sub_env.get_obs)
        for index, sub_env in enumerate(env.envs)
    }
    return [futures[index].result(timeout=120) for index in range(env.n_envs)]


def serial_get_obs(env: VectorEnv) -> list[dict]:
    """Run the original per-slot observation collection unconditionally."""
    return [sub_env.get_obs() for sub_env in env.envs]


def compare_images(serial: list[dict], grouped: list[dict]) -> None:
    for index, (serial_obs, grouped_obs) in enumerate(zip(serial, grouped)):
        if serial_obs.keys() != grouped_obs.keys():
            raise AssertionError(f"environment {index} observation keys differ")
        for name, serial_value in serial_obs.items():
            grouped_value = grouped_obs[name]
            if isinstance(serial_value, np.ndarray):
                np.testing.assert_array_equal(serial_value, grouped_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--wrist", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.warmups < 0 or args.repeats <= 0:
        raise ValueError("num-envs/repeats must be positive and warmups non-negative")

    task_config = load_task_config(args.rlinf_root)
    if args.wrist:
        task_config["camera"]["collect_wrist_camera"] = True
    env = VectorEnv(
        task_config=task_config,
        n_envs=args.num_envs,
        env_seeds=list(range(10_000, 10_000 + args.num_envs)),
    )
    try:
        env.reset()
        for _ in range(args.warmups):
            serial_get_obs(env)
            grouped_get_obs(env)

        serial_s: list[float] = []
        grouped_s: list[float] = []
        for _ in range(args.repeats):
            started_at = time.perf_counter()
            serial = serial_get_obs(env)
            serial_s.append(time.perf_counter() - started_at)

            started_at = time.perf_counter()
            grouped = grouped_get_obs(env)
            grouped_s.append(time.perf_counter() - started_at)
            compare_images(serial, grouped)

        serial_median = statistics.median(serial_s)
        grouped_median = statistics.median(grouped_s)
        print(
            json.dumps(
                {
                    "num_envs": args.num_envs,
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                    "wrist": args.wrist,
                    "serial_s": serial_s,
                    "grouped_s": grouped_s,
                    "serial_median_s": serial_median,
                    "grouped_median_s": grouped_median,
                    "grouped_vs_serial_percent": (
                        (grouped_median / serial_median - 1.0) * 100.0
                    ),
                    "images_exact": True,
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
