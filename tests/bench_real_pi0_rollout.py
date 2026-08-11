#!/usr/bin/env python3
"""Benchmark a real Pi0 + RoboTwin adjust_bottle rollout.

The script deliberately uses only the existing VectorEnv API.  ``--gpu-sim``
selects the internal shared-PhysX implementation; omitting it benchmarks the
clean pre-optimization Git baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict
from PIL import Image

from rlinf.models import get_model
from robotwin.envs.vector_env import VectorEnv


def sync_cuda() -> None:
    torch.cuda.synchronize()


def timed(total: defaultdict[str, float], name: str, fn):
    sync_cuda()
    started_at = time.perf_counter()
    value = fn()
    sync_cuda()
    total[name] += time.perf_counter() - started_at
    return value


def gpu_memory_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.splitlines()[0].strip())


class GpuMonitor:
    """Lightweight nvidia-smi monitor for the measured rollout window."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, float | int]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def _sample(self) -> None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            utilization, memory = (part.strip() for part in output.splitlines()[0].split(","))
            self.samples.append(
                {
                    "elapsed_s": time.perf_counter() - self._started_at,
                    "gpu_util_percent": int(utilization),
                    "gpu_memory_mib": int(memory),
                }
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()

    def summary(self) -> dict[str, float | int]:
        if not self.samples:
            return {}
        utilities = [sample["gpu_util_percent"] for sample in self.samples]
        memories = [sample["gpu_memory_mib"] for sample in self.samples]
        return {
            "gpu_monitor_samples": len(self.samples),
            "gpu_util_mean_percent": float(np.mean(utilities)),
            "gpu_util_peak_percent": int(max(utilities)),
            "gpu_memory_monitor_peak_mib": int(max(memories)),
        }


def deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_task_config(
    rlinf_root: Path,
    *,
    gpu_sim: bool,
    gpu_image_tensors: bool = False,
    gpu_profile: bool = False,
    gpu_defer_cpu_sync: bool = False,
    gpu_parallel_topp: bool = False,
    gpu_parallel_topp_workers: int | None = None,
    clear_cache_freq: int = 1_000_000,
) -> dict:
    """Resolve the exact environment plus ALOHA overrides used by Pi0 PPO."""
    with (
        rlinf_root
        / "examples"
        / "embodiment"
        / "config"
        / "env"
        / "robotwin_adjust_bottle.yaml"
    ).open(encoding="utf-8") as file:
        env_config = yaml.safe_load(file)
    with (
        rlinf_root
        / "examples"
        / "embodiment"
        / "config"
        / "robotwin_adjust_bottle_ppo_openpi.yaml"
    ).open(encoding="utf-8") as file:
        ppo_config = yaml.safe_load(file)

    task_config = copy.deepcopy(env_config["task_config"])
    deep_update(task_config, ppo_config["env"]["train"]["task_config"])
    task_config.update(
        save_path=tempfile.mkdtemp(prefix="robotwin-real-pi0-"),
        save_freq=None,
        eval_video_log=False,
        render_freq=0,
        clear_cache_freq=max(1, int(clear_cache_freq)),
    )
    # The clean pre-optimization checkout has no such internal option.
    if gpu_sim:
        task_config["gpu_sim"] = True
    if gpu_profile:
        if not gpu_sim:
            raise ValueError("gpu_profile requires --gpu-sim")
        task_config["gpu_profile"] = True
    if gpu_defer_cpu_sync:
        if not gpu_sim:
            raise ValueError("gpu_defer_cpu_sync requires --gpu-sim")
        task_config["gpu_defer_cpu_sync"] = True
    if gpu_parallel_topp:
        if not gpu_sim:
            raise ValueError("gpu_parallel_topp requires --gpu-sim")
        task_config["gpu_parallel_topp"] = True
        if gpu_parallel_topp_workers is not None:
            task_config["gpu_parallel_topp_workers"] = gpu_parallel_topp_workers
    if gpu_image_tensors:
        if not gpu_sim:
            raise ValueError("gpu_image_tensors requires --gpu-sim")
        task_config["gpu_image_tensor_obs"] = True
    return task_config


def as_rgb_array(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Match RoboTwinEnv.center_and_crop(..., center_crop=False) exactly."""
    if isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"expected HWC RGB(A) tensor, got {tuple(image.shape)}")
        return image[..., :3].contiguous()
    return np.array(Image.fromarray(np.array(image)).convert("RGB"))


def extract_obs(raw_obs: list[dict]) -> dict:
    """Match RoboTwinEnv._extract_obs_image for Pi0's head + wrist inputs."""
    main_images = []
    wrist_images = []
    states = []
    instructions = []
    for obs in raw_obs:
        main_image = as_rgb_array(obs["full_image"])
        main_images.append(
            main_image
            if isinstance(main_image, torch.Tensor)
            else torch.from_numpy(main_image)
        )
        views = []
        if obs.get("left_wrist_image") is not None:
            image = as_rgb_array(obs["left_wrist_image"])
            views.append(image if isinstance(image, torch.Tensor) else torch.from_numpy(image))
        if obs.get("right_wrist_image") is not None:
            image = as_rgb_array(obs["right_wrist_image"])
            views.append(image if isinstance(image, torch.Tensor) else torch.from_numpy(image))
        if len(views) != 2:
            raise RuntimeError("Pi0 ALOHA benchmark requires both wrist camera images")
        wrist_images.append(torch.stack(views))
        states.append(torch.from_numpy(np.asarray(obs["state"])))
        instructions.append(obs["instruction"])
    return {
        "main_images": torch.stack(main_images),
        "wrist_images": torch.stack(wrist_images),
        "extra_view_images": None,
        "states": torch.stack(states),
        "task_descriptions": instructions,
    }


def load_model(checkpoint: Path, rlinf_root: Path):
    model_cfg = OmegaConf.load(
        rlinf_root / "examples" / "embodiment" / "config" / "model" / "pi0.yaml"
    )
    with open_dict(model_cfg):
        model_cfg.model_path = str(checkpoint)
        model_cfg.num_action_chunks = 50
        model_cfg.action_dim = 14
        model_cfg.use_proprio = True
        model_cfg.add_value_head = True
        model_cfg.openpi.config_name = "pi0_aloha_robotwin"
        model_cfg.openpi.num_images_in_input = 3
        model_cfg.openpi.detach_critic_input = True
    model = get_model(model_cfg)
    model.eval().to("cuda")
    return model


def predict_actions(model, obs: dict, timings: defaultdict[str, float]) -> np.ndarray:
    def run_prediction() -> np.ndarray:
        # ``mode=train`` with values matches the PPO rollout actor's Pi0 path.
        with torch.inference_mode():
            actions, _ = model.predict_action_batch(
                obs, mode="train", compute_values=True
            )
        if not torch.is_tensor(actions):
            return np.asarray(actions)
        return actions.detach().cpu().numpy()

    return timed(timings, "model_predict_s", run_prediction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--env-seeds",
        type=int,
        nargs="+",
        help="Exact stable RoboTwin seeds; defaults to a contiguous range from --seed.",
    )
    parser.add_argument("--action-chunks", type=int, default=4)
    parser.add_argument(
        "--rollout-epochs",
        type=int,
        default=1,
        help="Run this many consecutive reset + real-policy rollout epochs in one process.",
    )
    parser.add_argument("--gpu-sim", action="store_true")
    parser.add_argument("--gpu-image-tensors", action="store_true")
    parser.add_argument("--gpu-profile", action="store_true")
    parser.add_argument("--gpu-defer-cpu-sync", action="store_true")
    parser.add_argument("--gpu-parallel-topp", action="store_true")
    parser.add_argument("--gpu-parallel-topp-workers", type=int)
    parser.add_argument(
        "--clear-cache-freq",
        type=int,
        default=1_000_000,
        help="Reset interval for SAPIEN cache release; ignored by prebuilt GPU slots.",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Measure environment reset time without loading/running Pi0.",
    )
    parser.add_argument("--reset-cycles", type=int, default=8)
    parser.add_argument("--gpu-monitor-output", type=Path)
    parser.add_argument("--gpu-monitor-interval-s", type=float, default=0.2)
    args = parser.parse_args()
    if (
        args.num_envs <= 0
        or args.action_chunks <= 0
        or args.rollout_epochs <= 0
        or args.reset_cycles <= 0
    ):
        raise ValueError("num-envs, action-chunks, rollout-epochs, and reset-cycles must be positive")
    if args.clear_cache_freq <= 0:
        raise ValueError("clear-cache-freq must be positive")
    if (
        args.gpu_parallel_topp_workers is not None
        and args.gpu_parallel_topp_workers <= 0
    ):
        raise ValueError("gpu-parallel-topp-workers must be positive")
    if args.env_seeds is not None and len(args.env_seeds) != args.num_envs:
        raise ValueError("--env-seeds must provide exactly --num-envs values")
    if args.gpu_monitor_interval_s <= 0:
        raise ValueError("gpu-monitor-interval-s must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if not args.reset_only and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --reset-only is used")
    if args.checkpoint is not None and not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)

    timings: defaultdict[str, float] = defaultdict(float)
    env_seeds = (
        args.env_seeds
        if args.env_seeds is not None
        else list(range(args.seed, args.seed + args.num_envs))
    )
    env = None
    model = None
    monitor = None
    try:
        started_at = time.perf_counter()
        env = VectorEnv(
            task_config=load_task_config(
                args.rlinf_root,
                gpu_sim=args.gpu_sim,
                gpu_image_tensors=args.gpu_image_tensors,
                gpu_profile=args.gpu_profile,
                gpu_defer_cpu_sync=args.gpu_defer_cpu_sync,
                gpu_parallel_topp=args.gpu_parallel_topp,
                gpu_parallel_topp_workers=args.gpu_parallel_topp_workers,
                clear_cache_freq=args.clear_cache_freq,
            ),
            n_envs=args.num_envs,
            env_seeds=env_seeds,
        )
        sync_cuda()
        timings["env_construct_s"] = time.perf_counter() - started_at
        timings["gpu_memory_after_env_mib"] = gpu_memory_mib()

        if args.reset_only:
            reset_times = []
            for _ in range(args.reset_cycles):
                started_at = time.perf_counter()
                env.reset()
                sync_cuda()
                reset_times.append(time.perf_counter() - started_at)
            print(
                json.dumps(
                    {
                        "mode": "environment_reset_only",
                        "gpu_sim": args.gpu_sim,
                        "num_envs": args.num_envs,
                        "env_seeds": env_seeds,
                        "clear_cache_freq": args.clear_cache_freq,
                        "reset_cycles": args.reset_cycles,
                        "reset_times_s": reset_times,
                        "mean_reset_s": float(np.mean(reset_times)),
                        "median_reset_s": float(np.median(reset_times)),
                        "env_construct_s": timings["env_construct_s"],
                    },
                    sort_keys=True,
                )
            )
            return

        started_at = time.perf_counter()
        model = load_model(args.checkpoint, args.rlinf_root)
        sync_cuda()
        timings["model_load_to_gpu_s"] = time.perf_counter() - started_at
        timings["gpu_memory_after_model_mib"] = gpu_memory_mib()

        # Warm the model once without advancing the physics.
        env.reset()
        warm_obs = timed(timings, "warmup_preprocess_s", lambda: extract_obs(env.get_obs()))
        _ = predict_actions(model, warm_obs, timings)
        timings["warmup_model_predict_s"] = timings.pop("model_predict_s")

        monitor = (
            GpuMonitor(args.gpu_monitor_interval_s)
            if args.gpu_monitor_output is not None
            else None
        )
        if monitor is not None:
            monitor.start()
        try:
            rollout_epoch_times = []
            for _ in range(args.rollout_epochs):
                started_at = time.perf_counter()
                env.reset()
                raw_obs = timed(timings, "reset_capture_s", env.get_obs)
                obs = timed(timings, "obs_preprocess_s", lambda: extract_obs(raw_obs))
                for _ in range(args.action_chunks):
                    actions = predict_actions(model, obs, timings)
                    raw_obs = timed(timings, "env_step_capture_s", lambda: env.step(actions))
                    obs = timed(timings, "obs_preprocess_s", lambda: extract_obs(raw_obs[0]))

                # PPO rollout collection performs one final policy call for bootstrap values.
                _ = predict_actions(model, obs, timings)
                rollout_epoch_times.append(time.perf_counter() - started_at)

            timings["inference_calls"] = (args.action_chunks + 1) * args.rollout_epochs
            timings["end_to_end_rollout_total_s"] = float(sum(rollout_epoch_times))
            timings["end_to_end_rollout_epoch_s"] = float(np.mean(rollout_epoch_times))
            timings["end_to_end_rollout_epoch_median_s"] = float(np.median(rollout_epoch_times))
            timings["rollout_epoch_times_s"] = rollout_epoch_times
            timings["gpu_memory_peak_mib"] = gpu_memory_mib()
            timings["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
            if args.gpu_profile:
                timings["gpu_runtime_profile"] = env.gpu_runtime.profile_snapshot()
        finally:
            if monitor is not None:
                monitor.stop()
                if args.gpu_monitor_output is not None:
                    args.gpu_monitor_output.parent.mkdir(parents=True, exist_ok=True)
                    args.gpu_monitor_output.write_text(
                        json.dumps(
                            {
                                "mode": "single_process_rlinf_adapter_real_pi0",
                                "gpu_sim": args.gpu_sim,
                                "gpu_image_tensors": args.gpu_image_tensors,
                                "gpu_defer_cpu_sync": args.gpu_defer_cpu_sync,
                                "gpu_parallel_topp": args.gpu_parallel_topp,
                                "gpu_parallel_topp_workers": args.gpu_parallel_topp_workers,
                                "num_envs": args.num_envs,
                                "env_seeds": env_seeds,
                                "action_chunks": args.action_chunks,
                                "samples": monitor.samples,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

        result = {
            "mode": "single_process_rlinf_adapter_real_pi0",
            "gpu_sim": args.gpu_sim,
            "gpu_image_tensors": args.gpu_image_tensors,
            "gpu_defer_cpu_sync": args.gpu_defer_cpu_sync,
            "gpu_parallel_topp": args.gpu_parallel_topp,
            "gpu_parallel_topp_workers": args.gpu_parallel_topp_workers,
            "num_envs": args.num_envs,
            "env_seeds": env_seeds,
            "action_chunks": args.action_chunks,
            "rollout_epochs": args.rollout_epochs,
            **timings,
        }
        if monitor is not None:
            result.update(monitor.summary())
        print(json.dumps(result, sort_keys=True))
    finally:
        if model is not None:
            model.to("cpu")
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
