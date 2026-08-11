#!/usr/bin/env python3
"""Measure a real OpenVLA-OFT + GPU VectorEnv rollout without new env APIs."""

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

from rlinf.envs.utils import center_crop_image
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
    """Sample device utilization during the measured rollout only."""

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
            # Monitoring must never make a valid benchmark fail.
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
        utils = [sample["gpu_util_percent"] for sample in self.samples]
        memories = [sample["gpu_memory_mib"] for sample in self.samples]
        return {
            "gpu_monitor_samples": len(self.samples),
            "gpu_util_mean_percent": float(np.mean(utils)),
            "gpu_util_peak_percent": int(max(utils)),
            "gpu_memory_monitor_peak_mib": int(max(memories)),
        }


def load_task_config(rlinf_root: Path, *, gpu_sim: bool) -> dict:
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
        save_path=tempfile.mkdtemp(prefix="robotwin-real-oft-"),
        save_freq=None,
        eval_video_log=False,
        render_freq=0,
        clear_cache_freq=1_000_000,
    )
    # The pre-optimization implementation does not know this internal option.
    # Omit it entirely when benchmarking that clean Git baseline.
    if gpu_sim:
        task_config["gpu_sim"] = True
    return task_config


def extract_obs_image(raw_obs: list[dict]) -> dict:
    """Match RoboTwinEnv's image/state conversion used before OFT inference."""
    images = []
    states = []
    instructions = []
    for obs in raw_obs:
        image = Image.fromarray(np.asarray(obs["full_image"])).convert("RGB")
        # ``RoboTwinEnv.center_and_crop`` materializes a writable NumPy array.
        images.append(torch.from_numpy(np.array(center_crop_image(image))))
        states.append(torch.from_numpy(np.asarray(obs["state"])))
        instructions.append(obs["instruction"])
    return {
        "main_images": torch.stack(images),
        "wrist_images": None,
        "states": torch.stack(states),
        "task_descriptions": instructions,
    }


def load_model(checkpoint: Path, rlinf_root: Path):
    model_cfg = OmegaConf.load(
        rlinf_root / "examples" / "embodiment" / "config" / "model" / "openvla_oft.yaml"
    )
    with open_dict(model_cfg):
        model_cfg.model_path = str(checkpoint)
        model_cfg.implement_version = "official"
        model_cfg.action_dim = 14
        model_cfg.num_action_chunks = 25
        model_cfg.use_proprio = True
        model_cfg.proprio_dim = 14
        model_cfg.unnorm_key = "place_empty_cup_1k"
        model_cfg.is_lora = True
        model_cfg.value_type = "action_level"
    model = get_model(model_cfg)
    model.eval().to("cuda")
    return model


def predict_actions(model, obs: dict, timings: defaultdict[str, float]) -> np.ndarray:
    inputs = timed(timings, "model_prepare_s", lambda: model.prepare_inputs(obs))
    actions, _ = timed(
        timings,
        "model_forward_s",
        lambda: model.predict_action_batch(
            input_ids=inputs[0],
            attention_mask=inputs[1],
            pixel_values=inputs[2],
            proprio=inputs[3],
            do_sample=True,
            temperature=1.6,
            top_k=-1,
        ),
    )
    return actions.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--action-chunks", type=int, default=8)
    parser.add_argument(
        "--gpu-sim",
        action="store_true",
        help="Enable the optimized shared-GPU-simulation implementation.",
    )
    parser.add_argument(
        "--gpu-monitor-output",
        type=Path,
        help="Optional JSON path for nvidia-smi samples during the measured epoch.",
    )
    parser.add_argument("--gpu-monitor-interval-s", type=float, default=0.2)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.action_chunks <= 0:
        raise ValueError("num-envs and action-chunks must be positive")
    if args.gpu_monitor_interval_s <= 0:
        raise ValueError("gpu-monitor-interval-s must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)

    timings: defaultdict[str, float] = defaultdict(float)
    env = None
    model = None
    monitor = None
    try:
        started_at = time.perf_counter()
        env = VectorEnv(
            task_config=load_task_config(args.rlinf_root, gpu_sim=args.gpu_sim),
            n_envs=args.num_envs,
            env_seeds=list(range(args.seed, args.seed + args.num_envs)),
        )
        sync_cuda()
        timings["env_construct_s"] = time.perf_counter() - started_at
        timings["gpu_memory_after_env_mib"] = gpu_memory_mib()

        started_at = time.perf_counter()
        model = load_model(args.checkpoint, args.rlinf_root)
        sync_cuda()
        timings["model_load_to_gpu_s"] = time.perf_counter() - started_at
        timings["gpu_memory_after_model_mib"] = gpu_memory_mib()

        # Do one inference only to remove first-kernel initialization from the
        # steady-state rollout metric.  It never advances the simulation.
        env.reset()
        warm_obs = timed(timings, "warmup_preprocess_s", lambda: extract_obs_image(env.get_obs()))
        _ = predict_actions(model, warm_obs, timings)
        timings["warmup_model_prepare_s"] = timings.pop("model_prepare_s")
        timings["warmup_model_forward_s"] = timings.pop("model_forward_s")

        monitor = (
            GpuMonitor(args.gpu_monitor_interval_s)
            if args.gpu_monitor_output is not None
            else None
        )
        if monitor is not None:
            monitor.start()
        try:
            started_at = time.perf_counter()
            env.reset()
            raw_obs = timed(timings, "reset_capture_s", env.get_obs)
            obs = timed(timings, "obs_preprocess_s", lambda: extract_obs_image(raw_obs))
            for _ in range(args.action_chunks):
                actions = predict_actions(model, obs, timings)
                raw_obs = timed(timings, "env_step_capture_s", lambda: env.step(actions))
                obs = timed(timings, "obs_preprocess_s", lambda: extract_obs_image(raw_obs[0]))

            # RLinf's training route requests one extra policy output for bootstrap
            # values after the final action chunk.
            _ = predict_actions(model, obs, timings)
            timings["inference_calls"] = args.action_chunks + 1
            timings["end_to_end_rollout_epoch_s"] = time.perf_counter() - started_at
            timings["gpu_memory_peak_mib"] = gpu_memory_mib()
            timings["torch_peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
            )
            result = {
                "mode": "single_process_rlinf_adapter_real_oft",
                "gpu_sim": args.gpu_sim,
                "num_envs": args.num_envs,
                "action_chunks": args.action_chunks,
                **timings,
            }
            if monitor is not None:
                result.update(monitor.summary())
            print(json.dumps(result, sort_keys=True))
        finally:
            if monitor is not None:
                monitor.stop()
                if args.gpu_monitor_output is not None:
                    args.gpu_monitor_output.parent.mkdir(parents=True, exist_ok=True)
                    args.gpu_monitor_output.write_text(
                        json.dumps(
                            {
                                "mode": "single_process_rlinf_adapter_real_oft",
                                "gpu_sim": args.gpu_sim,
                                "num_envs": args.num_envs,
                                "action_chunks": args.action_chunks,
                                "samples": monitor.samples,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
    finally:
        if model is not None:
            model.to("cpu")
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
