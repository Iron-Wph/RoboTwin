#!/usr/bin/env python3
"""Measure either RoboTwin VectorEnv backend on a fixed seed and action stream."""

from __future__ import annotations

import argparse
import copy
import json
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
    task_config["save_path"] = tempfile.mkdtemp(prefix="robotwin-baseline-")
    task_config["save_freq"] = None
    task_config["eval_video_log"] = False
    task_config["render_freq"] = 0
    task_config["clear_cache_freq"] = 1000000
    return task_config


def make_hold_actions(observations: list[dict], horizon: int) -> np.ndarray:
    """Hold each robot at its latest measured joint state for one action chunk."""
    states = np.stack(
        [np.asarray(observation["state"], dtype=np.float32) for observation in observations]
    )
    return np.repeat(states[:, None, :], horizon, axis=1)


def as_done_mask(values, num_envs: int) -> np.ndarray:
    done = np.asarray(values, dtype=bool).reshape(-1)
    if done.shape != (num_envs,):
        raise AssertionError(f"expected done shape {(num_envs,)}, got {done.shape}")
    return done


def run_until_done(
    env: VectorEnv,
    observations: list[dict],
    horizon: int,
    max_action_chunks: int,
) -> tuple[list[dict], dict]:
    """Run every slot until its task signals termination or truncation.

    This deliberately uses a state-hold policy: it measures the end-to-end
    environment rollout cost, including every task control tick and camera
    observation, without mixing in model inference or policy quality.
    """
    num_envs = len(observations)
    done = np.zeros(num_envs, dtype=bool)
    termination_count = 0
    truncation_count = 0
    first_done_action_steps = np.zeros(num_envs, dtype=np.int64)

    started_at = time.perf_counter()
    for action_chunk in range(1, max_action_chunks + 1):
        actions = make_hold_actions(observations, horizon)
        observations, _, terminations, truncations, _ = env.step(actions)
        terminated = as_done_mask(terminations, num_envs)
        truncated = as_done_mask(truncations, num_envs)
        newly_done = ~done & (terminated | truncated)
        first_done_action_steps[newly_done] = action_chunk * horizon
        termination_count += int(np.count_nonzero(newly_done & terminated))
        truncation_count += int(np.count_nonzero(newly_done & truncated))
        done |= terminated | truncated
        if done.all():
            elapsed = time.perf_counter() - started_at
            completed_action_steps = int(first_done_action_steps.sum())
            return observations, {
                "rollout_s": elapsed,
                "action_chunks": action_chunk,
                "completed_envs": int(done.sum()),
                "terminated_envs": termination_count,
                "truncated_envs": truncation_count,
                "per_env_action_steps": first_done_action_steps.tolist(),
                "completed_action_steps": completed_action_steps,
                "env_action_steps_per_s": completed_action_steps / elapsed,
                "episodes_per_s": num_envs / elapsed,
            }

    raise TimeoutError(
        f"no full completion after {max_action_chunks} action chunks; "
        f"done={done.tolist()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--gpu-sim", action="store_true")
    parser.add_argument(
        "--gpu-profile",
        action="store_true",
        help="include internal GPU control, PhysX, and observation timing totals",
    )
    parser.add_argument(
        "--complete-episode",
        action="store_true",
        help="run until every environment reports terminated or truncated",
    )
    parser.add_argument("--max-action-chunks", type=int, default=10000)
    args = parser.parse_args()

    if (
        args.num_envs <= 0
        or args.horizon <= 0
        or args.rollout_steps <= 0
        or args.max_action_chunks <= 0
    ):
        raise ValueError("environment and rollout sizes must be positive")

    task_config = load_task_config(args.rlinf_root)
    task_config["gpu_sim"] = args.gpu_sim
    task_config["gpu_profile"] = args.gpu_profile
    seeds = list(range(args.seed, args.seed + args.num_envs))
    timings: dict[str, float] = {}
    env = None
    try:
        start = time.perf_counter()
        env = VectorEnv(task_config=task_config, n_envs=args.num_envs, env_seeds=seeds)
        timings["construct_s"] = time.perf_counter() - start

        start = time.perf_counter()
        env.reset()
        timings["reset_s"] = time.perf_counter() - start

        start = time.perf_counter()
        observations = env.get_obs()
        timings["get_obs_s"] = time.perf_counter() - start

        states = np.stack([np.asarray(obs["state"], dtype=np.float32) for obs in observations])
        if states.shape != (args.num_envs, 14):
            raise AssertionError(f"expected states shape {(args.num_envs, 14)}, got {states.shape}")
        if args.complete_episode:
            next_obs, episode_stats = run_until_done(
                env,
                observations,
                args.horizon,
                args.max_action_chunks,
            )
            timings.update(episode_stats)
            timings["step_s"] = timings["rollout_s"] / timings["action_chunks"]
            run_metadata = {
                "mode": "complete_episode_hold_policy",
                "horizon": args.horizon,
            }
        else:
            actions = np.repeat(states[:, None, :], args.horizon, axis=1)
            start = time.perf_counter()
            for _ in range(args.rollout_steps):
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
            timings["rollout_s"] = time.perf_counter() - start
            timings["step_s"] = timings["rollout_s"] / args.rollout_steps
            run_metadata = {
                "mode": "fixed_steps",
                "rollout_steps": args.rollout_steps,
            }

            for name, values in {
                "rewards": rewards,
                "terminations": terminations,
                "truncations": truncations,
                "infos": infos,
            }.items():
                if len(values) != args.num_envs:
                    raise AssertionError(f"{name} returned {len(values)} entries")

        if len(next_obs) != args.num_envs:
            raise AssertionError("step returned the wrong number of observations")

        if args.complete_episode:
            # RLinf's non-auto-reset bootstrap executes the same three phases:
            # restore an episode, obtain its first observation, then run all
            # action chunks in that rollout epoch.  Keep it separate from
            # construction and model inference.
            timings["vector_env_rollout_epoch_s"] = (
                timings["reset_s"] + timings["get_obs_s"] + timings["rollout_s"]
            )

        if args.gpu_profile:
            if not args.gpu_sim:
                raise ValueError("--gpu-profile requires --gpu-sim")
            timings["gpu_profile"] = env.gpu_runtime.profile_snapshot()

        print(
            json.dumps(
                {
                    "backend": "gpu" if args.gpu_sim else "threaded",
                    "num_envs": args.num_envs,
                    **run_metadata,
                    **timings,
                },
                sort_keys=True,
            )
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
