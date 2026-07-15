"""Collect valid RoboTwin seeds without recording trajectories or HDF5 data.

The normal ``collect_data.py`` first plans a successful trajectory and then
records it. This entry point intentionally runs only the planning/checking
part. It is used by ``tools/collect_seeds_multi_gpu.sh`` so that every GPU
can search an independent, interleaved seed sequence.
"""

import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path

import yaml

sys.path.append("./")

import sapien.core as sapien  # noqa: F401 (initialises the SAPIEN backend)
from sapien.render import clear_cache

from envs import *  # noqa: F401,F403 (keeps parity with collect_data.py)
from script.embodiment import resolve_embodiment_config


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except Exception as exc:
        raise SystemExit(f"No such task: {task_name}") from exc


def load_task_args(task_name, task_config):
    config_path = Path("task_config") / f"{task_config}.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Task config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        args = yaml.load(file.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    args = resolve_embodiment_config(args, embodiment_config_path)
    args["task_config"] = task_config
    return args


def read_seed_file(seed_file):
    if not seed_file.is_file():
        return []

    seeds = []
    for token in seed_file.read_text(encoding="utf-8").split():
        try:
            seed = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid seed '{token}' in {seed_file}") from exc
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def write_seed_file(seed_file, seeds):
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = seed_file.with_name(seed_file.name + ".tmp")
    temporary_file.write_text("".join(f"{seed}\n" for seed in seeds), encoding="utf-8")
    os.replace(temporary_file, seed_file)


def close_task(task, clear_cache_now=False):
    """Close a partially-created task as well as a normal task."""

    try:
        task.close_env(clear_cache=clear_cache_now)
    except Exception:
        # setup_demo can fail before the task has a complete scene. Do not
        # hide the original planning error behind a cleanup error.
        if clear_cache_now:
            try:
                clear_cache()
            except Exception:
                pass


def collect_seeds(
    task_name,
    task_config,
    target_count,
    seed_file,
    seed_start=0,
    seed_step=1,
    max_attempts=0,
):
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    if seed_step <= 0:
        raise ValueError("seed_step must be positive")

    seed_file = Path(seed_file)
    seeds = read_seed_file(seed_file)
    if len(seeds) >= target_count:
        write_seed_file(seed_file, seeds)
        print(f"Already have {len(seeds)} seeds in {seed_file}")
        return seeds

    args = load_task_args(task_name, task_config)

    # This script must not enter the trajectory/HDF5 collection phase. The
    # worker directory is isolated in case an environment accidentally writes
    # auxiliary files while it is being planned.
    args["episode_num"] = target_count
    args["use_seed"] = False
    args["collect_data"] = False
    args["save_data"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    runtime_dir = seed_file.parent / f"{seed_file.stem}_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    args["save_path"] = str(runtime_dir)

    # If this worker is resumed, continue its own arithmetic progression.
    # Different workers use different seed_start values and the same step, so
    # their candidate sets are disjoint.
    current_seed = seed_start if not seeds else max(seeds) + seed_step
    attempts = 0
    failed = 0
    task = class_decorator(task_name)

    print(
        f"[{task_name}/{task_config}] target={target_count}, "
        f"seed_start={current_seed}, seed_step={seed_step}, "
        f"existing={len(seeds)}, output={seed_file}"
    )

    while len(seeds) < target_count:
        if max_attempts > 0 and attempts >= max_attempts:
            break

        attempts += 1
        print(
            f"Trying seed {current_seed} "
            f"(success {len(seeds)}/{target_count}, attempt {attempts})"
        )
        try:
            task.setup_demo(now_ep_num=len(seeds), seed=current_seed, **args)
            task.play_once()
            succeeded = bool(getattr(task, "plan_success", False) and task.check_success())
            if succeeded:
                seeds.append(current_seed)
                write_seed_file(seed_file, seeds)
                print(f"Accepted seed {current_seed} ({len(seeds)}/{target_count})")
            else:
                failed += 1
                print(f"Rejected seed {current_seed}")
        except UnStableError as exc:
            failed += 1
            print(f"Unstable seed {current_seed}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"Seed {current_seed} failed: {exc}")
            traceback.print_exc()
        finally:
            clear_now = attempts % max(1, int(args.get("clear_cache_freq", 5))) == 0
            close_task(task, clear_cache_now=clear_now)
            # A failed setup may leave the scene in a bad state, so make a
            # fresh task instance for every candidate.
            task = class_decorator(task_name)

        current_seed += seed_step
        if failed and failed % 10 == 0:
            print(f"Rejected/failed candidates so far: {failed}")

    write_seed_file(seed_file, seeds)
    print(
        f"Finished {seed_file}: {len(seeds)}/{target_count} valid seeds, "
        f"{failed} failed candidates, {attempts} attempts"
    )
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum candidates to try; 0 means keep searching until complete.",
    )
    args = parser.parse_args()

    # Match collect_data.py's backend smoke test, but do not enable a viewer.
    from test_render import Sapien_TEST

    Sapien_TEST()
    result = collect_seeds(
        task_name=args.task_name,
        task_config=args.task_config,
        target_count=args.num_seeds,
        seed_file=args.seed_file,
        seed_start=args.seed_start,
        seed_step=args.seed_step,
        max_attempts=args.max_attempts,
    )
    return 0 if len(result) >= args.num_seeds else 2


if __name__ == "__main__":
    sys.exit(main())
