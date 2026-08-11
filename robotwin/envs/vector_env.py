import importlib
import os
import gc
import sys
import copy

import cv2
import torch
import yaml
from envs import *

sys.path.append("../../")

import logging
import multiprocessing as mp
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import gymnasium as gym
import numpy as np
from description.utils.generate_episode_instructions import (
    generate_episode_descriptions,
)
from envs._GLOBAL_CONFIGS import *
from envs.utils.create_actor import UnStableError
from robotwin.envs.gpu_runtime import SharedGpuRuntime


LOG_LEVEL = os.getenv("VECTOR_ENV_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("concurrent.futures").setLevel(logging.WARNING)
logging.getLogger("curobo").setLevel(logging.ERROR)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def update_obs(observation):
    full_image = observation["observation"]["head_camera"]["rgb"]
    left_wrist_image = (
        observation["observation"].get("left_camera", {}).get("rgb", None)
    )
    right_wrist_image = (
        observation["observation"].get("right_camera", {}).get("rgb", None)
    )
    state = observation["joint_action"]["vector"]

    return {
        "full_image": full_image,
        "left_wrist_image": left_wrist_image,
        "right_wrist_image": right_wrist_image,
        "state": state,
    }


class SubEnv:
    def __init__(
        self,
        env_id: int,
        task_name: str,
        args: dict,
        env_seed: int = None,
        instruction_type = "seen",
        global_lock=None,
    ):
        self.env_id = env_id
        self.task_name = task_name
        self.args = args
        self.env_seed = env_seed
        if self.env_seed is None:
            self.env_seed = self.env_id
        self.instruction = None
        self.task = class_decorator(self.task_name)
        self.instruction_type = instruction_type
        self.global_lock = global_lock
        self.lock = threading.Lock()
        self.reset_count = 0
        self.clear_cache_freq = max(1, int(self.args.get("clear_cache_freq", 8)))

    def setup_task(self):
        self.close()
        self.task = class_decorator(self.task_name)

        with self.global_lock:
            with self.lock:
                trial_seed = self.env_seed
                is_valid = False
                while not is_valid:
                    try:
                        task = class_decorator(self.task_name)
                        task.setup_demo(
                            now_ep_num=trial_seed,
                            seed=trial_seed,
                            is_test=True,
                            **self.args,
                        )
                        episode_info = task.get_info()
                        is_valid = True
                    except Exception as e:
                        task.close_env()
                        trial_seed += 1
                        continue
                    task.close_env()

                self.episode_info_list = [episode_info]

    def create_instruction(self):
        task_descriptions = generate_episode_descriptions(
            self.task_name, self.episode_info_list, 1, self.env_seed
        )
        instruction = np.random.choice(task_descriptions[0][self.instruction_type])
        return instruction

    def step(self, actions):
        if self.get_instruction() is None:
            self.reset(env_seed=None)

        with self.lock:
            reward, termination, truncation, info = self.task.gen_sparse_reward_data(actions)
            obs = update_obs(self.task.get_obs())
            obs["instruction"] = self.task.get_instruction()

        return {
            "obs": obs,
            "reward": reward,
            "terminated": termination,
            "truncated": truncation,
            "info": info,
        }

    def reset(self, env_seed=None):
        with self.global_lock:
            with self.lock:
                if self.task is not None:
                    self.reset_count += 1
                    should_clear_cache = self.reset_count % self.clear_cache_freq == 0
                    self.task.close_env(clear_cache=should_clear_cache)
                if env_seed is not None:
                    self.env_seed = env_seed

                self.instruction = self.create_instruction()
                self.args["instruction"] = self.instruction

                trial_seed = self.env_seed
                is_valid = False
                while not is_valid:
                    try:
                        self.task.setup_demo(
                            now_ep_num=trial_seed, seed=trial_seed, **self.args
                        )
                        self.task.step_lim = self.args["step_lim"]
                        self.task.run_steps = 0
                        self.task.reward_step = 0
                        is_valid = True
                    except UnStableError as e:
                        logging.warning(
                            f"RoboTwin SubEnv {self.env_id} reset error with seed {trial_seed}, error: {e}, trying new seed: {trial_seed + 1}"
                        )
                        self.task.close_env(clear_cache=True)
                        trial_seed += 1
                        continue
                    except Exception as e:
                        logging.error(
                            f"RoboTwin SubEnv {self.env_id} reset error with seed {trial_seed}, error: {e}"
                        )
                        self.task.close_env(clear_cache=True)
                        raise

        return

    def get_obs(self):
        with self.lock:
            obs = self.task.get_obs()
            obs = update_obs(obs)
            obs["instruction"] = self.task.get_instruction()

        return obs

    def get_instruction(self):
        with self.lock:
            if self.task is None:
                return None
            return self.instruction

    def close(self, clear_cache=True):
        if self.task is not None:
            with self.lock:
                self.task.close_env(clear_cache=clear_cache)

    def check_seed(self, seed):
        setup_demo_success = False
        play_once_success = False

        t1 = time.time()
        with self.global_lock:
            with self.lock:
                try:
                    self.task.setup_demo(now_ep_num=seed, seed=seed, **self.args)
                    setup_demo_success = True
                    _ = self.task.get_obs()
                    _ = self.task.play_once()
                    if self.task.plan_success and self.task.check_success():
                        play_once_success = True
                except Exception as e:
                    logging.warning(
                        f"RoboTwin SubEnv {self.env_id} check_seed error with seed {seed}, error: {e}"
                    )
        t2 = time.time()
        result = {
            "setup_demo_success": setup_demo_success,
            "play_once_success": play_once_success,
            "cost_time": t2 - t1,
        }
        return result


class GpuTaskSlot:
    """One prebuilt physical GPU scene for a specific task seed."""

    def __init__(
        self,
        env_id,
        task_name,
        args,
        env_seed,
        instruction_type,
        runtime,
        validate_partial_resets=False,
    ):
        self.env_id = env_id
        self.task_name = task_name
        self.args = copy.deepcopy(args)
        self.env_seed = env_id if env_seed is None else env_seed
        self.instruction_type = instruction_type
        self.runtime = runtime
        self.validate_partial_resets = validate_partial_resets
        self.lock = threading.Lock()
        self.task = class_decorator(task_name)
        self.episode_info_list = None
        self.instruction = None
        self.initial_state = None

    def create_instruction(self):
        descriptions = generate_episode_descriptions(
            self.task_name, self.episode_info_list, 1, self.env_seed
        )
        return np.random.choice(descriptions[0][self.instruction_type])

    def construct(self):
        self.task.setup_demo(
            now_ep_num=self.env_seed,
            seed=self.env_seed,
            _shared_scene=self.runtime.scenes[self.env_id],
            _gpu_runtime=self.runtime,
            _gpu_env_id=self.env_id,
            **self.args,
        )
        if (
            self.runtime.defer_cpu_sync
            and not self.task.supports_gpu_deferred_cpu_sync()
        ):
            raise ValueError(
                f"{self.task_name} has no verified GPU deferred-sync success adapter; "
                "leave gpu_defer_cpu_sync disabled or validate this task first"
            )
        self.episode_info_list = [self.task.get_info()]
        self.instruction = self.create_instruction()
        self.task.set_instruction(self.instruction)
        self.runtime.record_env_topology(self.env_id)

    def finalize_construction(self):
        self.task.finalize_gpu_construction()
        self.task.step_lim = self.args["step_lim"]
        self.task.run_steps = 0
        self.task.reward_step = 0
        self.initial_state = self.runtime.capture_env_state(self.env_id)

    def step(self, actions):
        started_at = time.perf_counter()
        try:
            reward, termination, truncation, info = self.task.gen_sparse_reward_data(actions)
        finally:
            # Mark completion before observations: in deferred mode the last
            # slot performs the sole GPU->CPU pose sync, then releases every
            # slot waiting to read its final image/reward state.
            self.runtime.finish_batched_slot(self.env_id)
        if self.runtime.defer_cpu_sync:
            self.runtime.wait_for_cpu_sync()
            if not self.task.eval_success and self.task.check_success():
                self.task.eval_success = True
                info["success"] = True
                reward = np.array([1], dtype=np.float32)
                termination = np.array([1], dtype=np.int32)
        after_control = time.perf_counter()
        obs = update_obs(self.task.get_obs())
        self.runtime.record_slot_timing(
            after_control - started_at,
            time.perf_counter() - after_control,
        )
        obs["instruction"] = self.instruction
        return {
            "obs": obs,
            "reward": reward,
            "terminated": termination,
            "truncated": truncation,
            "info": info,
        }

    def reset(self):
        untouched_states = (
            self.runtime.capture_unrelated_env_states(self.env_id)
            if self.validate_partial_resets
            else ()
        )
        self.instruction = self.create_instruction()
        self.task.set_instruction(self.instruction)
        self.runtime.restore_env_state(self.initial_state)
        if self.validate_partial_resets:
            self.runtime.assert_partial_restore_isolated(
                self.initial_state, untouched_states
            )
        self.task.robot.invalidate_gpu_control_cache()
        self.task.run_steps = 0
        self.task.reward_step = 0
        self.task.take_action_cnt = 0
        self.task.eval_success = False

    def get_obs(self):
        with self.lock:
            obs = update_obs(self.task.get_obs())
            obs["instruction"] = self.instruction
            return obs

    def close(self, clear_cache=True):
        return


class GpuSubEnv:
    """Logical RLinf slot backed by a pool of prebuilt physical GPU scenes."""

    def __init__(self, env_id, slots, initial_seed):
        self.env_id = env_id
        self.slots = {slot.env_seed: slot for slot in slots}
        self.active_slot = self._select_slot(initial_seed)

    def _select_slot(self, seed):
        try:
            return self.slots[seed]
        except KeyError as error:
            available = sorted(self.slots)
            raise NotImplementedError(
                f"GPU reset seed {seed} is not in gpu_reset_seed_pool; available seeds: {available}"
            ) from error

    @property
    def runtime_env_id(self):
        return self.active_slot.env_id

    def finalize_construction(self):
        for slot in self.slots.values():
            slot.finalize_construction()

    def step(self, actions):
        return self.active_slot.step(actions)

    def reset(self, env_seed=None):
        if env_seed is not None:
            self.active_slot = self._select_slot(env_seed)
        self.active_slot.reset()

    def get_obs(self):
        return self.active_slot.get_obs()

    def check_seed(self, seed):
        """Run the existing seed check against one already-built GPU scene."""
        setup_demo_success = False
        play_once_success = False
        started_at = time.time()
        snapshots = []
        try:
            slot = self._select_slot(seed)
            snapshots = [
                self.active_slot.runtime.capture_env_state(env_id)
                for env_id in range(self.active_slot.runtime.num_envs)
            ]
            slot.reset()
            setup_demo_success = True
            self.active_slot.runtime.begin_batched_step([slot.env_id])
            try:
                _ = slot.task.get_obs()
                _ = slot.task.play_once()
                play_once_success = slot.task.plan_success and slot.task.check_success()
            finally:
                self.active_slot.runtime.finish_batched_slot(slot.env_id)
                self.active_slot.runtime.end_batched_step()
        except Exception as error:
            logging.warning(
                "RoboTwin GPU SubEnv %s check_seed error with seed %s: %s",
                self.env_id,
                seed,
                error,
            )
        finally:
            for state in snapshots:
                self.active_slot.runtime.restore_env_state(state)

        return {
            "setup_demo_success": setup_demo_success,
            "play_once_success": play_once_success,
            "cost_time": time.time() - started_at,
        }

    def close(self, clear_cache=True):
        return


class VectorEnv(gym.Env):
    def __init__(
        self,
        task_config,
        n_envs,
        env_seeds=None,
        instruction_type="seen",
    ):
        self.env_seeds = env_seeds
        if self.env_seeds is not None:
            assert len(self.env_seeds) == n_envs
        assets_path = os.getenv("ASSETS_PATH")
        self.task_name = task_config.get("task_name")

        head_camera_type = "D435"
        rdt_step = 10
        args = task_config

        args["planner_backend"] = args.get("planner_backend", "curobo") # Choices: [curobo, mplib]
        args["clear_cache_freq"] = max(1, int(args.get("clear_cache_freq", 8)))

        embodiment_type = args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        args["head_camera_h"] = _camera_config[head_camera_type]["h"]
        args["head_camera_w"] = _camera_config[head_camera_type]["w"]

        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise "No embodiment files"
            return robot_file

        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args

        if len(embodiment_type) == 1:
            args["left_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[1])
            )
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise "embodiment items should be 1 or 3"

        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_embodiment_config(
            args["right_robot_file"]
        )

        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = str(embodiment_type[0]) + "_" + str(embodiment_type[1])

        args["embodiment_name"] = embodiment_name

        args["rdt_step"] = rdt_step
        args["save_path"] += f"/{args['task_name']}_reward"

        args["n_envs"] = n_envs
        args["action_dim"] = 14

        args["eval_mode"] = True
        args["eval_video_log"] = False
        args["render_freq"] = 0

        self.args = args
        self.n_envs = n_envs
        self.gpu_sim = bool(args.get("gpu_sim", False))
        if self.gpu_sim:
            # This is task-construction state, not a new RLinf-facing option.
            # GPU rollouts use the CUDA image path for head and both wrists.
            # Tensor observations are opt-in because several legacy policies
            # still require NumPy/PIL images.
            args["_gpu_camera_readback"] = True
            args["_gpu_image_tensor_obs"] = bool(
                args.get("gpu_image_tensor_obs", False)
            )
        initial_seeds = self.env_seeds or list(range(n_envs))
        configured_pool = args.get("gpu_reset_seed_pool", [])
        self.gpu_seed_pool_by_env = [
            list(dict.fromkeys([*configured_pool, initial_seed]))
            for initial_seed in initial_seeds
        ]

        self.envs = []
        self.instruction_type = instruction_type

        self.global_lock = threading.Lock()

        self.env_thread_pool = ThreadPoolExecutor(max_workers=n_envs)
        self.gpu_runtime = (
            SharedGpuRuntime(
                num_envs=sum(len(seed_pool) for seed_pool in self.gpu_seed_pool_by_env),
                device=args.get("gpu_device", "cuda:0"),
                scene_spacing=float(args.get("gpu_scene_spacing", 10.0)),
                enable_render=True,
                profile=bool(args.get("gpu_profile", False)),
                defer_cpu_sync=bool(args.get("gpu_defer_cpu_sync", False)),
                parallel_topp=bool(args.get("gpu_parallel_topp", False)),
                parallel_topp_workers=args.get("gpu_parallel_topp_workers"),
            )
            if self.gpu_sim
            else None
        )

        self._init_envs()

    def _init_envs(self):
        if self.gpu_sim:
            physical_env_id = 0
            for i in range(self.n_envs):
                slots = []
                for seed in self.gpu_seed_pool_by_env[i]:
                    slot = GpuTaskSlot(
                        env_id=physical_env_id,
                        task_name=self.task_name,
                        args=self.args,
                        env_seed=seed,
                        instruction_type=self.instruction_type,
                        runtime=self.gpu_runtime,
                        validate_partial_resets=bool(
                            self.args.get("gpu_validate_partial_reset", False)
                        ),
                    )
                    slot.construct()
                    slots.append(slot)
                    physical_env_id += 1
                sub_env = GpuSubEnv(
                    env_id=i,
                    slots=slots,
                    initial_seed=self.env_seeds[i] if self.env_seeds else i,
                )
                self.envs.append(sub_env)
            self.gpu_runtime.initialize()
            for sub_env in self.envs:
                sub_env.finalize_construction()
            self.gpu_runtime.apply_articulation_controls()
            return
        for i in range(self.n_envs):
            sub_env = SubEnv(
                env_id=i,
                task_name=self.task_name,
                args=self.args,
                env_seed=self.env_seeds[i] if self.env_seeds else None,
                instruction_type=self.instruction_type,
                global_lock=self.global_lock,
            )
            sub_env.setup_task()
            self.envs.append(sub_env)

    def transform(self, results):
        res_dict = defaultdict(list)
        for res in results:
            for k, v in res.items():
                res_dict[k].append(v)

        res_dict = dict(res_dict)
        return (
            res_dict["obs"],
            res_dict["reward"],
            res_dict["terminated"],
            res_dict["truncated"],
            res_dict["info"],
        )

    def step(self, actions):
        if len(self.envs) == 0:
            self._init_envs()

        if self.gpu_sim:
            self.gpu_runtime.begin_batched_step(
                [sub_env.runtime_env_id for sub_env in self.envs]
            )
        step_futures = {}
        for i in range(self.n_envs):
            future = self.env_thread_pool.submit(self.envs[i].step, actions[i])
            step_futures[i] = future

        results = []
        try:
            for i in range(self.n_envs):
                future = step_futures[i]
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                except Exception as e:
                    raise RuntimeError(f"SubEnv {i} step error: {e}")
        finally:
            if self.gpu_sim:
                self.gpu_runtime.end_batched_step()

        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
            self.transform(results)
        )

        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv

    def reset(self, env_idx=None, env_seeds=None):
        if len(self.envs) == 0:
            self._init_envs()

        if env_idx is None:
            env_idx = list(range(self.n_envs))
        elif isinstance(env_idx, (list, tuple)):
            env_idx = list(env_idx)
        elif isinstance(env_idx, torch.Tensor):
            env_idx = env_idx.tolist()
        else:
            env_idx = [env_idx]

        if self.gpu_sim:
            for reset_position, idx in enumerate(env_idx):
                if 0 <= idx < self.n_envs:
                    seed = None
                    if env_seeds is not None and len(env_seeds) == len(env_idx):
                        seed = env_seeds[reset_position]
                    self.envs[idx].reset(env_seed=seed)
            return

        reset_futures = {}
        for idx in env_idx:
            if 0 <= idx < self.n_envs:
                seed = None
                if env_seeds is not None and len(env_seeds) == len(env_idx):
                    seed_idx = env_idx.index(idx)
                    seed = env_seeds[seed_idx]

                future = self.env_thread_pool.submit(
                    self.envs[idx].reset, env_seed=seed
                )
                reset_futures[idx] = future

        for idx in env_idx:
            if 0 <= idx < self.n_envs:
                future = reset_futures[idx]
                try:
                    future.result(timeout=120)
                except Exception as e:
                    raise RuntimeError(f"SubEnv {idx} reset error: {e}")

    def get_obs(self):
        if self.gpu_sim:
            active_slots = [sub_env.active_slot for sub_env in self.envs]
            can_batch_rgb = all(
                slot.task.data_type.get("rgb", True)
                and not any(
                    slot.task.data_type.get(name, False)
                    for name in (
                        "third_view",
                        "mesh_segmentation",
                        "actor_segmentation",
                        "depth",
                        "pointcloud",
                    )
                )
                and not getattr(slot.task, "crazy_random_light", False)
                for slot in active_slots
            )
            entries_by_env = (
                {
                    slot.env_id: slot.task.cameras.observation_camera_entries()
                    for slot in active_slots
                }
                if can_batch_rgb
                else {}
            )
            # RenderSystemGroup has a fixed setup/synchronization cost for
            # every compatible camera group.  Decide from the workload of each
            # group, rather than a hard-coded environment count or a sum over
            # unrelated cameras.  On the standard 320x240 head camera, four
            # scenes are slower when grouped and six scenes are faster.  The
            # same total pixels split across head/wrist groups does not
            # necessarily amortize each group's setup cost.
            camera_group_pixels = {}
            for entries in entries_by_env.values():
                for name, camera in entries:
                    group_key = (name, int(camera.width), int(camera.height))
                    camera_group_pixels[group_key] = (
                        camera_group_pixels.get(group_key, 0)
                        + int(camera.width) * int(camera.height)
                    )
            use_camera_group = any(
                pixels >= 460_800 for pixels in camera_group_pixels.values()
            )
            if use_camera_group:
                # Prepare poses and render state exactly once per active scene,
                # then capture matching cameras from all scenes together.
                for slot in active_slots:
                    slot.task._update_render(force=True)
                try:
                    rgba_by_env = self.gpu_runtime.capture_batched_rgba(
                        entries_by_env,
                        keep_cuda=bool(
                            self.args.get("gpu_image_tensor_obs", False)
                        ),
                    )
                    for slot in active_slots:
                        slot.task.cameras.set_batched_rgba(rgba_by_env[slot.env_id])
                except Exception:
                    # Keep the existing per-camera path available for uncommon
                    # renderer/camera combinations; no rollout interface changes.
                    for slot in active_slots:
                        slot.task.cameras._batched_rgba = None
                    use_camera_group = False

            if use_camera_group:
                # get_obs only packages the pre-captured image tensors/arrays;
                # it does not issue a second render. The result stays on CUDA
                # when ``gpu_image_tensor_obs`` is enabled for a compatible
                # model adapter.
                obs_futures = {
                    index: self.env_thread_pool.submit(env.get_obs)
                    for index, env in enumerate(self.envs)
                }
                observations = [None] * self.n_envs
                for index, future in obs_futures.items():
                    try:
                        observations[index] = future.result(timeout=120)
                    except Exception as error:
                        raise RuntimeError(
                            f"GPU SubEnv {index} observation error: {error}"
                        )
                return observations

        obs_venv = []
        for env in self.envs:
            obs_venv.append(env.get_obs())

        return obs_venv

    def close(self, clear_cache=True):
        for env in self.envs:
            env.close(clear_cache=clear_cache)

        if self.gpu_runtime is not None:
            self.gpu_runtime.close()

        if clear_cache:
            for env in self.envs:
                env = None
            self.envs = []
            gc.collect()
            torch.cuda.empty_cache()

    def check_seeds(self, seeds: list[int]):
        assert len(seeds) == self.n_envs
        if self.gpu_sim:
            return [env.check_seed(seed) for env, seed in zip(self.envs, seeds)]
        check_futures = {}
        for i in range(self.n_envs):
            future = self.env_thread_pool.submit(self.envs[i].check_seed, seeds[i])
            check_futures[i] = future

        results = [None] * self.n_envs
        for future in as_completed(check_futures.values(), timeout=120):
            for idx, f in check_futures.items():
                if f == future:
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        raise RuntimeError(f"SubEnv {idx} check seed error: {e}")
                    break

        return results


if __name__ == "__main__":
    mp.set_start_method("spawn")  # solve CUDA compatibility problem
    task_name = "place_shoe"
    n_envs = 4
    steps = 30
    horizon = 10
    action_dim = 14
    times = 10
    env = VectorEnv(task_name, n_envs, horizon)
    actions = np.zeros((n_envs, horizon, action_dim))
    for t in range(times):
        prev_obs_venv, reward_venv, truncation, termination, info_venv = (
            env.reset()
        )
        for step in range(steps):
            actions += np.random.randn(n_envs, horizon, action_dim) * 0.05
            actions = np.clip(actions, 0, 1)
            obs_venv, reward_venv, truncation, termination, info_venv = env.step(
                actions
            )

            # 测试partial reset功能
            if step % 10 == 0:
                # 重置所有环境
                env.reset()
            elif step % 5 == 0:
                # 只重置环境0和2
                env.reset(env_idx=[0, 2])
            
            obs = (
                env.get_obs()
            )
        env.close()
