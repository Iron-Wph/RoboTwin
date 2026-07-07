import importlib
import os
import gc
import sys

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


LOG_LEVEL = os.getenv("VECTOR_ENV_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("concurrent.futures").setLevel(logging.WARNING)
logging.getLogger("curobo").setLevel(logging.ERROR)
_LOGGED_SINGLE_ARM_STATE_KEYS = set()
_LOGGED_ACTION_DIM_CONFIGS = set()


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def update_obs(observation, args=None):
    raw_camera_obs = observation["observation"]
    full_image = raw_camera_obs["head_camera"]["rgb"]
    left_wrist_image = raw_camera_obs.get("left_camera", {}).get("rgb", None)
    right_wrist_image = raw_camera_obs.get("right_camera", {}).get("rgb", None)
    if args is not None and args.get("single_arm", False):
        active_arm = args.get("active_arm", "right")
        arm_key = f"{active_arm}_arm"
        gripper_key = f"{active_arm}_gripper"
        joint_action = observation["joint_action"]
        available_keys = tuple(joint_action.keys())
        missing_keys = [
            key for key in (arm_key, gripper_key) if key not in joint_action
        ]
        log_key = (active_arm, available_keys)
        if log_key not in _LOGGED_SINGLE_ARM_STATE_KEYS:
            _LOGGED_SINGLE_ARM_STATE_KEYS.add(log_key)
            logging.warning(
                "RoboTwin single-arm state key check: active_arm=%s, "
                "arm_key=%s exists=%s, gripper_key=%s exists=%s, "
                "available_joint_action_keys=%s",
                active_arm,
                arm_key,
                arm_key in joint_action,
                gripper_key,
                gripper_key in joint_action,
                list(available_keys),
            )
        if missing_keys:
            raise KeyError(
                "RoboTwin single-arm state key mismatch: "
                f"active_arm={active_arm}, missing_keys={missing_keys}, "
                f"available_joint_action_keys={list(available_keys)}"
            )
        state = np.array(
            list(joint_action[arm_key]) + [joint_action[gripper_key]],
            dtype=np.float32,
        )
        shape_log_key = (active_arm, state.shape)
        if shape_log_key not in _LOGGED_SINGLE_ARM_STATE_KEYS:
            _LOGGED_SINGLE_ARM_STATE_KEYS.add(shape_log_key)
            logging.warning(
                "RoboTwin single-arm state assembled: active_arm=%s, "
                "state_shape=%s, expected_dim=%d",
                active_arm,
                state.shape,
                len(joint_action[arm_key]) + 1,
            )
        if active_arm == "left":
            right_wrist_image = None
        else:
            left_wrist_image = None
    else:
        state = observation["joint_action"]["vector"]

    obs = {
        "full_image": full_image,
        "left_wrist_image": left_wrist_image,
        "right_wrist_image": right_wrist_image,
        "state": state,
    }
    if args is not None:
        for camera_name in args.get("debug_camera_names", []):
            camera_obs = raw_camera_obs.get(camera_name, None)
            if camera_obs is not None and camera_obs.get("rgb", None) is not None:
                obs[f"{camera_name}_image"] = camera_obs["rgb"]
    return obs


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
            obs = update_obs(self.task.get_obs(), self.args)
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
            obs = update_obs(obs, self.args)
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
            left_embodiment_config = get_embodiment_config(args["left_robot_file"])
            single_arm = args.get(
                "single_arm", left_embodiment_config.get("dual_arm") is False
            )
            args["single_arm"] = bool(single_arm)
            args["active_arm"] = args.get("active_arm", "right")
            if args["single_arm"]:
                args["dual_arm"] = False
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[1])
            )
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
            args["single_arm"] = bool(args.get("single_arm", False))
            args["active_arm"] = args.get("active_arm", "right")
            if args["single_arm"]:
                args["dual_arm"] = False
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
        left_arm_dim = len(args["left_embodiment_config"]["arm_joints_name"][0])
        right_arm_dim = len(args["right_embodiment_config"]["arm_joints_name"][1])
        if args.get("single_arm", False):
            active_arm = args.get("active_arm", "right")
            if active_arm not in ("left", "right"):
                raise ValueError(
                    f"active_arm must be 'left' or 'right', got {active_arm}"
                )
            selected_arm_dim = left_arm_dim if active_arm == "left" else right_arm_dim
            args["action_dim"] = selected_arm_dim + 1
            action_layout = f"{active_arm}_arm({selected_arm_dim}) + {active_arm}_gripper(1)"
        else:
            args["action_dim"] = left_arm_dim + 1 + right_arm_dim + 1
            active_arm = "dual"
            action_layout = (
                f"left_arm({left_arm_dim}) + left_gripper(1) + "
                f"right_arm({right_arm_dim}) + right_gripper(1)"
            )

        action_dim_log_key = (
            embodiment_name,
            bool(args.get("single_arm", False)),
            active_arm,
            left_arm_dim,
            right_arm_dim,
            args["action_dim"],
        )
        if action_dim_log_key not in _LOGGED_ACTION_DIM_CONFIGS:
            _LOGGED_ACTION_DIM_CONFIGS.add(action_dim_log_key)
            logging.warning(
                "RoboTwin action dim check: embodiment=%s, single_arm=%s, "
                "active_arm=%s, left_arm_dim=%d, right_arm_dim=%d, "
                "action_dim=%d, layout=%s",
                embodiment_name,
                bool(args.get("single_arm", False)),
                active_arm,
                left_arm_dim,
                right_arm_dim,
                args["action_dim"],
                action_layout,
            )

        args["eval_mode"] = True
        args["eval_video_log"] = False
        args["render_freq"] = 0

        self.args = args
        self.n_envs = n_envs

        self.envs = []
        self.instruction_type = instruction_type

        self.global_lock = threading.Lock()

        self.env_thread_pool = ThreadPoolExecutor(max_workers=n_envs)

        self._init_envs()

    def _init_envs(self):
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

        step_futures = {}
        for i in range(self.n_envs):
            future = self.env_thread_pool.submit(self.envs[i].step, actions[i])
            step_futures[i] = future

        results = []
        for i in range(self.n_envs):
            future = step_futures[i]
            try:
                result = future.result(timeout=120)
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"SubEnv {i} step error: {e}")

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
        obs_venv = []
        for env in self.envs:
            obs_venv.append(env.get_obs())

        return obs_venv

    def close(self, clear_cache=True):
        for env in self.envs:
            env.close(clear_cache=clear_cache)

        if clear_cache:
            for env in self.envs:
                env = None
            self.envs = []
            gc.collect()
            torch.cuda.empty_cache()

    def check_seeds(self, seeds: list[int]):
        assert len(seeds) == self.n_envs
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
    times = 10
    env = VectorEnv(task_name, n_envs, horizon)
    action_dim = env.args["action_dim"]
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
