"""Internal SAPIEN GPU runtime shared by one RoboTwin ``VectorEnv``.

``VectorEnv`` stays the only RLinf-facing interface.  This class only owns a
fixed set of local SAPIEN scenes and their GPU state snapshots.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
import threading
import time

import sapien
import torch


@dataclass(frozen=True)
class GpuEnvState:
    """GPU state required to restart one already-constructed environment."""

    env_id: int
    rigid_dynamic_components: tuple[sapien.physx.PhysxRigidDynamicComponent, ...]
    rigid_dynamic_rows: torch.Tensor
    rigid_dynamic_data: torch.Tensor
    articulations: tuple[sapien.physx.PhysxArticulation, ...]
    articulation_rows: torch.Tensor
    articulation_qpos: torch.Tensor
    articulation_qvel: torch.Tensor
    articulation_qf: torch.Tensor
    articulation_target_qpos: torch.Tensor
    articulation_target_qvel: torch.Tensor


class SharedGpuRuntime:
    """Own one GPU PhysX system and all scenes local to one EnvWorker.

    Scene components must be fully created before :meth:`initialize`.  After
    that point a reset restores state buffers; adding/removing a component is
    deliberately rejected by the surrounding vector-environment integration.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: str = "cuda:0",
        scene_spacing: float = 10.0,
        enable_render: bool = True,
        profile: bool = False,
        defer_cpu_sync: bool = False,
        parallel_topp: bool = False,
        parallel_topp_workers: int | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if scene_spacing <= 0:
            raise ValueError(f"scene_spacing must be positive, got {scene_spacing}")

        # GPU PhysX keeps contact/patch buffers globally.  The SAPIEN defaults
        # fit a small scene but overflow once a single system owns many
        # RoboTwin scenes.  Size the buffers from *physical* scenes (including
        # prebuilt reset slots) before creating the GPU system.  This is an
        # internal construction detail; VectorEnv's public API is unchanged.
        self.gpu_memory_config = self._scaled_gpu_memory_config(num_envs)
        sapien.physx.set_gpu_memory_config(**self.gpu_memory_config)
        if not sapien.physx.is_gpu_enabled():
            sapien.physx.enable_gpu()

        self.num_envs = num_envs
        self.device = sapien.Device(device)
        self.scene_spacing = scene_spacing
        # This is intentionally opt-in and is only enabled for tasks that
        # explicitly declare their chunk-boundary success semantics verified.
        # Most RoboTwin tasks query CPU SAPIEN state in ``check_success`` every
        # physics tick, so their exact fallback remains the default.
        self.defer_cpu_sync = defer_cpu_sync
        # TOPP belongs to one action-chunk's preparation phase, before the
        # shared PhysX scheduler begins its first tick.  A separate executor
        # avoids nested submission to VectorEnv's slot executor (which would
        # deadlock when every slot is already occupied).  It is deliberately
        # opt-in until real MPLib trajectory-equivalence tests have passed.
        if parallel_topp_workers is not None and parallel_topp_workers <= 0:
            raise ValueError("parallel_topp_workers must be positive when set")
        self.parallel_topp = bool(parallel_topp)
        default_topp_workers = min(
            os.cpu_count() or 1,
            max(2, 2 * num_envs),
        )
        self.parallel_topp_workers = (
            int(parallel_topp_workers)
            if parallel_topp_workers is not None
            else default_topp_workers
        )
        self._topp_executor = (
            ThreadPoolExecutor(
                max_workers=self.parallel_topp_workers,
                thread_name_prefix="robotwin-mplib-topp",
            )
            if self.parallel_topp
            else None
        )
        self.physx = sapien.physx.PhysxGpuSystem(device=self.device)
        self.scenes: list[sapien.Scene] = []
        self.render_systems: list[sapien.render.RenderSystem] = []
        self._env_rigid_components: list[list[sapien.physx.PhysxRigidDynamicComponent] | None] = [
            None for _ in range(num_envs)
        ]
        self._env_articulations: list[list[sapien.physx.PhysxArticulation] | None] = [
            None for _ in range(num_envs)
        ]
        self._registered_rigid_components: set[sapien.physx.PhysxRigidDynamicComponent] = set()
        self._registered_articulations: set[sapien.physx.PhysxArticulation] = set()
        self._initialized = False
        self._scheduler_condition = threading.Condition()
        self._scheduled_active: set[int] | None = None
        self._scheduled_finished: set[int] = set()
        self._scheduled_waiting: set[int] = set()
        self._scheduler_generation = 0
        self._batched_step_count = 0
        self._control_queue_lock = threading.Lock()
        self._queued_targets: list[
            tuple[int, tuple[int, ...], tuple[float, ...], tuple[float, ...] | None]
        ] = []
        self._queued_forces: list[tuple[int, tuple[float, ...]]] = []
        self._profile_enabled = profile
        self._profile_lock = threading.Lock()
        self._profile = {
            "slot_control_s": 0.0,
            "slot_observation_s": 0.0,
            "slot_calls": 0,
            "topp_s": 0.0,
            "topp_calls": 0,
            "physics_apply_s": 0.0,
            "physics_step_s": 0.0,
            "physics_sync_s": 0.0,
            "physics_sync_calls": 0,
            "physics_steps": 0,
            "trajectory_prepare_s": 0.0,
            "control_dispatch_s": 0.0,
            "success_check_s": 0.0,
            "render_update_s": 0.0,
            "camera_capture_s": 0.0,
            "camera_readback_s": 0.0,
            "camera_group_capture_s": 0.0,
            "camera_group_readback_s": 0.0,
            "camera_group_total_s": 0.0,
            "camera_group_calls": 0,
            "control_queue_flush_s": 0.0,
            "control_queue_flushes": 0,
            "control_queue_target_values": 0,
            "control_queue_force_values": 0,
        }

        grid_width = math.ceil(math.sqrt(num_envs))
        for env_id in range(num_envs):
            systems: list[sapien.System] = [self.physx]
            if enable_render:
                render_system = sapien.render.RenderSystem(self.device)
                self.render_systems.append(render_system)
                systems.append(render_system)
            scene = sapien.Scene(systems=systems)
            x = (env_id % grid_width) * scene_spacing
            y = (env_id // grid_width) * scene_spacing
            self.physx.set_scene_offset(scene, [x, y, 0.0])
            self.scenes.append(scene)

    @staticmethod
    def _scaled_gpu_memory_config(num_physical_envs: int) -> dict[str, int]:
        """Return SAPIEN GPU-buffer capacities for one shared PhysX system.

        Twenty-four fixed RoboTwin scenes complete with the SAPIEN defaults.
        Above that point we scale with a 25% safety margin.  In particular,
        this covers the capacities SAPIEN requested at 32 and 40 scenes during
        the Pi0 ``adjust_bottle`` benchmark, without hard-coding a task name.
        """
        if num_physical_envs <= 0:
            raise ValueError("num_physical_envs must be positive")
        reference_scenes = 24
        scale = 1.0
        if num_physical_envs > reference_scenes:
            scale = (num_physical_envs / reference_scenes) * 1.25

        defaults = {
            "temp_buffer_capacity": 16_777_216,
            "max_rigid_contact_count": 524_288,
            "max_rigid_patch_count": 81_920,
            "heap_capacity": 67_108_864,
            "found_lost_pairs_capacity": 262_144,
            "found_lost_aggregate_pairs_capacity": 1_024,
            "total_aggregate_pairs_capacity": 1_024,
            "collision_stack_size": 4_194_304,
        }
        return {
            name: int(math.ceil(capacity * scale))
            for name, capacity in defaults.items()
        }

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """Freeze scene topology and allocate SAPIEN GPU buffers."""
        if self._initialized:
            raise RuntimeError("SharedGpuRuntime is already initialized")
        if any(components is None for components in self._env_rigid_components):
            raise RuntimeError("call record_env_topology() for every environment first")
        self.physx.gpu_init()
        self._seed_initial_component_poses()
        self._initialized = True

    def prepare_for_cpu_query(self) -> None:
        """Refresh construction-time buffers needed by RoboTwin's planner.

        This may only run before the final :meth:`initialize`; it never advances
        physics and is not a runtime topology-change escape hatch.
        """
        if self._initialized:
            raise RuntimeError("GPU topology is fixed after initialize()")
        self.physx.gpu_init()

    def record_env_topology(self, env_id: int) -> None:
        """Associate components created for one slot with that slot before GPU init.

        A SAPIEN scene sharing a ``PhysxGpuSystem`` reports the system-wide actor
        collection.  Recording the components immediately after one task finishes
        construction is therefore the only reliable task-independent ownership map.
        """
        self._check_env_id(env_id)
        if self._initialized:
            raise RuntimeError("component topology is fixed after initialize()")
        if self._env_rigid_components[env_id] is not None:
            raise RuntimeError(f"environment {env_id} topology is already recorded")

        rigid_components = []
        for actor in self.scenes[env_id].get_all_actors():
            component = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if component is not None and component not in self._registered_rigid_components:
                rigid_components.append(component)
        articulations = [
            articulation
            for articulation in self.scenes[env_id].get_all_articulations()
            if articulation not in self._registered_articulations
        ]
        self._env_rigid_components[env_id] = rigid_components
        self._env_articulations[env_id] = articulations
        self._registered_rigid_components.update(rigid_components)
        self._registered_articulations.update(articulations)

    def capture_env_state(self, env_id: int) -> GpuEnvState:
        """Capture poses, velocities, joint states, and drive targets of one slot."""
        self._require_initialized()
        rigid_rows, articulation_rows = self._rows_for_env(env_id)
        return GpuEnvState(
            env_id=env_id,
            rigid_dynamic_components=tuple(self._env_rigid_components[env_id] or []),
            rigid_dynamic_rows=rigid_rows,
            rigid_dynamic_data=self._rows(self._tensor("cuda_rigid_dynamic_data"), rigid_rows),
            articulations=tuple(self._env_articulations[env_id] or []),
            articulation_rows=articulation_rows,
            articulation_qpos=self._rows(self._tensor("cuda_articulation_qpos"), articulation_rows),
            articulation_qvel=self._rows(self._tensor("cuda_articulation_qvel"), articulation_rows),
            articulation_qf=self._rows(self._tensor("cuda_articulation_qf"), articulation_rows),
            articulation_target_qpos=self._rows(
                self._tensor("cuda_articulation_target_qpos"), articulation_rows
            ),
            articulation_target_qvel=self._rows(
                self._tensor("cuda_articulation_target_qvel"), articulation_rows
            ),
        )

    def restore_env_state(self, state: GpuEnvState) -> None:
        """Restore exactly one fixed-topology slot from :meth:`capture_env_state`."""
        self._require_initialized()
        self._check_env_id(state.env_id)
        rigid_rows, articulation_rows = self._rows_for_components(
            state.rigid_dynamic_components, state.articulations
        )
        self._write_rows("cuda_rigid_dynamic_data", rigid_rows, state.rigid_dynamic_data)
        self._write_rows("cuda_articulation_qpos", articulation_rows, state.articulation_qpos)
        self._write_rows("cuda_articulation_qvel", articulation_rows, state.articulation_qvel)
        self._write_rows("cuda_articulation_qf", articulation_rows, state.articulation_qf)
        self._write_rows(
            "cuda_articulation_target_qpos",
            articulation_rows,
            state.articulation_target_qpos,
        )
        self._write_rows(
            "cuda_articulation_target_qvel",
            articulation_rows,
            state.articulation_target_qvel,
        )

        self.physx.gpu_apply_rigid_dynamic_data()
        self.physx.gpu_apply_articulation_qpos()
        self.physx.gpu_apply_articulation_qvel()
        self.physx.gpu_apply_articulation_qf()
        self.physx.gpu_apply_articulation_target_position()
        self.physx.gpu_apply_articulation_target_velocity()
        self.physx.gpu_update_articulation_kinematics()
        self.physx.sync_poses_gpu_to_cpu()

    def capture_unrelated_env_states(self, env_id: int) -> tuple[GpuEnvState, ...]:
        """Snapshot all GPU rows which a partial reset must leave untouched.

        This is intentionally a debug/validation primitive.  Normal resets do
        not clone every slot, while tests can prove that a slot-local reset
        never writes another slot's physics rows.
        """
        self._check_env_id(env_id)
        return tuple(
            self.capture_env_state(other_env_id)
            for other_env_id in range(self.num_envs)
            if other_env_id != env_id
        )

    def assert_partial_restore_isolated(
        self,
        expected_state: GpuEnvState,
        untouched_states: tuple[GpuEnvState, ...],
    ) -> None:
        """Assert a slot restore changed exactly the selected GPU rows."""
        self._require_initialized()
        self._assert_state_matches(
            self.capture_env_state(expected_state.env_id),
            expected_state,
            label=f"restored env {expected_state.env_id}",
            exact=False,
        )
        for untouched_state in untouched_states:
            self._assert_state_matches(
                self.capture_env_state(untouched_state.env_id),
                untouched_state,
                label=f"untouched env {untouched_state.env_id}",
                exact=True,
            )

    @staticmethod
    def _assert_state_matches(
        actual: GpuEnvState,
        expected: GpuEnvState,
        *,
        label: str,
        exact: bool,
    ) -> None:
        if actual.env_id != expected.env_id:
            raise AssertionError(f"{label}: mismatched environment id")
        tensor_fields = (
            "rigid_dynamic_rows",
            "rigid_dynamic_data",
            "articulation_rows",
            "articulation_qpos",
            "articulation_qvel",
            "articulation_qf",
            "articulation_target_qpos",
            "articulation_target_qvel",
        )
        for field in tensor_fields:
            actual_tensor = getattr(actual, field)
            expected_tensor = getattr(expected, field)
            # PhysX normalizes restored quaternion rows, producing harmless
            # one-ULP differences in the reset slot.  Rows belonging to every
            # other slot must remain bitwise identical.
            matches = (
                torch.equal(actual_tensor, expected_tensor)
                if exact
                else torch.allclose(
                    actual_tensor,
                    expected_tensor,
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
            )
            if not matches:
                raise AssertionError(f"{label}: GPU state differs in {field}")

    def apply_articulation_controls(self) -> None:
        """Commit the joint force and drive-target buffers written by every slot."""
        self._require_initialized()
        self._flush_queued_articulation_controls()
        self.physx.gpu_apply_articulation_qf()
        self.physx.gpu_apply_articulation_target_position()
        self.physx.gpu_apply_articulation_target_velocity()

    @property
    def batched_step_count(self) -> int:
        return self._batched_step_count

    @property
    def control_generation(self) -> int:
        """Current shared physics tick, used to avoid duplicate force writes."""
        return self._scheduler_generation

    @property
    def batched_step_active(self) -> bool:
        """Whether task-local control loops are currently synchronized as a batch."""
        return self._scheduled_active is not None

    def record_slot_timing(self, control_s: float, observation_s: float) -> None:
        """Record optional per-slot timings without changing rollout behavior."""
        if not self._profile_enabled:
            return
        with self._profile_lock:
            self._profile["slot_control_s"] += control_s
            self._profile["slot_observation_s"] += observation_s
            self._profile["slot_calls"] += 1

    def record_topp_timing(self, duration_s: float) -> None:
        if not self._profile_enabled:
            return
        with self._profile_lock:
            self._profile["topp_s"] += duration_s
            self._profile["topp_calls"] += 1

    def run_topp_pair(self, left_call, right_call):
        """Run independent left/right TOPP calls with preserved failures.

        Each arm owns a separate MPLib planner.  Returning exceptions instead
        of raising here preserves the legacy behavior: a failed left-arm TOPP
        still permits the right-arm fallback/trajectory to be evaluated.
        """

        def call_safely(callback):
            try:
                return callback(), None
            except Exception as error:  # legacy TOPP path converts errors to a fallback
                return None, error

        if self._topp_executor is None:
            return call_safely(left_call), call_safely(right_call)
        left_future = self._topp_executor.submit(call_safely, left_call)
        right_future = self._topp_executor.submit(call_safely, right_call)
        return left_future.result(), right_future.result()

    def close(self) -> None:
        """Release private CPU workers owned by this runtime."""
        if self._topp_executor is not None:
            self._topp_executor.shutdown(wait=True)
            self._topp_executor = None

    def record_profile_timing(self, name: str, duration_s: float) -> None:
        """Record an internal rollout phase without exposing a new env API."""
        if not self._profile_enabled:
            return
        with self._profile_lock:
            if name not in self._profile:
                raise KeyError(f"unknown GPU profile phase: {name}")
            self._profile[name] += duration_s

    def queue_articulation_targets(
        self,
        articulation_index: int,
        dof_offsets: list[int] | tuple[int, ...],
        positions,
        velocities=None,
    ) -> bool:
        """Queue one articulation control write for the next shared physics tick.

        Task code keeps its original control order and TOPP output.  Only the
        independent CUDA scatter writes are coalesced after every active slot
        reaches the shared scheduler barrier.
        """
        if not self.batched_step_active:
            return False
        offsets = tuple(int(offset) for offset in dof_offsets)
        position_values = tuple(float(value) for value in positions)
        velocity_values = (
            None if velocities is None else tuple(float(value) for value in velocities)
        )
        if not offsets or len(offsets) != len(position_values):
            raise ValueError("articulation targets require one position per DOF offset")
        if velocity_values is not None and len(velocity_values) != len(offsets):
            raise ValueError("articulation target velocities must match DOF offsets")
        with self._control_queue_lock:
            self._queued_targets.append(
                (int(articulation_index), offsets, position_values, velocity_values)
            )
        return True

    def queue_articulation_force(self, articulation_index: int, force) -> bool:
        """Queue one passive-force write for the next shared physics tick."""
        if not self.batched_step_active:
            return False
        force_values = tuple(float(value) for value in force)
        if not force_values:
            return True
        with self._control_queue_lock:
            self._queued_forces.append((int(articulation_index), force_values))
        return True

    def profile_snapshot(self) -> dict[str, float | int]:
        """Return aggregate timings collected when ``profile=True`` was requested."""
        with self._profile_lock:
            return dict(self._profile)

    def capture_batched_rgba(self, camera_entries_by_env, *, keep_cuda: bool = False):
        """Capture RGB from active scene cameras in SAPIEN camera groups.

        The caller supplies only current logical slots, so prebuilt reset scenes
        that are inactive in this VectorEnv action are never rendered. CUDA
        image tensors are retained when the private RLinf adapter can consume
        them; otherwise the original ``uint8`` NumPy representation is kept.
        """
        self._require_initialized()
        if not camera_entries_by_env:
            return {}

        active_env_ids = list(camera_entries_by_env)
        if len(set(active_env_ids)) != len(active_env_ids):
            raise ValueError("each batched camera environment must appear once")
        for env_id in active_env_ids:
            self._check_env_id(env_id)

        render_group = sapien.render.RenderSystemGroup(
            [self.scenes[env_id].get_render_system() for env_id in active_env_ids]
        )
        grouped_cameras = {}
        for env_id in active_env_ids:
            for camera_name, camera in camera_entries_by_env[env_id]:
                group_key = (camera_name, int(camera.width), int(camera.height))
                grouped_cameras.setdefault(group_key, []).append(
                    (env_id, camera_name, camera)
                )

        output = {env_id: {} for env_id in active_env_ids}
        total_started_at = time.perf_counter() if self._profile_enabled else None
        capture_s = 0.0
        readback_s = 0.0
        for cameras in grouped_cameras.values():
            camera_group = render_group.create_camera_group(
                [camera for _, _, camera in cameras], ["Color"]
            )
            started_at = time.perf_counter() if self._profile_enabled else None
            camera_group.take_picture()
            # RenderCameraGroup submits work on SAPIEN's CUDA stream.  PyTorch
            # CPU copies do not implicitly wait for that stream, so synchronize
            # once per camera group before exposing an image to RLinf.
            torch.cuda.synchronize(self.device)
            after_capture = time.perf_counter() if self._profile_enabled else None
            rgba = camera_group.get_picture_cuda("Color").torch()
            rgba = rgba.mul(255).clamp(0, 255).to(torch.uint8)
            if self._profile_enabled:
                assert started_at is not None and after_capture is not None
                capture_s += after_capture - started_at
            if keep_cuda:
                for (env_id, camera_name, _), image in zip(cameras, rgba):
                    output[env_id][camera_name] = image
            else:
                copied_rgba = rgba.cpu().numpy()
                for (env_id, camera_name, _), image in zip(cameras, copied_rgba):
                    output[env_id][camera_name] = image
            if self._profile_enabled:
                readback_s += time.perf_counter() - after_capture

        # RenderSystemGroup owns SAPIEN-side synchronization.  Release it
        # before profiling so deferred work is charged to this camera phase.
        del render_group

        if self._profile_enabled:
            assert total_started_at is not None
            with self._profile_lock:
                self._profile["camera_group_capture_s"] += capture_s
                self._profile["camera_group_readback_s"] += readback_s
                self._profile["camera_group_total_s"] += (
                    time.perf_counter() - total_started_at
                )
                self._profile["camera_group_calls"] += len(grouped_cameras)
        return output

    def begin_batched_step(self, env_ids: list[int]) -> None:
        """Start one VectorEnv action call with the specified active slots."""
        self._require_initialized()
        active = set(env_ids)
        if not active:
            return
        if any(env_id < 0 or env_id >= self.num_envs for env_id in active):
            raise IndexError("batched step includes an invalid environment index")
        # A stale command would otherwise be applied to the first tick of a
        # different VectorEnv.step() call.  Check before publishing the new
        # scheduler state so an exception leaves the scheduler untouched.
        with self._control_queue_lock:
            if self._queued_targets or self._queued_forces:
                raise RuntimeError("control queue was not drained before a new GPU batch")
        with self._scheduler_condition:
            if self._scheduled_active is not None:
                raise RuntimeError("a GPU batched step is already active")
            self._scheduled_active = active
            self._scheduled_finished = set()
            self._scheduled_waiting = set()

    def step_slot(self, env_id: int) -> None:
        """Synchronize a task-local control tick into exactly one shared PhysX step."""
        with self._scheduler_condition:
            if self._scheduled_active is None or env_id not in self._scheduled_active:
                raise RuntimeError("slot stepped outside an active GPU batch")
            if env_id in self._scheduled_finished:
                raise RuntimeError("a finished slot attempted another physics step")
            generation = self._scheduler_generation
            self._scheduled_waiting.add(env_id)
            self._advance_when_ready_locked()
            while generation == self._scheduler_generation:
                self._scheduler_condition.wait()

    def finish_batched_slot(self, env_id: int) -> None:
        """Mark a task as complete so longer trajectories can keep stepping."""
        with self._scheduler_condition:
            if self._scheduled_active is None or env_id not in self._scheduled_active:
                return
            self._scheduled_finished.add(env_id)
            self._scheduled_waiting.discard(env_id)
            self._advance_when_ready_locked()
            if self._scheduled_finished == self._scheduled_active:
                if self.defer_cpu_sync:
                    self._sync_poses_gpu_to_cpu_locked()
                self._scheduled_active = None
                self._scheduler_condition.notify_all()

    def wait_for_cpu_sync(self) -> None:
        """Wait until a deferred batch has published its final CPU poses.

        Only ``GpuTaskSlot`` uses this private barrier.  It lets all slots
        finish their GPU-only control loop before any of them asks RoboTwin for
        an observation or performs its chunk-boundary CPU success fallback.
        """
        if not self.defer_cpu_sync:
            return
        with self._scheduler_condition:
            while self._scheduled_active is not None:
                self._scheduler_condition.wait()

    def end_batched_step(self) -> None:
        """Check that every task submitted for an action call left the scheduler."""
        with self._scheduler_condition:
            if self._scheduled_active is not None:
                raise RuntimeError("not every GPU slot completed the batched step")

    def _advance_when_ready_locked(self) -> None:
        assert self._scheduled_active is not None
        pending = self._scheduled_active - self._scheduled_finished
        if pending and pending.issubset(self._scheduled_waiting):
            started_at = time.perf_counter() if self._profile_enabled else None
            self.apply_articulation_controls()
            after_apply = time.perf_counter() if self._profile_enabled else None
            self.physx.step()
            after_step = time.perf_counter() if self._profile_enabled else None
            if not self.defer_cpu_sync:
                self._sync_poses_gpu_to_cpu_locked()
            if self._profile_enabled:
                assert started_at is not None and after_apply is not None and after_step is not None
                self._profile["physics_apply_s"] += after_apply - started_at
                self._profile["physics_step_s"] += after_step - after_apply
                self._profile["physics_steps"] += 1
            self._batched_step_count += 1
            self._scheduled_waiting.clear()
            self._scheduler_generation += 1
            self._scheduler_condition.notify_all()

    def _sync_poses_gpu_to_cpu_locked(self) -> None:
        """Synchronize PhysX poses once and charge the operation to profiling."""
        started_at = time.perf_counter() if self._profile_enabled else None
        self.physx.sync_poses_gpu_to_cpu()
        if self._profile_enabled:
            assert started_at is not None
            self._profile["physics_sync_s"] += time.perf_counter() - started_at
            self._profile["physics_sync_calls"] += 1

    def _flush_queued_articulation_controls(self) -> None:
        """Scatter all task-local controls once per shared scheduler generation."""
        with self._control_queue_lock:
            queued_targets = self._queued_targets
            queued_forces = self._queued_forces
            self._queued_targets = []
            self._queued_forces = []
        if not queued_targets and not queued_forces:
            return

        started_at = time.perf_counter() if self._profile_enabled else None
        target_value_count = 0
        force_value_count = 0
        if queued_targets:
            rows: list[int] = []
            columns: list[int] = []
            positions: list[float] = []
            velocity_rows: list[int] = []
            velocity_columns: list[int] = []
            velocities: list[float] = []
            for articulation_index, offsets, target_positions, target_velocities in queued_targets:
                rows.extend([articulation_index] * len(offsets))
                columns.extend(offsets)
                positions.extend(target_positions)
                if target_velocities is not None:
                    velocity_rows.extend([articulation_index] * len(offsets))
                    velocity_columns.extend(offsets)
                    velocities.extend(target_velocities)
            target_qpos = self._tensor("cuda_articulation_target_qpos")
            row_tensor = torch.as_tensor(rows, device=target_qpos.device, dtype=torch.long)
            column_tensor = torch.as_tensor(columns, device=target_qpos.device, dtype=torch.long)
            target_qpos[row_tensor, column_tensor] = torch.as_tensor(
                positions, device=target_qpos.device, dtype=target_qpos.dtype
            )
            if velocities:
                target_qvel = self._tensor("cuda_articulation_target_qvel")
                velocity_row_tensor = torch.as_tensor(
                    velocity_rows, device=target_qvel.device, dtype=torch.long
                )
                velocity_column_tensor = torch.as_tensor(
                    velocity_columns, device=target_qvel.device, dtype=torch.long
                )
                target_qvel[velocity_row_tensor, velocity_column_tensor] = torch.as_tensor(
                    velocities, device=target_qvel.device, dtype=target_qvel.dtype
                )
            target_value_count = len(positions) + len(velocities)

        if queued_forces:
            force_rows: list[int] = []
            force_columns: list[int] = []
            force_values: list[float] = []
            for articulation_index, force in queued_forces:
                force_rows.extend([articulation_index] * len(force))
                force_columns.extend(range(len(force)))
                force_values.extend(force)
            qf = self._tensor("cuda_articulation_qf")
            qf[
                torch.as_tensor(force_rows, device=qf.device, dtype=torch.long),
                torch.as_tensor(force_columns, device=qf.device, dtype=torch.long),
            ] = torch.as_tensor(force_values, device=qf.device, dtype=qf.dtype)
            force_value_count = len(force_values)

        if self._profile_enabled:
            assert started_at is not None
            with self._profile_lock:
                self._profile["control_queue_flush_s"] += time.perf_counter() - started_at
                self._profile["control_queue_flushes"] += 1
                self._profile["control_queue_target_values"] += target_value_count
                self._profile["control_queue_force_values"] += force_value_count

    def step(self) -> None:
        """Advance all local scenes exactly once, then expose their CPU poses."""
        self._require_initialized()
        self.physx.step()
        self.physx.sync_poses_gpu_to_cpu()

    def _rows_for_env(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_env_id(env_id)
        rigid_components = self._env_rigid_components[env_id]
        articulations = self._env_articulations[env_id]
        if rigid_components is None or articulations is None:
            raise RuntimeError(f"environment {env_id} topology was not recorded")
        return self._rows_for_components(rigid_components, articulations)

    def _seed_initial_component_poses(self) -> None:
        """Copy construction-time CPU poses into the newly allocated GPU buffers."""
        rigid_data = self._tensor("cuda_rigid_dynamic_data")
        for components in self._env_rigid_components:
            assert components is not None
            for component in components:
                pose = component.get_entity_pose()
                rigid_data[component.gpu_pose_index, :3] = torch.as_tensor(
                    pose.p, device=rigid_data.device, dtype=rigid_data.dtype
                )
                rigid_data[component.gpu_pose_index, 3:7] = torch.as_tensor(
                    pose.q, device=rigid_data.device, dtype=rigid_data.dtype
                )
        self.physx.gpu_apply_rigid_dynamic_data()

        rigid_body_data = self._tensor("cuda_rigid_body_data")
        for articulations in self._env_articulations:
            assert articulations is not None
            for articulation in articulations:
                root_link = articulation.get_links()[0]
                pose = root_link.get_entity_pose()
                rigid_body_data[root_link.gpu_pose_index, :3] = torch.as_tensor(
                    pose.p, device=rigid_body_data.device, dtype=rigid_body_data.dtype
                )
                rigid_body_data[root_link.gpu_pose_index, 3:7] = torch.as_tensor(
                    pose.q, device=rigid_body_data.device, dtype=rigid_body_data.dtype
                )
        self.physx.gpu_apply_articulation_root_pose()
        self.physx.sync_poses_gpu_to_cpu()

    def _rows_for_components(
        self,
        rigid_components: tuple[sapien.physx.PhysxRigidDynamicComponent, ...] | list[sapien.physx.PhysxRigidDynamicComponent],
        articulations: tuple[sapien.physx.PhysxArticulation, ...] | list[sapien.physx.PhysxArticulation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._tensor("cuda_rigid_dynamic_data").device
        return (
            torch.tensor(
                sorted(component.gpu_pose_index for component in rigid_components),
                device=device,
                dtype=torch.long,
            ),
            torch.tensor(
                sorted(articulation.gpu_index for articulation in articulations),
                device=device,
                dtype=torch.long,
            ),
        )

    def _write_rows(self, name: str, rows: torch.Tensor, values: torch.Tensor) -> None:
        target = self._tensor(name)
        if rows.ndim != 1 or values.shape[0] != rows.numel():
            raise ValueError(f"invalid snapshot rows for {name}")
        if rows.numel() == 0:
            return
        if rows.min().item() < 0 or rows.max().item() >= target.shape[0]:
            raise IndexError(f"snapshot row is outside {name}")
        target[rows] = values.to(device=target.device, dtype=target.dtype)

    def _rows(self, data: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return data[rows].clone()

    def _tensor(self, name: str) -> torch.Tensor:
        return getattr(self.physx, name).torch()

    def _check_env_id(self, env_id: int) -> None:
        if not 0 <= env_id < self.num_envs:
            raise IndexError(f"invalid environment index {env_id}")

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("call initialize() after scene construction first")
