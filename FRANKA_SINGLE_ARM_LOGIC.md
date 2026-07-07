# Franka-Panda 单臂处理逻辑说明

本文档说明当前 `franka-panda` 单臂支持的实现方式，以及它和原始双臂逻辑的区别。这里描述的是当前分支 `franka-panda-single-arm-rl` 上已经提交的实现，不是最终推荐的理想重构形态。

## 总览

当前 RoboTwin 里的单臂支持分成三层：

1. `robotwin/envs/vector_env.py`
   - 根据 `task_config.single_arm` 和 `task_config.active_arm` 决定 env 对外暴露的 `action_dim` 和 policy state。
2. `envs/_base_task.py`
   - 在任务基础类里区分单臂/双臂。
   - 单臂动作只包含 active arm 的关节和夹爪。
   - 双臂动作仍然使用原始 `left + left_gripper + right + right_gripper` 格式。
3. `envs/robot/robot.py`
   - 在 robot 层保存 `active_arm`。
   - 单臂 homestate 和夹爪状态查询映射到 active arm。

对 `franka-panda` 单臂来说，动作维度是：

```text
7 arm joints + 1 gripper = 8 dims
```

双臂模式仍然是：

```text
left_arm_dim + 1 left_gripper + right_arm_dim + 1 right_gripper
```

如果左右都是 7 关节，则双臂是：

```text
7 + 1 + 7 + 1 = 16 dims
```

## 配置入口

单臂通过 task config 控制：

```yaml
task_config:
  embodiment: [franka-panda]
  single_arm: true
  active_arm: right
  dual_arm: false
```

关键语义：

- `single_arm: true`：启用单臂 env 行为。
- `active_arm: right`：选择 RoboTwin 内部 left/right API 槽位中的哪一个作为当前单臂。
- `dual_arm: false`：任务基础类里不走双臂动作拆分。

`active_arm` 不等于“物体一定放左边/右边”。它只是单臂模式下选择使用哪一套 planner/gripper/joint API。

例如 `place_empty_cup` 里的物体随机性仍然保留：

```python
def load_actors(self):
    tag = np.random.randint(0, 2)
    cup_xlim = [[0.15, 0.3], [-0.3, -0.15]]
    coaster_lim = [[-0.05, 0.1], [-0.1, 0.05]]
```

也就是说，单臂不会因为 `active_arm=right` 就强制把物体放右边。

## VectorEnv 层

`robotwin/envs/vector_env.py` 负责把 RoboTwin task 包装成外部 RL env。

### state 输出

当前 `update_obs()` 对单臂和双臂的 state 输出不同：

```python
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
        state = np.array(
            list(joint_action[arm_key]) + [joint_action[gripper_key]],
            dtype=np.float32,
        )
        if active_arm == "left":
            right_wrist_image = None
        else:
            left_wrist_image = None
    else:
        state = observation["joint_action"]["vector"]
```

区别：

- 单臂：`state = active_arm_arm + active_arm_gripper`
- 双臂：`state = observation["joint_action"]["vector"]`

对 Franka-Panda 单臂，state 是 8 维。

### action_dim 输出

`VectorEnv` 根据 `single_arm` 动态设置 `args["action_dim"]`：

```python
left_arm_dim = len(args["left_embodiment_config"]["arm_joints_name"][0])
right_arm_dim = len(args["right_embodiment_config"]["arm_joints_name"][1])
if args.get("single_arm", False):
    active_arm = args.get("active_arm", "right")
    args["action_dim"] = (left_arm_dim if active_arm == "left" else right_arm_dim) + 1
else:
    args["action_dim"] = left_arm_dim + 1 + right_arm_dim + 1
```

因此：

- `single_arm=true, active_arm=right`：`right_arm_dim + 1`
- `single_arm=true, active_arm=left`：`left_arm_dim + 1`
- `single_arm=false`：`left_arm_dim + 1 + right_arm_dim + 1`

## Base_Task 初始化

`envs/_base_task.py` 在 `_init_task_env_()` 中保存单臂配置：

```python
self.single_arm = kwags.get("single_arm", False)
self.active_arm = kwags.get("active_arm", "right")
if self.active_arm not in ["left", "right"]:
    raise ValueError(f"active_arm must be 'left' or 'right', not {self.active_arm}")
self.dual_arm = False if self.single_arm else kwags.get("dual_arm", True)
```

这段逻辑的效果是：

- 只要 `single_arm=True`，`self.dual_arm` 强制为 `False`。
- 双臂行为只在 `single_arm=False` 时保留。
- `active_arm` 只允许 `"left"` 或 `"right"`。

## 单臂动作校验

单臂 action 维度由 active arm 的 joint state 长度决定：

```python
def _single_arm_action_dim(self):
    return len(self._arm_joint_state(self._active_arm()))
```

`_arm_joint_state()` 返回的是：

```python
def _arm_joint_state(self, arm_tag):
    if arm_tag == "left":
        return self.robot.get_left_arm_jointState()
    if arm_tag == "right":
        return self.robot.get_right_arm_jointState()
    raise ValueError(f"arm_tag must be 'left' or 'right', not {arm_tag}")
```

`get_*_arm_jointState()` 本身包含：

```text
arm joints + gripper
```

所以对 Franka-Panda 来说就是 8 维。

动作进入单臂 executor 前会被校验：

```python
def _prepare_single_arm_actions(self, chunk_actions):
    actions = np.asarray(chunk_actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    expected_dim = self._single_arm_action_dim()
    if actions.shape[-1] != expected_dim:
        raise ValueError(
            f"single-arm action dimension mismatch for {self._active_arm()} arm: "
            f"expected {expected_dim}, got {actions.shape[-1]}"
        )
    return actions
```

这和双臂的区别是：

- 单臂只检查 `active_arm_dim + 1`
- 双臂检查 `left_dim + 1 + right_dim + 1`

## 单臂 qpos 执行路径

当前单臂核心执行函数是 `_execute_single_arm_qpos_actions()`：

```python
def _execute_single_arm_qpos_actions(self, actions, downsample=True):
    actions = self._prepare_single_arm_actions(actions)
    arm_tag = self._active_arm()
    jointstate = self._arm_joint_state(arm_tag)
    arm_dim = len(jointstate) - 1

    arm_actions = actions[:, :arm_dim]
    gripper_actions = actions[:, arm_dim]
    current_qpos = np.array(jointstate[:arm_dim])
    current_gripper = np.array(jointstate[arm_dim:arm_dim + 1])

    arm_path = np.vstack((current_qpos, arm_actions))
    gripper_path = np.hstack((current_gripper, gripper_actions))
    arm_path = self.compress_path(arm_path)

    topp_flag, result, n_step = self._plan_single_arm_qpos_path(
        arm_tag, arm_path, downsample=downsample
    )
    gripper = self._interpolate_gripper_path(
        gripper_path, n_step, len(gripper_actions)
    )

    now_id = 0
    while now_id < n_step:
        if topp_flag:
            self._set_arm_joints(
                arm_tag,
                result["position"][now_id],
                result["velocity"][now_id],
            )
        if not self.fix_gripper:
            self._set_arm_gripper(arm_tag, gripper[now_id])

        now_id += 1
        self.scene.step()
        self._update_render()

        if self.check_success():
            self.eval_success = True
            return True
    return False
```

执行步骤：

1. 校验 action 维度必须是单臂维度。
2. 根据 `active_arm` 读取当前 qpos。
3. 拆出：
   - `arm_actions`
   - `gripper_actions`
4. 构建 active arm 的轨迹。
5. 只对 active arm 调 TOPP。
6. 控制循环里只调用：
   - `_set_arm_joints(active_arm, ...)`
   - `_set_arm_gripper(active_arm, ...)`
7. 每步 `scene.step()` 和 `_update_render()`。
8. 如果 `check_success()` 成功，设置 `eval_success=True`。

这个路径不会构造 inactive arm 的动作，也不会把单臂动作 expand 成双臂动作。

## 双臂 qpos 执行路径

双臂路径仍然是旧逻辑。核心动作布局如下：

```python
left_jointstate = self.robot.get_left_arm_jointState()
right_jointstate = self.robot.get_right_arm_jointState()
left_arm_dim = len(left_jointstate) - 1
right_arm_dim = len(right_jointstate) - 1
current_jointstate = np.array(left_jointstate + right_jointstate)

left_arm_actions, left_gripper_actions = (
    actions[:, :left_arm_dim],
    actions[:, left_arm_dim],
)
right_arm_actions, right_gripper_actions = (
    actions[:, left_arm_dim + 1:left_arm_dim + right_arm_dim + 1],
    actions[:, left_arm_dim + right_arm_dim + 1],
)
```

双臂布局是：

```text
[left_arm_qpos, left_gripper, right_arm_qpos, right_gripper]
```

然后分别构造：

```python
left_path = np.vstack((left_current_qpos, left_arm_actions))
right_path = np.vstack((right_current_qpos, right_arm_actions))
```

再分别做 TOPP：

```python
times, left_pos, left_vel, acc, duration = self.robot.left_mplib_planner.TOPP(...)
times, right_pos, right_vel, acc, duration = self.robot.right_mplib_planner.TOPP(...)
```

控制循环里两个臂按进度交替执行：

```python
if now_left_id < left_n_step:
    self.robot.set_arm_joints(..., "left")
    self.robot.set_gripper(..., "left")

if now_right_id < right_n_step:
    self.robot.set_arm_joints(..., "right")
    self.robot.set_gripper(..., "right")
```

## sparse reward 路径

当前 `gen_sparse_reward_data()` 会先判断是否单臂：

```python
def gen_sparse_reward_data(self, chunk_actions, action_type="qpos"):
    if self._single_arm_enabled():
        return self._gen_sparse_reward_data_single_arm(chunk_actions, action_type)

    infos = {
        "success": False,
    }
    ...
```

单臂 sparse reward 当前走 `_gen_sparse_reward_data_single_arm()`：

```python
def _gen_sparse_reward_data_single_arm(self, chunk_actions, action_type="qpos"):
    if action_type != "qpos":
        raise NotImplementedError("single-arm RoboTwin RL currently supports qpos actions only")

    infos = {
        "success": False,
    }
    reward = np.array([0], dtype=np.float32)
    termination = np.array([0], dtype=np.int32)
    truncation = np.array([0], dtype=np.int32)

    if getattr(self, "eval_success", False):
        infos["success"] = True
        reward = np.array([1], dtype=np.float32)
        termination = np.array([1], dtype=np.int32)
        return reward, termination, truncation, infos

    if self.take_action_cnt == self.step_lim:
        truncation = np.array([1], dtype=np.int32)
        return reward, termination, truncation, infos

    actions = self._prepare_single_arm_actions(chunk_actions)
    self.take_action_cnt += actions.shape[0]

    self._update_render()
    if self.render_freq:
        self.viewer.render()

    if self._execute_single_arm_qpos_actions(actions, downsample=True):
        infos["success"] = True
        reward = np.array([1], dtype=np.float32)
        termination = np.array([1], dtype=np.int32)
        return reward, termination, truncation, infos
```

区别：

- 单臂：先校验 8D，然后只控制 active arm。
- 双臂：继续使用旧的 16D 双臂拆分逻辑。

## take_action 路径

`take_action()` 也在入口处分支：

```python
def take_action(self, action, action_type:Literal['qpos', 'ee']='qpos'):
    if self._single_arm_enabled():
        if action_type != "qpos":
            raise NotImplementedError("single-arm RoboTwin RL currently supports qpos actions only")
        if self.take_action_cnt == self.step_lim or self.eval_success:
            return
        ...
        actions = self._prepare_single_arm_actions(action)
        self.take_action_cnt += actions.shape[0]
        ...
        success = self._execute_single_arm_qpos_actions(actions, downsample=False)
        ...
        return

    if self.take_action_cnt == self.step_lim or self.eval_success:
        return

    ...
```

区别：

- 单臂 `take_action()` 走 `_execute_single_arm_qpos_actions(..., downsample=False)`。
- 双臂 `take_action()` 走原始双臂拆分和双臂控制循环。

## dense reward 路径

当前 `gen_dense_reward_data()` 和 `gen_dense_reward_once()` 里也加了单臂分支：

```python
if self._single_arm_enabled():
    self._execute_single_arm_qpos_actions(actions, downsample=False)
    self._append_episode_arm_traces()
    ...
    continue
```

`gen_dense_reward_once()` 里类似：

```python
if self._single_arm_enabled():
    self._execute_single_arm_qpos_actions(actions, downsample=True)
    if step > chunk_actions.shape[0] - 3:
        obs_return.append(self.get_obs())

    self._update_render()
    self._append_episode_arm_traces()

    self.reward.update()
    ...
    continue
```

也就是说：

- 单臂 dense reward 不再走旧的 `left_path/right_path` 双臂拆分。
- 单臂直接使用 `_execute_single_arm_qpos_actions()`。
- 双臂仍然保留原始路径。

## episode trace 记录

dense reward 里原来有很多重复代码：

```python
self.episode_left_eef_poses = [self.robot.get_left_ee_pose()]
self.episode_right_eef_poses = [self.robot.get_right_ee_pose()]
self.episode_left_joint_states = [self.robot.get_left_arm_jointState()]
self.episode_right_joint_states = [self.robot.get_right_arm_jointState()]
self.episode_left_gripper_state = [self.robot.is_left_gripper_open()]
self.episode_right_gripper_state = [self.robot.is_right_gripper_open()]
```

以及循环后 append：

```python
self.episode_left_eef_poses.append(np.array(self.robot.get_left_ee_pose()))
self.episode_right_eef_poses.append(np.array(self.robot.get_right_ee_pose()))
self.episode_left_joint_states.append(np.array(self.robot.get_left_arm_jointState()))
self.episode_right_joint_states.append(np.array(self.robot.get_right_arm_jointState()))
self.episode_left_gripper_state.append(self.robot.is_left_gripper_open())
self.episode_right_gripper_state.append(self.robot.is_right_gripper_open())
```

当前实现把它抽成了：

```python
def _reset_episode_arm_traces(self):
    ...

def _append_episode_arm_traces(self):
    ...
```

底层 snapshot 是：

```python
def _current_episode_arm_snapshots(self):
    if self._single_arm_enabled():
        arm_tag = self._active_arm()
        eef_pose = np.array(self._arm_ee_pose(arm_tag))
        joint_state = np.array(self._arm_joint_state(arm_tag))
        gripper_open = self._arm_gripper_open(arm_tag)
        return (
            eef_pose,
            eef_pose.copy(),
            joint_state,
            joint_state.copy(),
            gripper_open,
            gripper_open,
        )

    return (
        np.array(self.robot.get_left_ee_pose()),
        np.array(self.robot.get_right_ee_pose()),
        np.array(self.robot.get_left_arm_jointState()),
        np.array(self.robot.get_right_arm_jointState()),
        self.robot.is_left_gripper_open(),
        self.robot.is_right_gripper_open(),
    )
```

这里的区别很关键：

- 双臂：left trace 记录左臂，right trace 记录右臂。
- 单臂：left trace 和 right trace 都记录 active arm 的同一份状态。

为什么单臂要这么做：

原有 reward 代码经常默认存在 `episode_left_*` 和 `episode_right_*` 两套轨迹，并从左右臂里选择离物体更近的一侧。如果单臂时只写一侧，另一侧可能是冻结的无效状态，reward 会误判。所以当前实现让两侧 trace 都等于 active arm。

这不是最干净的设计，但可以兼容旧 reward 接口。

## gripper 状态查询

很多任务里的 `check_success()` 会写成：

```python
return (
    ...
    and self.is_left_gripper_open()
    and self.is_right_gripper_open()
)
```

单臂模式下，如果仍然分别检查左右夹爪，会导致 inactive arm 状态影响 success。

所以 `Base_Task` 层做了 wrapper：

```python
def is_left_gripper_open(self):
    if self._single_arm_enabled():
        return self._arm_gripper_open(self._active_arm())
    return self.robot.is_left_gripper_open()

def is_right_gripper_open(self):
    if self._single_arm_enabled():
        return self._arm_gripper_open(self._active_arm())
    return self.robot.is_right_gripper_open()
```

`Robot` 层也做了类似映射：

```python
def _status_gripper_val(self, arm_tag):
    if self._single_arm_enabled():
        arm_tag = self._active_arm()
    return self.left_gripper_val if arm_tag == "left" else self.right_gripper_val

def is_left_gripper_open(self):
    return self._status_gripper_val("left") > 0.8

def is_right_gripper_open(self):
    return self._status_gripper_val("right") > 0.8
```

这样即使任务代码直接调用 `self.robot.is_left_gripper_open()` 和 `self.robot.is_right_gripper_open()`，单臂模式下也会映射到 active arm。

## Robot 层 homestate

原始双臂逻辑会设置左右两套 homestate：

```python
for i, joint in enumerate(self.left_arm_joints):
    joint.set_drive_target(self.left_homestate[i])

for i, joint in enumerate(self.right_arm_joints):
    joint.set_drive_target(self.right_homestate[i])
```

单臂模式下改为只设置 active arm：

```python
def move_to_homestate(self):
    if self._single_arm_enabled():
        if self._active_arm() == "left":
            joint_list = self.left_arm_joints
            homestate = self.left_homestate
        else:
            joint_list = self.right_arm_joints
            homestate = self.right_homestate
        for i, joint in enumerate(joint_list):
            joint.set_drive_target(homestate[i])
        return

    for i, joint in enumerate(self.left_arm_joints):
        joint.set_drive_target(self.left_homestate[i])

    for i, joint in enumerate(self.right_arm_joints):
        joint.set_drive_target(self.right_homestate[i])
```

原因：

`embodiment: [franka-panda]` 时，RoboTwin 会把同一个 robot entity 同时作为 left/right entity 引用。若单臂还顺序设置 left 和 right homestate，后设置的一侧可能覆盖前一侧，导致实际初始姿态和 active arm 不一致。

## `play_once()` 的单臂逻辑

以 `place_empty_cup` 为例：

```python
def play_once(self):
    cup_pose = self.cup.get_pose().p
    if self.single_arm:
        arm_tag = ArmTag(self.active_arm)
    else:
        arm_tag = ArmTag("right" if cup_pose[0] > 0 else "left")
```

区别：

- 双臂：根据物体 x 位置选择左臂或右臂。
- 单臂：始终用 `active_arm`。

这只影响专家轨迹/seed 检查，不改变 `load_actors()` 的物体随机性。

## `is_in_hand()` 的单臂逻辑

旧双臂逻辑通过 contact 判断左右夹爪是否抓住物体。单臂新增了简单距离判断：

```python
def is_in_hand(self, actor):
    if self._single_arm_enabled():
        arm_tag = self._active_arm()
        eef_pose = np.array(self._arm_ee_pose(arm_tag))[:3]
        actor_pose = actor.pose() if hasattr(actor, "pose") else actor.get_pose().p
        in_hand = np.linalg.norm(eef_pose - actor_pose) < 0.05
        if arm_tag == "left":
            return in_hand, False
        return False, in_hand

    if self.dual_arm:
        contacts = self.scene.get_contacts()
        ...
```

区别：

- 单臂：返回 `(left_in_hand, right_in_hand)`，但只有 active arm 对应的一侧可能为 True。
- 双臂：仍然按接触点分别判断左右夹爪。

## RLinF wrapper 里的动作适配

RLinF 的 `RoboTwinEnv` 从 RoboTwin `VectorEnv` 读取真实 env action dim：

```python
self.venv = VectorEnv(...)
self.action_dim = int(self.venv.args["action_dim"])
```

如果没有 `policy_adapter`，policy action dim 就等于 env action dim：

```python
adapter_cfg = self.cfg.get("policy_adapter", {}) or {}
self.policy_action_dim = int(adapter_cfg.get("action_dim", self.action_dim))
self.policy_state_dim = int(adapter_cfg.get("state_dim", self.action_dim))
```

因此 MLP/CNN 原生测试时：

```yaml
actor:
  model:
    action_dim: 8
    obs_dim: 8
```

就会直接输出 8D Franka 单臂动作。

老 Pi0 smoke config 使用了 adapter：

```python
if self.action_adapter is None:
    self._validate_env_action_dim(actions)
    return actions

if self.action_adapter != "aloha14_to_franka8_smoke":
    raise ValueError(...)
...
adapted_actions = np.zeros((*actions.shape[:-1], 8), dtype=actions.dtype)
```

这条路径只是为了让 14D ALOHA checkpoint 临时跑 Franka 单臂 env，不是底层单臂支持的主路径。

## 单臂和双臂区别总结

| 项目 | 单臂 Franka-Panda | 双臂 |
|---|---|---|
| 配置 | `single_arm: true` | `single_arm: false` |
| active arm | `active_arm: left/right` 决定内部 API 槽位 | 通常由任务逻辑选择左右臂 |
| action dim | `active_arm_dim + 1`，Franka 是 8 | `left_dim + 1 + right_dim + 1`，两 Franka 是 16 |
| state dim | active arm qpos + gripper | left + right full vector |
| 控制对象 | 只控制 active arm | 同时控制 left/right |
| qpos path | 只构造 active arm path | 构造 left path 和 right path |
| TOPP | active arm planner | left/right planner |
| gripper success check | left/right 查询都映射到 active arm | left/right 分别查询 |
| episode reward trace | left/right trace 都写 active arm | left/right trace 分别写左右臂 |
| 物体随机性 | 不因 active arm 改变 | 原始任务随机性 |
| Pi0 14D 兼容 | 通过 RLinF adapter 转 8D | 原始 14D 或任务对应维度 |

## 当前实现的不足

当前实现可以跑原生 8D 单臂动作，但结构还不够理想。

主要问题：

1. 单臂 executor 是单独函数 `_execute_single_arm_qpos_actions()`，双臂 executor 仍散落在 `take_action()`、`gen_sparse_reward_data()`、`gen_dense_reward_data()`、`gen_dense_reward_once()` 里。
2. `gen_dense_reward_data()` 和 `gen_dense_reward_once()` 里仍保留大量双臂旧代码，只是在前面加了单臂 shortcut。
3. episode trace helper 放在 `get_obs()` 前面，可读性不好，容易误解成 observation 逻辑被改。
4. `gen_sparse_reward_data()` 的底层控制还不是一个统一的“同时处理单双臂”的接口。

更合理的后续重构方向：

```python
def _execute_qpos_chunk_actions(self, actions, downsample=True):
    if self._single_arm_enabled():
        return self._execute_single_arm_qpos_actions(actions, downsample)
    return self._execute_dual_arm_qpos_actions(actions, downsample)
```

然后：

```python
take_action()
gen_sparse_reward_data()
gen_dense_reward_data()
gen_dense_reward_once()
```

都只调用这个统一底层 executor。

这样代码结构会更清楚：

- action 维度检查集中在一个地方。
- 单臂/双臂控制差异集中在一个地方。
- reward/update/video 等逻辑不再重复控制细节。

## 当前结论

当前分支已经实现了：

- Franka-Panda 单臂 8D action。
- 双臂原始 action 格式保留。
- RLinF 无 adapter 模式可直接用 MLP/CNN 输出 8D action。
- 任务物体随机性不受 `active_arm` 影响。
- 常见 success/reward 查询兼容单臂。

但从代码结构上看，下一步应该继续把 sparse/dense/take_action 的底层 qpos 执行统一起来，而不是长期保留当前这种“单臂单独 shortcut，双臂旧逻辑散落各处”的状态。
