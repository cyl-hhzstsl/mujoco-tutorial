# 第 8 章 · 05 — mink: 专业级逆运动学库

> **目标**: 掌握 mink 库的核心 API，用几行代码实现比手写 Jacobian IK 更强大、更安全的逆运动学求解，
> 理解约束优化在机器人控制中的实际应用。

## 为什么需要 mink

在 `01_kinematics.py` 中，我们手写了一个 Jacobian 伪逆 IK 求解器。它能工作，但存在明显局限：

| 手写 IK 的局限 | mink 的解决方案 |
| :------------- | :------------- |
| 只能处理位置目标 | 同时支持位置 + 姿态目标 |
| 关节限位靠事后 clip | 关节限位作为约束参与优化 |
| 无碰撞检测 | 内置碰撞回避约束 |
| 单目标 | 多目标加权组合 |
| 奇异点靠阻尼硬编码 | QP 求解器自动处理冲突 |

**mink** (MuJoCo Inverse Kinematics) 将 IK 表述为**二次规划 (QP)** 问题：

```
minimize    ∑ wᵢ ‖Jᵢ Δq − Δxᵢ‖²       (多任务加权)
subject to  q_min ≤ q + Δq ≤ q_max     (关节限位)
            |Δq| ≤ dt · v_max           (速度限制)
            碰撞距离 ≥ 安全边距           (碰撞回避)
```

---

## 1. 安装

```bash
pip install mink
```

mink 依赖 MuJoCo，如果你已经按本教程安装过环境，只需 `pip install mink` 即可。

---

## 2. 核心概念：三个积木

mink 的 API 围绕三个核心对象：

```
┌─────────────────────────────────────────────────────┐
│                    mink 架构                         │
│                                                     │
│  Configuration ── 机器人当前状态 (qpos 的包装)        │
│       │                                             │
│       ▼                                             │
│  FrameTask ── 任务空间目标                            │
│    "把末端执行器移到这里"                               │
│    "让基座朝向那个方向"                                │
│       │                                             │
│       ▼                                             │
│  solve_ik() ── QP 求解                               │
│    输入: Configuration + [Tasks] + [Limits]          │
│    输出: 关节速度 Δq                                  │
│       │                                             │
│       ▼                                             │
│  integrate_inplace() ── 更新 qpos                    │
│    q_new = q + dt · Δq                              │
└─────────────────────────────────────────────────────┘
```

### 2.1 Configuration

`Configuration` 是对 MuJoCo 模型状态的包装，管理 qpos 并提供运动学计算：

```python
import mujoco
from mink import Configuration

model = mujoco.MjModel.from_xml_path("robot.xml")
configuration = Configuration(model)

# 内部持有一个 MjData，可以读写 qpos
configuration.data.qpos[:] = initial_qpos
configuration.update()  # 触发 mj_forward
```

### 2.2 FrameTask

`FrameTask` 定义一个任务空间目标 —— "让某个 frame 到达某个位姿"：

```python
from mink import FrameTask

task = FrameTask(
    frame_name="end_effector",   # MuJoCo 中的 site/body 名称
    frame_type="site",           # "site" 或 "body"
    position_cost=1.0,           # 位置误差的权重
    orientation_cost=1.0,        # 姿态误差的权重（0 = 不关心姿态）
)
```

设定目标：

```python
# 方式 1: 只关心位置
task.set_target(SE3.from_translation(np.array([0.5, 0.0, 0.3])))

# 方式 2: 从当前 configuration 复制（"保持当前位姿"）
task.set_target_from_configuration(configuration)
```

### 2.3 solve_ik

核心求解函数，一行搞定 IK：

```python
from mink import solve_ik

vel = solve_ik(
    configuration,          # 当前状态
    tasks=[task],           # 任务列表
    dt=0.01,                # 时间步长
    solver="daqp",          # QP 求解器
    damping=1e-3,           # 阻尼（数值稳定性）
)
```

更新状态：

```python
configuration.integrate_inplace(vel, dt=0.01)
```

---

## 3. 最小示例：位置 IK

用我们熟悉的 3 连杆臂，对比手写 IK 和 mink IK：

```python
import mujoco
import numpy as np
from mink import Configuration, FrameTask, solve_ik
from mink.lie import SE3

# 加载 3 连杆臂模型
model = mujoco.MjModel.from_xml_string(ARM_XML)
configuration = Configuration(model)

# 定义任务: 末端到达目标位置
task = FrameTask(
    frame_name="end_effector",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=0.0,  # 平面臂不关心姿态
)
task.set_target(SE3.from_translation(np.array([0.5, 0.0, 0.0])))

# 迭代求解
dt = 0.01
for i in range(300):
    vel = solve_ik(configuration, [task], dt, solver="daqp", damping=1e-3)
    configuration.integrate_inplace(vel, dt)

    ee_pos = configuration.data.site_xpos[site_id]
    error = np.linalg.norm(target - ee_pos)
    if error < 1e-4:
        break
```

对比手写 IK 的 `JacobianIKSolver` 需要 50+ 行代码，mink 只需 ~10 行。

---

## 4. 约束：Limits

mink 的真正威力在于**约束**。约束不是事后修正，而是在优化中同时满足。

### 4.1 ConfigurationLimit（关节限位）

```python
from mink import ConfigurationLimit

config_limit = ConfigurationLimit(model=model)
```

直接从 MJCF 模型读取 `jnt_range`，确保求解结果不会超出关节限位。

### 4.2 VelocityLimit（速度限制）

```python
from mink import VelocityLimit

velocity_limit = VelocityLimit(
    model,
    velocities={"shoulder": np.pi, "elbow": np.pi, "wrist": np.pi},
)
```

限制每个关节的最大速度（rad/s），防止控制信号过于激进。

### 4.3 CollisionAvoidanceLimit（碰撞回避）

```python
from mink import CollisionAvoidanceLimit

collision_limit = CollisionAvoidanceLimit(
    model=model,
    geom_pairs=[(["link1", "link2"], ["obstacle"])],
    minimum_distance_from_collisions=0.01,  # 1cm 安全边距
)
```

指定哪些几何体对需要避免碰撞。

### 使用约束

```python
limits = [config_limit, velocity_limit, collision_limit]

vel = solve_ik(
    configuration,
    tasks=[task],
    dt=dt,
    solver="daqp",
    damping=1e-3,
    limits=limits,    # 传入约束列表
)
```

---

## 5. 多任务 IK

真实场景中，机器人通常需要同时满足多个目标。mink 通过权重控制优先级：

```python
# 任务 1: 左手到达目标（高优先级）
left_task = FrameTask(
    frame_name="left_hand",
    frame_type="site",
    position_cost=10.0,      # 高权重
    orientation_cost=1.0,
)

# 任务 2: 保持视线方向（低优先级）
gaze_task = FrameTask(
    frame_name="head",
    frame_type="body",
    position_cost=0.0,       # 不关心头部位置
    orientation_cost=0.1,    # 低权重
)

vel = solve_ik(configuration, [left_task, gaze_task], dt)
```

当多任务冲突时（比如左手够不到而不低头），QP 求解器会根据权重做最优折衷。

---

## 6. 实时控制循环

mink 设计为**实时友好**，典型使用模式：

```python
import time

dt = 0.002  # 与 MuJoCo timestep 一致

while True:
    # 1. 更新目标（比如从遥操作设备读取）
    task.set_target(SE3.from_translation(get_target()))

    # 2. 求解 IK
    vel = solve_ik(configuration, [task], dt, solver="daqp", limits=limits)

    # 3. 积分
    configuration.integrate_inplace(vel, dt)

    # 4. 将结果写入仿真器
    sim_data.qpos[:] = configuration.data.qpos
    mujoco.mj_step(model, sim_data)

    time.sleep(dt)
```

---

## 7. 与 01_kinematics.py 的对比

| 维度 | 手写 Jacobian IK | mink |
| :--- | :-------------- | :--- |
| 代码量 | 100+ 行 | ~10 行 |
| 位置 IK | ✓ | ✓ |
| 姿态 IK | 需要额外实现 | ✓ 内置 |
| 关节限位 | 事后 clip | 优化内约束 |
| 碰撞回避 | 无 | ✓ 内置 |
| 多任务 | 需要手动加权 | ✓ 自动 QP |
| 速度限制 | 无 | ✓ 内置 |
| 数值稳定性 | 手动调阻尼 | QP 求解器保证 |
| 适用场景 | 学习原理 | 生产使用 |

**建议**: 先用 `01_kinematics.py` 理解 IK 原理，再用 mink 做实际项目。

---

## 8. 数据工程视角

| 场景 | mink 的价值 |
| :--- | :---------- |
| 轨迹生成 | 快速生成满足约束的机器人轨迹数据 |
| 数据校验 | 用 IK 验证末端轨迹是否在关节限位内可达 |
| 遥操作数据处理 | 将人手位姿映射到机器人关节角 |
| 数据增强 | 给定不同初始 qpos，生成多条到达同一目标的轨迹 |
| 实时标注 | 将笛卡尔空间标签转换为关节空间标签 |

---

## 9. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              mink — 核心要点                                   │
│                                                              │
│  Configuration: 管理机器人状态 (qpos 的包装)                    │
│                                                              │
│  FrameTask: 定义 "让 frame 到达目标位姿" 的任务                 │
│    position_cost / orientation_cost 控制权重                   │
│                                                              │
│  solve_ik(): 将 IK 表述为 QP 问题一次求解                      │
│    支持多任务 + 多约束                                         │
│                                                              │
│  Limits: 关节限位 / 速度限制 / 碰撞回避                        │
│    约束参与优化，而非事后修正                                    │
│                                                              │
│  适用: 遥操作、轨迹规划、数据生成、实时控制                      │
└──────────────────────────────────────────────────────────────┘
```
