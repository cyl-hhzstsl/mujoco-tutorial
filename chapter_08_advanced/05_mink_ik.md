# 第 8 章 · 05 — mink: 专业级逆运动学库

> **目标**: 透彻掌握 mink 把 IK 转成 QP 的完整链路 —— 包括 `mink.Task` 和 `mink.Limit` 两大基类的接口契约、
> 内置实现的源码级解读、自定义 Task / Limit 的方法，以及在多约束多任务场景下的调参经验。

## 内容地图

```
1. 为什么需要 mink
2. 安装与依赖
3. 数学背景：IK 如何变成 QP
4. 三大核心对象：Configuration / Task / Limit
5. mink.Task 详解（基类 + 内置 + 自定义）
6. mink.Limit 详解（基类 + 内置 + 自定义）
7. solve_ik 内部解剖
8. 渐进式实战：从单目标到多约束
9. 调参指南
10. 常见陷阱
11. 与手写 Jacobian IK 的对照
12. 数据工程视角
13. 总结
```

---

## 1. 为什么需要 mink

在 `01_kinematics.md` 中，我们手写了一个 Jacobian 伪逆 IK 求解器。它能工作，但存在明显局限：

| 手写 IK 的局限 | mink 的解决方案 |
| :------------- | :------------- |
| 只能处理位置目标 | 同时支持位置 + 姿态目标（SE(3) 误差） |
| 关节限位靠事后 clip | 关节限位作为**不等式约束**参与优化 |
| 无碰撞检测 | 内置碰撞回避约束（基于 `mj_geomDistance`） |
| 单目标 | **多目标加权组合**，自动 QP 折衷 |
| 奇异点靠阻尼硬编码 | LM damping + 全局 damping 两层正则化 |
| 速度无界 | 内置速度上限约束 |

mink 的核心思想：**所有 IK 问题都是带约束的二次规划（QP）**。

```
minimize    Σ wᵢ ‖Jᵢ Δq + αᵢ eᵢ‖²       （多任务加权 → QP 目标 H, c）
            + damping · ‖Δq‖²              （全局正则化）
subject to  G Δq ≤ h                       （关节/速度/碰撞 → 不等式 G, h）
            A Δq = b                       （硬性等式约束 → A, b）
```

`Δq` 是当前 q 的切空间扰动；除以 `dt` 得到速度 `v`。

---

## 2. 安装与依赖

```bash
pip install mink
```

mink 依赖：
- `mujoco`（MJCF 模型 + 雅可比）
- `qpsolvers`（QP 后端，默认推荐 `daqp`，可选 `osqp`、`quadprog`、`cvxopt` 等）
- `numpy`

安装 daqp：

```bash
pip install daqp
```

> mink 不强制装某一个 QP 求解器，但 `daqp` 是文档推荐的默认选项 —— 对中小规模问题快、稳定、不需要 LP 预求解。

---

## 3. 数学背景：IK 如何变成 QP

### 3.1 一阶任务动力学

mink 的所有 task 都遵循"一阶动力学"假设：

\[
J(q)\,\Delta q \;\approx\; -\alpha\,e(q)
\]

- `e(q) ∈ ℝᵏ`：任务误差（k 是任务维度，比如 `FrameTask` 是 6 维 SE(3) 误差）
- `J(q) ∈ ℝᵏˣⁿᵛ`：误差对配置切空间的雅可比
- `α ∈ [0, 1]`：task 的 `gain`，控制误差的收敛速率（α=1 时一步消除，α 越小越平滑）

直觉：**期望我们走一步 Δq，就能让误差按 α 比例减小**。

### 3.2 多任务 → QP 目标

把上式做成最小二乘，并按 `cost` 加权：

\[
\min_{\Delta q}\;\sum_i \big\|\,W_i\,J_i\,\Delta q + W_i\,\alpha_i\,e_i \,\big\|^2
\]

这就是一个 QP，写成标准形：

\[
\min_{\Delta q} \;\tfrac{1}{2}\Delta q^\top H\, \Delta q + c^\top \Delta q
\]

其中

\[
H = \sum_i (W_i J_i)^\top (W_i J_i),\qquad c = \sum_i \alpha_i\, e_i^\top W_i^\top (W_i J_i)
\]

mink 在 `Task._assemble_qp` 里就是这么算的（后面 §5.1 会贴源码）。

### 3.3 限制 → QP 不等式

每个 `Limit` 把"关节角不超界 / 速度不超限 / 几何不互穿"翻译成形如

\[
G(q)\,\Delta q \;\le\; h(q)
\]

的线性不等式（在切空间里线性化）。

mink 的 `solve_ik` 把所有 task 拼成 `(H, c)`，所有 limit 拼成 `(G, h)`，调 QP 求解器：

```
min  ½ Δqᵀ H Δq + cᵀ Δq
s.t. G Δq ≤ h
     A Δq = b   ← 可选：当传 constraints=[task] 时把那个 task 转成等式
```

---

## 4. 三大核心对象

```
┌────────────────────────────────────────────────────────────────┐
│                          mink 架构                              │
│                                                                │
│   Configuration —— 机器人当前状态 (qpos / qvel / mjData 包装)    │
│        │                                                       │
│        │ 提供 get_transform_frame_to_world / get_frame_jacobian │
│        ▼                                                       │
│   ┌──────────────┐         ┌──────────────┐                    │
│   │  Task (软)   │         │ Limit (硬)   │                    │
│   │ 出 (H, c)    │         │ 出 (G, h)    │                    │
│   │ 进 QP 目标   │         │ 进 QP 约束   │                    │
│   └──────┬───────┘         └──────┬───────┘                    │
│          └──────────┬─────────────┘                            │
│                     ▼                                          │
│              solve_ik() —— 拼 QP，调 daqp                       │
│                     │                                          │
│                     ▼                                          │
│           v (广义速度) ── integrate_inplace ──▶ qpos 更新        │
└────────────────────────────────────────────────────────────────┘
```

### 4.1 `Configuration`

`Configuration` 是 `MjModel + MjData` 的薄包装，**所有 task / limit 都通过它读取当前 qpos 和雅可比**。

```python
import mujoco
from mink import Configuration

model = mujoco.MjModel.from_xml_path("robot.xml")
configuration = Configuration(model)

configuration.data.qpos[:] = initial_qpos
configuration.update()                # 触发 mj_forward，更新 xpos / xmat

ee_pose = configuration.get_transform_frame_to_world("ee_site", "site")
J = configuration.get_frame_jacobian("ee_site", "site")   # 6 × nv
```

更新 qpos 的两种方式：

```python
# 方式 1: 求解 IK 后用 v 积分
configuration.integrate_inplace(v, dt)

# 方式 2: 直接写 qpos（不推荐，绕过了 free joint 的 SE3 retraction）
configuration.data.qpos[:] = new_qpos
configuration.update()
```

### 4.2 `Task` / `Limit` 简介

| | `mink.Task` | `mink.Limit` |
|---|---|---|
| 作用 | **软约束**：在 QP 目标里加项 | **硬约束**：在 QP 不等式里加行 |
| 接口 | `compute_error` + `compute_jacobian` | `compute_qp_inequalities` |
| 违反代价 | 有限（按 `cost` 权重） | 无限（违反 = QP 不可行） |
| 适用场景 | 末端跟踪、姿态偏好、姿态保持 | 关节限位、速度上限、碰撞回避 |

### 4.3 `solve_ik` 流程

```python
from mink import solve_ik

v = solve_ik(
    configuration,
    tasks=[task1, task2],            # 列表 of Task
    dt=0.01,
    solver="daqp",                   # qpsolvers 后端名
    damping=1e-3,                    # 全局 LM 阻尼，加在 H 对角上
    limits=[limit1, limit2, ...],    # 列表 of Limit；None → 默认只用 ConfigurationLimit
    constraints=[hard_task],         # 可选：把某个 task 升级为等式约束
)
configuration.integrate_inplace(v, dt)
```

---

## 5. `mink.Task` 详解

`mink.Task` 是 mink 里所有"软目标"的基类。

### 5.1 基类源码精读

```python
class Task(BaseTask):
    def __init__(self, cost, gain=1.0, lm_damping=0.0):
        if not 0.0 <= gain <= 1.0:
            raise InvalidGain("`gain` must be in the range [0, 1]")
        if lm_damping < 0.0:
            raise InvalidDamping("`lm_damping` must be >= 0")
        self.cost = cost                # ndarray, 长度 = error 维度 k
        self.gain = gain                # 标量 ∈ [0, 1]
        self.lm_damping = lm_damping    # 标量 ≥ 0

    @abc.abstractmethod
    def compute_error(self, configuration) -> np.ndarray: ...

    @abc.abstractmethod
    def compute_jacobian(self, configuration) -> np.ndarray: ...

    def _assemble_qp(self, error, jacobian, eye_nv) -> Objective:
        weighted_error = self.cost * (-self.gain * error)            # k 维
        weighted_jacobian = self.cost[:, None] * jacobian            # k × nv
        mu = self.lm_damping * (weighted_error @ weighted_error)
        H = weighted_jacobian.T @ weighted_jacobian                  # nv × nv
        if mu > 0.0:
            H = H + mu * eye_nv
        c = -weighted_error @ weighted_jacobian                      # nv
        return Objective(H, c)
```

> 注意 `cost` 是**向量**，每一维的误差都可以有独立权重。例如 `FrameTask` 让 `cost[:3]` 控制位置、`cost[3:]` 控制姿态。

#### 三个参数的物理意义

| 参数 | 在公式中的位置 | 直观含义 |
|---|---|---|
| `cost` | 残差权重 `Wᵢ` | 该任务的"重要程度"。越大 → QP 越愿意为它牺牲其他任务 |
| `gain` α | `J Δq ≈ -α e` | 残差**收敛比率**。1 = 一步打到位（dead-beat），<1 = 平滑收敛 |
| `lm_damping` | `μ · ‖e‖² · I` | **目标越远 → 阻尼越大**，避免远离目标时步长过大撞约束。仅在 `lm_damping > 0` 时生效 |

> **为什么 `lm_damping` 要乘 `‖e‖²`？** —— 让阻尼随误差自适应：误差大时（远离目标）多加阻尼防超调；误差小时（接近目标）几乎无阻尼，保证收敛精度。这是 LM 算法的精髓，把 Gauss-Newton（无阻尼）和梯度下降（高阻尼）结合起来。

#### `_assemble_qp` 拆解

第 4 行 `weighted_error = cost * (-gain * error)` 计算 `Wᵢ · (-α · eᵢ)`。
第 5 行 `weighted_jacobian = cost[:, None] * jacobian` 把每行雅可比按 cost 缩放。
第 8 行 `H = Jᵀ J` 是标准最小二乘的 Hessian。
第 11 行 `c = -weighted_error @ weighted_jacobian` 是线性项。

→ 最终这个 task 在 QP 里的贡献是 `½ Δqᵀ H Δq + cᵀ Δq`，等价于 `½ ‖W·J·Δq + W·α·e‖²` 这一项的展开。

### 5.2 内置 `FrameTask` 详解

mink 最常用的 task。它让机器人某个 frame（site/body/geom）追踪一个 SE(3) 目标。

```python
from mink import FrameTask
from mink.lie import SE3

task = FrameTask(
    frame_name="end_effector",
    frame_type="site",          # "site" | "body" | "geom"
    position_cost=1.0,           # 位置权重，标量或长度 3 的向量
    orientation_cost=1.0,        # 姿态权重，标量或长度 3 的向量
    gain=1.0,                    # 默认 1（dead-beat）
    lm_damping=1e-3,             # 远离目标时加阻尼
)

task.set_target(SE3.from_translation([0.5, 0.0, 0.3]))
# 或者
task.set_target_from_configuration(configuration)   # "保持当前位姿"
```

#### 关键源码片段

```python
class FrameTask(Task):
    k: int = 6     # 6 维 SE3 误差（3 平移 + 3 旋转）

    def __init__(self, frame_name, frame_type, position_cost, orientation_cost, ...):
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        ...
        self.set_position_cost(position_cost)        # 写入 cost[:3]
        self.set_orientation_cost(orientation_cost)  # 写入 cost[3:]
```

`set_position_cost` 接受标量（广播到 3 维）或长度 3 的向量：

```python
task.set_position_cost(1.0)              # x/y/z 等权重
task.set_position_cost([1.0, 1.0, 5.0])  # z 比 xy 重 5 倍
```

#### 误差是 SE(3) "right-minus"

```python
def compute_error(self, configuration):
    # error = log(T_target⁻¹ · T_current) ∈ se(3)
    target = self.transform_target_to_world
    frame  = configuration._get_transform_frame_to_world_wxyz_xyz(...)
    return target.minus(SE3(wxyz_xyz=frame))     # 6 维 twist
```

→ 不是简单的 `xyz_target - xyz_current`，而是 SE(3) 流形上的对数映射。这保证了**姿态误差的几何正确性**（避免欧拉角奇异）。

### 5.3 自定义 Task 完整范例：`GroundAvoidanceTask`

为什么要自定义？—— 当你需要"单边软约束"（比如**身体某部位不能低于某高度**），mink 没有内置实现，但可以用 `Task` 几行写出来。

```python
import numpy as np
import mink


class GroundAvoidanceTask(mink.Task):
    """单边软约束：当 body 的世界 z 低于 min_height 时，沿 z 轴产生向上推力。"""

    def __init__(self, model, body_name, min_height=0.0, weight=1.0, gain=0.2):
        # k=1 维任务（只看 z），cost 用单元素向量
        super().__init__(cost=np.array([weight]), gain=gain, lm_damping=0.0)
        self.model = model
        self.body_name = body_name
        self.min_height = min_height

    def compute_error(self, configuration) -> np.ndarray:
        tf = configuration.get_transform_frame_to_world(self.body_name, "body")
        current_z = tf.translation()[2]
        # error = current_z - min_height
        # 关键：只在 "低于阈值" 时激活；高度足够时 error=0 → 整项静默
        error = current_z - self.min_height
        if error >= 0.0:
            error = 0.0
        return np.array([error])

    def compute_jacobian(self, configuration) -> np.ndarray:
        # body 的 6×nv frame Jacobian，取第 2 行（z 平移）
        J = configuration.get_frame_jacobian(self.body_name, "body")
        return J[2:3, :]                          # 1 × nv
```

#### 工作原理（结合 §5.1 公式）

代入 `_assemble_qp`：

- `gain = 0.2`，`error = e_z`（≤ 0 才激活），`cost = [weight]`
- `weighted_error = weight · (-0.2 · e_z) = -0.2·weight·e_z`
- 当 `e_z = 0`（达标）：`weighted_error = 0` → `H = 0`，`c = 0`，**整项不出现在 QP 里**
- 当 `e_z < 0`（违规）：QP 希望 `J_z · Δq ≈ -gain · e_z > 0`，即沿 z 给 body 向上的速度

→ 实现了"只在违规时激活的单边推力"，且代价随 `weight` 加权。

> 这就是 `cyl_xgmr` 项目里防止机器人手腕、肘、腰、骨盆穿地的实际做法（`core/ik_constraints.py`）。

### 5.4 其他常用内置 Task

| Task | 用途 | error 维度 |
|---|---|---|
| `PostureTask` | 让 qpos 接近一个参考姿态（保持站姿、回零位） | nv |
| `ComTask` | 让重心追踪 3D 目标位置（人形机器人平衡） | 3 |
| `RelativeFrameTask` | 让两个 frame 的相对位姿追踪目标（双臂协调） | 6 |
| `DampingTask` | 让所有速度趋近 0（阻尼运动） | nv |

它们都是 `mink.Task` 的子类，区别只在 `compute_error` / `compute_jacobian` 的具体实现。

---

## 6. `mink.Limit` 详解

`mink.Limit` 是所有"硬约束"的基类。

### 6.1 基类源码

```python
class Constraint(NamedTuple):
    """G(q) Δq ≤ h(q)；G 和 h 都为 None 时表示该 limit 当前未激活。"""
    G: np.ndarray | None = None
    h: np.ndarray | None = None

    @property
    def inactive(self) -> bool:
        return self.G is None and self.h is None


class Limit(abc.ABC):
    @abc.abstractmethod
    def compute_qp_inequalities(self, configuration, dt) -> Constraint:
        """返回 (G, h) 描述 G·Δq ≤ h；若当前不需要约束，返回 Constraint()。"""
```

只有一个抽象方法，签名很简单。返回 `Constraint(None, None)` 等价于"当前帧不施加这个 limit"。

### 6.2 `ConfigurationLimit` —— 关节角硬限位

```python
config_limit = ConfigurationLimit(
    model=model,
    gain=0.95,                          # 越接近 1 越激进
    min_distance_from_limits=0.0,       # >0 收紧，<0 允许穿越（不推荐）
)
```

**做了什么**：把 MJCF 中每个关节的 `<joint range="lo hi">` 转成

\[
\frac{\text{gain}\cdot(q_{\min}-q)}{1} \;\le\; \Delta q \;\le\; \text{gain}\cdot(q_{\max}-q)
\]

注意：
- **free joint 自动跳过**（free joint 没有 range）
- 没有声明 `range` 的 joint 也跳过
- 用 `mj_differentiatePos` 计算 `Δq_max = qmax ⊖ q`，正确处理球关节等流形

#### 源码精华

```python
def compute_qp_inequalities(self, configuration, dt):
    # 上界
    delta_q_max = np.empty((nv,))
    mujoco.mj_differentiatePos(model, qvel=delta_q_max, dt=1.0,
                               qpos1=configuration.q, qpos2=self.upper)
    # 下界
    delta_q_min = np.empty((nv,))
    mujoco.mj_differentiatePos(model, qvel=delta_q_min, dt=1.0,
                               qpos1=self.lower, qpos2=configuration.q)
    p_min = self.gain * delta_q_min[self.indices]
    p_max = self.gain * delta_q_max[self.indices]
    G = np.vstack([self.projection_matrix, -self.projection_matrix])  # 上下两半
    h = np.hstack([p_max, p_min])
    return Constraint(G=G, h=h)
```

→ 每个有限位的 dof 贡献 2 行（上、下界）。`projection_matrix` 是 nv × nv 单位阵的子集，按 dof 索引取行。

### 6.3 `VelocityLimit` —— 速度上限

```python
from mink import VelocityLimit

velocity_limit = VelocityLimit(
    model,
    velocities={
        "shoulder_pitch": 3.14,        # rad/s
        "elbow":          5.0,
    },
)
```

约束形如 `|Δq[i]| ≤ vmax[i] · dt`。每个限速 dof 同样贡献 2 行 G。

> **不要给 free joint 加速度限**——会束缚整体平移/旋转，导致机器人"卡在原地"。`cyl_xgmr` 的 `_build_ik_limits` 显式过滤了 free joint，就是这个原因。

### 6.4 `CollisionAvoidanceLimit` —— 几何对避障

```python
from mink import CollisionAvoidanceLimit

collision_limit = CollisionAvoidanceLimit(
    model=model,
    geom_pairs=[
        # (组A, 组B)：每对组都做笛卡尔积展开成 a×b 个 geom-geom pair
        (["left_hand", "right_hand"], ["torso", "pelvis"]),
    ],
    gain=0.85,
    minimum_distance_from_collisions=0.01,    # 1cm 安全壳
    collision_detection_distance=0.05,        # 5cm 起开始关注（应 > min）
    bound_relaxation=0.0,
)
```

#### 内部三步：geom 解析 → 距离 → QP 行

```python
def compute_qp_inequalities(self, configuration, dt):
    upper_bound = np.full(N, np.inf)             # N = max_num_contacts
    coefficient_matrix = np.zeros((N, nv))
    for idx, (g1, g2) in enumerate(self.geom_id_pairs):
        dist = mj_geomDistance(model, data, g1, g2, distmax, fromto)
        if abs(dist - distmax) < 1e-12:
            continue                              # 太远，不约束
        # 接触法线方向的相对雅可比： n^T (J2 - J1)
        row = compute_contact_normal_jacobian(model, data, g1, g2, fromto, ...)
        if dist > min_dist:
            upper_bound[idx] = gain * (dist - min_dist) / dt + relaxation
        else:
            upper_bound[idx] = relaxation
        sign = -1.0 if dist >= 0 else 1.0          # ⭐ v0.0.13 关键 bugfix
        coefficient_matrix[idx] = sign * row
    return Constraint(G=coefficient_matrix, h=upper_bound)
```

**几何意义**（详细推导见 [GroundAvoidanceTask 后续解读]）：

| 状态 | sign | 约束含义 |
|---|---|---|
| `dist > min_dist`（安全） | -1 | 允许沿法线接近，但速度上限 = `gain·(dist-min)/dt` |
| `min_dist ≥ dist ≥ 0`（贴近） | -1 | 接近速度上限 = relaxation（≈ 0） |
| `dist < 0`（已穿透） | +1 | 必须以正速度**远离**，避免锁死 |

#### 构造时的三条过滤

`__init__` 把 `geom_pairs` 展开成 `geom_id_pairs` 时会过滤：

1. **同 weld 体**（一起焊死的 body）—— 距离不可改，加约束无意义
2. **父子链路**（kinematic chain 上相邻 link）—— 几乎总是几何重叠，会让 IK 死锁
3. **contype/conaffinity 不通过** —— 沿用 MJCF 的位掩码

### 6.5 自定义 Limit 完整范例：`FootGroundContactLimit`

需求：**指定的 geom 必须严格在地面之上**（不像 `GroundAvoidanceTask` 那样允许偶尔下沉）。

```python
import numpy as np
import mujoco as mj
import mink


class FootGroundContactLimit(mink.Limit):
    """硬约束：foot_geom_names 列出的 geom，其世界 z 必须 ≥ min_height。"""

    def __init__(self, model, foot_geom_names, min_height=0.01, gain=0.5):
        self.model = model
        self.gain = gain
        self.min_height = min_height
        self.geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n)
                         for n in foot_geom_names]
        self.geom_ids = [g for g in self.geom_ids if g >= 0]
        self.geom_body_ids = [model.geom_bodyid[g] for g in self.geom_ids]

    def compute_qp_inequalities(self, configuration, dt) -> mink.Limit.Constraint:
        model = configuration.model
        data  = configuration.data
        rows, bounds = [], []

        for gid, bid in zip(self.geom_ids, self.geom_body_ids):
            current_z = data.geom_xpos[gid][2]
            margin = current_z - self.min_height
            if margin > 0.05:                         # 离地面够远，跳过
                continue

            # 取该 geom 中心的点雅可比（3 × nv），只要 z 行
            jacp = np.zeros((3, model.nv))
            mj.mj_jac(model, data, jacp, None, data.geom_xpos[gid], bid)
            J_z = jacp[2:3, :]                         # 1 × nv

            # 约束：z 不能再下降
            #   J_z · Δq ≥ -gain * margin
            # ↔ -J_z · Δq ≤  gain * margin
            rows.append(-J_z[0])
            bounds.append(self.gain * margin)

        if not rows:
            return mink.Constraint()                   # 全部安全 → 不加约束
        G = np.vstack(rows)
        h = np.array(bounds)
        return mink.Constraint(G=G, h=h)
```

要点：
1. **按需返回 `Constraint()`**：当所有几何都离地面足够远，**整个 limit 这一帧失效**，节省 QP 行数。
2. **每个违规几何贡献 1 行 G**：把"该 geom z 不能再降"线性化为 `-J_z · Δq ≤ gain·margin`。
3. **gain ∈ (0, 1]**：和 `ConfigurationLimit` 同义——margin 一帧最多被吃掉的比例。

---

## 7. `solve_ik` 内部解剖

```python
def solve_ik(configuration, tasks, dt, solver, damping=1e-12,
             limits=None, constraints=None, **kwargs):
    configuration.check_limits(safety_break=safety_break)
    problem = build_ik(configuration, tasks, dt, damping, limits, constraints)
    result  = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    if not result.found:
        raise NoSolutionFound(solver)
    delta_q = result.x
    return delta_q / dt
```

`build_ik` 拼三部分：

```python
def build_ik(configuration, tasks, dt, damping, limits, constraints):
    H, c = _compute_qp_objective(configuration, tasks, damping)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    A, b = _compute_qp_equalities(configuration, constraints)
    return qpsolvers.Problem(H, c, G, h, A, b)
```

### 7.1 目标函数 `(H, c)`

```python
def _compute_qp_objective(configuration, tasks, damping):
    H = np.eye(nv) * damping              # 全局 LM 阻尼
    c = np.zeros(nv)
    for task in tasks:
        H_task, c_task = task.compute_qp_objective(configuration)
        H += H_task
        c += c_task
    return H, c
```

→ **所有 task 的 H 相加，c 相加**。`damping` 加在对角上是为了保证 `H ≻ 0`（QP 可解）。

### 7.2 不等式约束 `(G, h)`

```python
def _compute_qp_inequalities(configuration, limits, dt):
    if limits is None:
        limits = [ConfigurationLimit(configuration.model)]   # 默认兜底
    G_list, h_list = [], []
    for limit in limits:
        ineq = limit.compute_qp_inequalities(configuration, dt)
        if not ineq.inactive:
            G_list.append(ineq.G)
            h_list.append(ineq.h)
    if not G_list:
        return None, None
    return np.vstack(G_list), np.hstack(h_list)
```

→ 各 limit 的 G **垂直堆叠**，h **拼接**。`inactive` 的 limit 自动跳过。

> **特殊值**：传 `limits=[]`（空列表）显式禁用所有约束；传 `limits=None`（默认）只启用 `ConfigurationLimit`。

### 7.3 等式约束 `(A, b)` —— `constraints` 参数的玄机

```python
def _compute_qp_equalities(configuration, constraints):
    if not constraints:
        return None, None
    A_list, b_list = [], []
    for task in constraints:
        jacobian = task.compute_jacobian(configuration)
        feedback = -task.gain * task.compute_error(configuration)
        A_list.append(jacobian)
        b_list.append(feedback)
    return np.vstack(A_list), np.hstack(b_list)
```

`constraints` 参数接收的是 **`Task` 实例列表**，但用法不同：把它们的 "`J·Δq = -gain·e`" 当作**严格等式**强制满足。

→ 适用于"绝对必须命中"的目标，比如双臂遥操作中"主手必须严格匹配 pose"，副手任务再做最小二乘折衷。

\[
\begin{aligned}
\min_{\Delta q}\ & \tfrac{1}{2}\Delta q^\top H \Delta q + c^\top \Delta q \\
\text{s.t.}\ & G\,\Delta q \le h \\
& A\,\Delta q = b
\end{aligned}
\]

---

## 8. 渐进式实战

### 8.1 最小例子：位置 IK

用 3 连杆臂，对比手写 IK 和 mink IK：

```python
import mujoco
import numpy as np
from mink import Configuration, FrameTask, solve_ik
from mink.lie import SE3

model = mujoco.MjModel.from_xml_string(ARM_XML)
configuration = Configuration(model)

task = FrameTask(
    frame_name="end_effector",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=0.0,             # 平面臂不关心姿态
    lm_damping=1e-3,
)
task.set_target(SE3.from_translation([0.5, 0.0, 0.0]))

dt = 0.01
for i in range(300):
    v = solve_ik(configuration, [task], dt, solver="daqp", damping=1e-3)
    configuration.integrate_inplace(v, dt)
    error = np.linalg.norm(task.compute_error(configuration))
    if error < 1e-4:
        break
```

对比手写 IK 的 50+ 行 `JacobianIKSolver`，mink 只需要 ~10 行。

### 8.2 加入姿态目标

```python
task = FrameTask(
    frame_name="end_effector", frame_type="site",
    position_cost=[1.0, 1.0, 5.0],    # z 比 xy 重 5 倍
    orientation_cost=0.5,              # 姿态权重比位置低
)
target = SE3.from_rotation_and_translation(
    rotation=SO3.from_z_radians(np.pi / 4),
    translation=[0.5, 0.0, 0.3],
)
task.set_target(target)
```

### 8.3 加入关节限位

```python
from mink import ConfigurationLimit

config_limit = ConfigurationLimit(model, gain=0.95)

v = solve_ik(configuration, [task], dt, solver="daqp",
             damping=1e-3, limits=[config_limit])
```

### 8.4 加入速度限制

```python
from mink import VelocityLimit

velocity_limit = VelocityLimit(model, velocities={
    "shoulder": 2.0, "elbow": 3.0, "wrist": 5.0,
})

limits = [config_limit, velocity_limit]
```

### 8.5 加入碰撞回避

```python
from mink import CollisionAvoidanceLimit

collision_limit = CollisionAvoidanceLimit(
    model,
    geom_pairs=[(["link2", "link3"], ["obstacle"])],
    minimum_distance_from_collisions=0.01,
    collision_detection_distance=0.05,         # ← 注意要 > min_distance
)

limits = [config_limit, velocity_limit, collision_limit]
```

> **避坑**：`min_distance == detection_distance` 时没有"线性减速带"，机器人会**突然刹车**。生产中通常让 `detection ≈ min + 2~5cm`。

### 8.6 加入自定义 Task

把 §5.3 的 `GroundAvoidanceTask` 用上：

```python
ground_tasks = [
    GroundAvoidanceTask(model, "left_hand",  min_height=0.05, weight=20),
    GroundAvoidanceTask(model, "right_hand", min_height=0.05, weight=20),
    GroundAvoidanceTask(model, "pelvis",     min_height=0.10, weight=80),
]

tasks = [main_track_task, *ground_tasks]
v = solve_ik(configuration, tasks, dt, solver="daqp",
             damping=1e-3, limits=limits)
```

### 8.7 多任务加权

真实场景：双手都要追踪，同时保持视线方向。

```python
left_hand = FrameTask("left_hand",  "site", position_cost=10.0, orientation_cost=1.0)
right_hand = FrameTask("right_hand", "site", position_cost=10.0, orientation_cost=1.0)
gaze = FrameTask("head", "body", position_cost=0.0, orientation_cost=0.5)

tasks = [left_hand, right_hand, gaze]
```

QP 求解器会按权重做最优折衷——当左手够不到时，宁可低头也要保持手位置。

---

## 9. 调参指南

### 9.1 Task 参数

| 参数 | 典型值 | 调高的代价 | 调低的代价 |
|---|---|---|---|
| `cost`（位置） | 1 ~ 100 | 牺牲其他任务 | 跟踪不准 |
| `cost`（姿态） | 0.1 ~ 10 | 在奇异点附近 IK 难收敛 | 姿态飘 |
| `gain` | 1.0（默认） | 高频抖动 | 跟踪滞后 |
| `lm_damping` | 1e-3 ~ 1e-1 | 收敛慢 | 远离目标时步长爆炸 |

> 经验：先把所有 task 的 `lm_damping` 设 0，跑通后只在出现"跨大步"问题的 task 上加 `1e-3`。

### 9.2 Limit 参数

| Limit | 关键参数 | 经验值 |
|---|---|---|
| `ConfigurationLimit` | `gain` | 0.5 ~ 0.95 |
| `VelocityLimit` | velocities dict | 物理 vmax × 0.6~0.8 |
| `CollisionAvoidanceLimit` | `gain`, `min`, `detection` | `0.5 / 0.01 / 0.05` |

### 9.3 全局 `damping`

```python
v = solve_ik(..., damping=1e-3)
```

- `1e-12`（默认）：几乎不加阻尼，对良好定义的 IK 问题最快收敛
- `1e-6 ~ 1e-3`：中等正则化，应对大部分实际问题
- `1e-2 ~ 1e-1`：QP 不可行或抖动严重时的兜底

### 9.4 兜底策略（生产代码模式）

```python
try:
    v = solve_ik(configuration, tasks, dt, "daqp",
                 damping=1e-3, limits=limits)
except NoSolutionFound:
    # QP 不可行，提高阻尼降级求解
    v = solve_ik(configuration, tasks, dt, "daqp",
                 damping=5e-2, limits=limits)
```

---

## 10. 常见陷阱

### 10.1 把 `cost` 当 scalar 传

```python
# 错：会触发广播，但语义不清晰
super().__init__(cost=1.0, ...)

# 对：cost 必须是 ndarray，长度等于 error 维度 k
super().__init__(cost=np.array([1.0]), ...)
```

### 10.2 自定义 Task 忘记返回 ndarray

```python
def compute_error(self, configuration):
    return -0.5            # ❌ 标量
    return np.array([-0.5])  # ✓ 1 维 ndarray
```

### 10.3 雅可比维度不一致

`compute_error` 返回 k 维，`compute_jacobian` 必须返回 k×nv。否则 `_assemble_qp` 在 `cost[:, None] * jacobian` 这步广播失败。

### 10.4 自定义 Limit 不需要时也返回非空 G/h

返回 `Constraint()`（默认全 None）才算"非激活"。返回全 0 的 G 和 inf 的 h 虽然语义上等效，但会让 QP 多吃几行无意义约束。

### 10.5 给 free joint 加 `VelocityLimit`

会让根 6DoF 卡死，机器人无法移动。永远过滤掉 free joint：

```python
free_joints = {
    mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
    for j in range(model.njnt)
    if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
}
velocities = {k: v for k, v in velocities.items() if k not in free_joints}
```

### 10.6 `CollisionAvoidanceLimit` 静默失效

写了 `geom_pairs` 但发现完全没作用？检查：
- 几何名是否拼对（`mj_name2id` 返回 -1 时静默丢弃）
- 是否被三条过滤规则全部干掉（同 weld、父子、contype 不匹配）
- `detection_distance` 是否够大覆盖你关心的场景

### 10.7 `gain ∈ (0, 1]` 严格成立

- `Task.gain` 允许 0（错误项消失，等同关闭该 task），但 `ConfigurationLimit.gain` **必须 > 0**，传 0 会抛 `LimitDefinitionError`。

---

## 11. 与手写 Jacobian IK 的对照

| 维度 | 手写 (01_kinematics.py) | mink |
| :--- | :-------------- | :--- |
| 代码量 | 100+ 行 | ~10 行（基础场景） |
| 位置 IK | ✓ | ✓ |
| 姿态 IK | 需要额外实现 | ✓ 内置（SE(3) log） |
| 关节限位 | 事后 clip（可能违反） | 优化内不等式约束 |
| 碰撞回避 | 无 | ✓ 内置 |
| 多任务 | 需要手动加权堆叠 | ✓ 自动 QP 折衷 |
| 速度限制 | 无 | ✓ 内置 |
| 等式约束 | 无 | ✓ 通过 `constraints` 参数 |
| 数值稳定性 | 手动调阻尼 | LM + 全局 damping |
| 自定义 task/limit | 重写整个求解器 | 继承 `Task` / `Limit`，写两个方法 |
| 适用场景 | 学习原理 | 生产使用 |

**学习路径**：先用 `01_kinematics.py` 理解 Jacobian 伪逆的来龙去脉，再用 mink 时就能"猜"出 QP 内部在干什么。

---

## 12. 数据工程视角

| 场景 | mink 的价值 |
| :--- | :---------- |
| 轨迹生成 | 快速生成满足关节/碰撞约束的轨迹数据 |
| 数据校验 | 用 IK 反演末端轨迹是否在工作空间内可达 |
| 遥操作数据处理 | 主端人手位姿 → 从端机器人关节角 |
| 数据增强 | 同目标不同初始 qpos → 多条轨迹 |
| 实时标注 | 笛卡尔空间标签 → 关节空间标签 |
| 运动重定向 | 人体动捕 + 多个 FrameTask + GroundAvoidanceTask → 机器人 qpos（参见 `cyl_xgmr` 项目） |

---

## 13. 总结

```
┌──────────────────────────────────────────────────────────────────┐
│                    mink — 核心要点                                  │
│                                                                  │
│ ① 数学：IK = QP                                                   │
│   min ½ Δqᵀ H Δq + cᵀ Δq                                         │
│   s.t. G Δq ≤ h ,  A Δq = b                                      │
│                                                                  │
│ ② Configuration: MjModel + MjData 包装，提供 fk / jacobian        │
│                                                                  │
│ ③ Task (软约束)                                                   │
│   - 接口: compute_error + compute_jacobian                       │
│   - 参数: cost (权重), gain (收敛率), lm_damping (自适应正则)      │
│   - 内置: FrameTask, PostureTask, ComTask, RelativeFrameTask     │
│   - 自定义: 继承 mink.Task, 实现两个方法                            │
│                                                                  │
│ ④ Limit (硬约束)                                                  │
│   - 接口: compute_qp_inequalities → Constraint(G, h)             │
│   - 内置: ConfigurationLimit, VelocityLimit,                      │
│           CollisionAvoidanceLimit                                │
│   - 自定义: 继承 mink.Limit, 实现一个方法                           │
│                                                                  │
│ ⑤ solve_ik                                                       │
│   - 把 tasks 堆成 (H, c)                                          │
│   - 把 limits 堆成 (G, h)                                         │
│   - 把 constraints 堆成 (A, b)  ← 等式约束模式                      │
│   - 调 qpsolvers，返回 v = Δq / dt                                │
│                                                                  │
│ ⑥ 调参口诀                                                        │
│   先 cost 后 gain；远离目标加 lm_damping；                          │
│   碰撞约束 detection > min；free joint 永远不要加速度限             │
└──────────────────────────────────────────────────────────────────┘
```

**进一步阅读**：
- mink 官方文档：https://kevinzakka.github.io/mink/
- 论文：Differential Inverse Kinematics（Khatib 1987 → Kanoun 2011 hierarchy）
- 实战项目：[`cyl_xgmr`](../../../../RX/python/cyl_xgmr) 用 mink 做人形机器人运动重定向
