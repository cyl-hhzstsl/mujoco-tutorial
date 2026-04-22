# 逆运动学 (IK) 方案全景对比

> 给定目标位姿，求关节角度 —— 这件事有很多种做法，各有取舍。

---

## 1. 解析法 (Analytical / Closed-Form)

直接对特定运动学结构推导关节角度的闭式表达式。

```
θ₂ = atan2(±√(1 - c²), c)   其中 c = (x² + y² - L₁² - L₂²) / (2L₁L₂)
θ₁ = atan2(y, x) - atan2(L₂sinθ₂, L₁ + L₂cosθ₂)
```

| 优点 | 缺点 |
|:-----|:-----|
| 极快（微秒级） | 仅适用于特定结构（≤6 DOF，满足 Pieper 条件） |
| 精确解，无迭代误差 | 每种机器人需要单独推导 |
| 可枚举所有解 | 无法处理冗余自由度 |
| 确定性，无初始值依赖 | 约束（碰撞、限位）需事后处理 |

**代表库**: `ikfast` (OpenRAVE 自动生成解析解代码)
**典型适用**: UR5/UR10、PUMA 560、SCARA 等经典工业臂

---

## 2. Jacobian 伪逆 / 阻尼最小二乘 (DLS)

迭代地用 Jacobian 矩阵将任务空间误差映射到关节空间增量。

```
Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ Δx    (阻尼最小二乘)
q ← q + α · Δq
```

| 优点 | 缺点 |
|:-----|:-----|
| 通用，适用任意运动链 | 迭代方法，可能收敛慢 |
| 原理简单，易于理解和实现 | 奇异点附近需要阻尼调参 |
| 支持冗余自由度 | 约束（限位/碰撞）靠事后 clip |
| MuJoCo 原生提供 Jacobian API | 多目标需要手动加权拼接 |

**本教程实现**: `chapter_08_advanced/01_kinematics.py` 中的 `JacobianIKSolver`

**关键参数**:

| 参数 | 默认值 | 含义 |
|:-----|------:|:-----|
| max_iter | 200 | 最大迭代次数 |
| tol | 1e-4 | 收敛容差 (m) |
| step_size | 0.5 | 学习率 α |
| damping | 1e-4 | 阻尼系数 λ（防止奇异） |

---

## 3. QP 优化 IK (mink / Pink)

将 IK 建模为**二次规划 (Quadratic Programming)**，约束在优化中同时满足。

```
minimize    ∑ wᵢ ‖Jᵢ Δq − Δxᵢ‖²       (多任务加权)
subject to  q_min ≤ q + Δq ≤ q_max     (关节限位)
            |Δq| ≤ dt · v_max           (速度限制)
            碰撞距离 ≥ 安全边距           (碰撞回避)
```

| 优点 | 缺点 |
|:-----|:-----|
| 约束在优化中同时满足（非事后修正） | 需要 QP 求解器依赖（daqp/osqp） |
| 多任务 + 多约束自动权衡 | 比纯 Jacobian 法计算量稍大 |
| 代码量极少（~10 行） | 仍然是局部优化 |
| 碰撞回避、速度限制内置 | 学习门槛略高于 Jacobian 法 |
| 实时友好（ms 级） | |

**代表库**: `mink` (MuJoCo 生态), `pink` (Pinocchio 生态)
**本教程实现**: `chapter_08_advanced/05_mink_ik.py`

---

## 4. MuJoCo mocap body（仿真驱动 IK）

利用 MuJoCo 的 mocap body + weld equality constraint，让物理引擎的约束求解器隐式地做 IK。

```python
data.mocap_pos[0] = target_pos
data.mocap_quat[0] = target_quat
# weld constraint 将末端焊到 mocap body
# MuJoCo 约束求解器自动驱动关节到达目标
for _ in range(steps):
    mujoco.mj_step(model, data)
```

| 优点 | 缺点 |
|:-----|:-----|
| 零额外依赖，MuJoCo 原生 | 不是真正的 IK 求解器，是物理仿真 |
| 物理一致（力矩/碰撞自然处理） | 收敛慢（需要仿真多步） |
| 天然处理接触和碰撞 | 有残余误差，精度取决于 PD 增益 |
| 适合仿真中的交互控制 | 不适合离线轨迹生成 |

---

## 5. 非线性优化 (NLopt / SciPy / CasADi / Drake)

将 IK 完整建模为非线性优化问题，灵活度最高。

```python
from scipy.optimize import minimize

def cost(q):
    x_current = FK(q)
    return ‖x_current - x_target‖² + λ·‖q - q_ref‖²

result = minimize(cost, q0, method='SLSQP',
                  bounds=joint_limits,
                  constraints=collision_constraints)
```

| 优点 | 缺点 |
|:-----|:-----|
| 最灵活，任意目标函数和约束 | 最慢（ms ～ s 级） |
| 可加正则化 / 偏好姿态 | 高度依赖初始值 |
| 支持全身优化（whole-body） | 非凸问题，可能陷入局部最优 |
| 精度可以很高 | 需要调参经验 |

**代表库**: SciPy, NLopt, CasADi, Drake

---

## 6. 采样型 IK (TRAC-IK / BioIK / RelaxedIK)

组合策略：先用 Jacobian 法快速尝试，失败则随机采样重启，大幅提高成功率。

| 优点 | 缺点 |
|:-----|:-----|
| 成功率极高（> 99%） | 非确定性（每次结果可能不同） |
| 自动处理多解和奇异 | 延迟不稳定（ms ～ 100ms） |
| 即插即用，通用性好 | 主要在 ROS / Pinocchio 生态 |

**代表库**: TRAC-IK (ROS), BioIK (MoveIt)

---

## 7. 学习型 IK (Neural IK)

用神经网络学习 IK 映射：`target_pose → qpos`。

| 优点 | 缺点 |
|:-----|:-----|
| 推理极快（GPU 批量推理） | 需要大量训练数据 |
| 可以学习全局映射 | 精度不如优化方法 |
| 适合批量数据生成 | 泛化到新机器人需重新训练 |

**代表**: IKFlow, Neural IK 相关论文

---

## 综合对比

| 方案 | 速度 | 精度 | 通用性 | 约束处理 | 代码复杂度 | 适用场景 |
|:-----|:----:|:----:|:------:|:--------:|:----------:|:---------|
| 解析法 | ★★★★★ | ★★★★★ | ★☆☆☆☆ | ☆☆☆☆☆ | ★★★☆☆ | 特定工业臂，极致实时 |
| Jacobian 伪逆 | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★☆☆☆ | ★★★★☆ | 学习原理，简单场景 |
| QP 优化 (mink) | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★★★ | ★★★★★ | 生产级机器人控制 |
| MuJoCo mocap | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | 仿真内交互 |
| 非线性优化 | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ | ★★☆☆☆ | 离线规划，全身优化 |
| 采样型 (TRAC-IK) | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ROS 生态，高成功率 |
| 神经网络 | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | 批量推理，实时性极高 |

> ★ 越多越好。"代码复杂度"指使用时的简洁程度（★ 多 = 用起来简单）。

---

## 选型决策树

```
你的需求是什么？
│
├── 学习 IK 原理？
│   └── Jacobian 伪逆 → chapter_08_advanced/01_kinematics.py
│
├── 特定机器人 + 极致速度？
│   └── 解析法 / ikfast
│
├── 通用机器人 + 需要约束（限位/碰撞）？
│   └── mink / pink → chapter_08_advanced/05_mink_ik.py
│
├── MuJoCo 仿真中的交互控制？
│   └── mocap body + weld constraint
│
├── 离线轨迹优化 / 全身运动规划？
│   └── CasADi / Drake / NLopt
│
├── ROS 生态 + 高成功率？
│   └── TRAC-IK / BioIK
│
└── 批量数据生成 / GPU 推理？
    └── Neural IK
```
