# 第 4 章 · 02 — 对比不同类型机器人

> **目标**: 深入对比固定基座机械臂与移动机器人的 qpos 结构差异，理解 free joint 对数据维度的影响。

## 核心知识点

1. 固定基座 vs 浮动基座的 qpos 布局
2. 不同机器人类型的 DOF 对比
3. 关节范围 (joint range) 的含义
4. nq ≠ nv 的根本原因（复习与实战）

---

## 1. 四种机器人模型

脚本使用了简化但结构忠实的 MJCF 模型：

| 模型 | 类型 | 基座 | nq | nv | nu | njnt |
| :--- | :--- | :--- | --: | --: | --: | ---: |
| UR5e | 6 轴机械臂 | 固定 | 6 | 6 | 6 | 6 |
| Franka | 7 轴 + 手指 | 固定 | 9 | 9 | 9 | 9 |
| Go2 | 四足机器人 | 浮动 | 19 | 18 | 12 | 13 |
| H1 | 人形机器人 | 浮动 | 26 | 25 | 19 | 20 |

---

## 2. qpos 结构对比

### 固定基座机械臂

```
UR5e:    qpos = [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]
                 ← 全部 6 个 hinge 关节角 →

Franka:  qpos = [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆, θ₇, d₁, d₂]
                 ← 7 个 hinge 关节角 →     ← 2 slide →
```

- 没有 free joint，qpos 直接从关节值开始
- nq == nv（hinge 和 slide 的 qpos/qvel 维度相同，都是 1）
- 所有 DOF 都有对应的执行器

### 浮动基座移动机器人

```
Go2:     qpos = [x, y, z, qw, qx, qy, qz, θ₁, ..., θ₁₂]
                 ← 7 (free joint) →       ← 12 hinge →

H1:      qpos = [x, y, z, qw, qx, qy, qz, θ₁, ..., θ₁₉]
                 ← 7 (free joint) →       ← 19 hinge →
```

- 有 free joint，qpos 前 7 位是基座位姿
- nq = nv + 1（四元数 4 维 → 角速度 3 维，多出 1 维）
- 基座 6 DOF 没有直接执行器（靠与地面的接触力间接驱动）

---

## 3. 自由度 (DOF) 分析

```
                   总DOF(nv)  base DOF  关节DOF  执行器(nu)
  ──────────────   ─────────  ────────  ───────  ─────────
  UR5e                    6         0        6         6
  Franka                  9         0        9         9
  Go2                    18         6       12        12
  H1                     25         6       19        19
```

### 关键公式

\[
\text{总 DOF} = \text{base DOF} + \text{关节 DOF}
\]

- 固定基座：base DOF = 0
- 浮动基座：base DOF = 6（3 平移 + 3 旋转）

### 欠驱动 (Underactuation)

浮动基座机器人的 base 6 DOF 无直接执行器：

\[
nu < nv \quad \text{（欠驱动）}
\]

Go2 的 12 个电机只能控制 12 个腿关节，但机体有 18 个 DOF。基座的运动完全靠腿部与地面的接触力产生。

---

## 4. 关节类型分布

```
  机器人        free   hinge   slide   总计
  ──────────   ─────  ──────  ──────  ─────
  UR5e             0       6       0      6
  Franka           0       7       2      9
  Go2              1      12       0     13
  H1               1      19       0     20
```

### 观察

- **机械臂**主要由 hinge 关节构成，Franka 额外有 slide（手指）
- **移动机器人**始终有 1 个 free joint + 大量 hinge
- 本教程的所有模型中没有 ball joint（实际机器人也很少使用）

---

## 5. 关节范围 (Joint Range)

每个受限关节都有一个范围 `[lo, hi]`，存储在 `model.jnt_range` 中（弧度）。

```python
if model.jnt_limited[j]:
    lo, hi = model.jnt_range[j]        # 弧度
    lo_deg, hi_deg = np.degrees(lo), np.degrees(hi)
```

### 典型范围示例

| 机器人 | 关节 | 范围（度） | 说明 |
| :----- | :--- | ---------: | :--- |
| UR5e | shoulder_pan | -360° ~ 360° | 全周旋转 |
| UR5e | elbow | -180° ~ 180° | 半周 |
| Franka | j4 | -175.9° ~ -4.0° | 单向受限 |
| Franka | finger_l | 0 ~ 0.04 (米) | slide 关节，线位移 |
| Go2 | FL_calf_j | -158.7° ~ -35.0° | 小腿只能向后弯 |
| H1 | l_knee_j | -14.9° ~ 117.5° | 膝盖只能向前弯 |

> **注意**：slide 关节的 range 单位是米（不是弧度），表示线位移范围。

---

## 6. nq ≠ nv 的根本原因（再次强调）

| 场景 | nq | nv | 差值 | 原因 |
| :--- | --: | --: | ---: | :--- |
| 只有 hinge/slide | 相等 | 相等 | 0 | 位置和速度维度 1:1 |
| 有 1 个 free joint | nv+1 | nv | +1 | 四元数(4) vs 角速度(3) |
| 有 1 个 ball joint | nv+1 | nv | +1 | 同上 |
| 有 n 个四元数关节 | nv+n | nv | +n | 每个四元数多占 1 维 |

\[
nq - nv = \text{free关节数} + \text{ball关节数}
\]

---

## 7. 可视化对比

脚本会生成 `robot_comparison.png`，包含两张图：

1. **DOF 堆叠柱状图**：Base DOF（红色）+ Joint DOF（蓝色）
2. **nq / nv / nu 并排对比**：三组柱状图

```python
plt.savefig("robot_comparison.png", dpi=150)
```

---

## 8. 总结

```
┌──────────────────────────────────────────────────────────────┐
│            机器人类型对比 — 核心结论                            │
│                                                              │
│  固定基座 (UR5e, Franka, ALOHA):                              │
│    • 无 free joint，qpos 全部是关节值                          │
│    • nq == nv，维度一致                                       │
│    • nu == nv，完全驱动                                       │
│                                                              │
│  浮动基座 (Go2, H1):                                          │
│    • 有 free joint，qpos 前 7 位 = 基座位姿                    │
│    • nq == nv + 1（四元数多 1 维）                              │
│    • nu < nv，基座 6 DOF 欠驱动                                │
│                                                              │
│  数据平台影响:                                                │
│    • 存储轨迹时必须区分基座 qpos 和关节 qpos                    │
│    • 浮动基座的四元数需要归一化                                 │
│    • 不同机器人的 qpos 维度不同，不能直接拼接                    │
└──────────────────────────────────────────────────────────────┘
```
