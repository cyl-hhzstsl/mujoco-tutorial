# Jacobian 矩阵详解：从数学定义到机器人应用

> Jacobian 矩阵是连接关节空间与任务空间的桥梁，几乎所有运动学、动力学、控制算法都绕不开它。

---

## 1. 数学定义

### 1.1 一般定义

给定向量值函数 **f** : ℝⁿ → ℝᵐ，其 Jacobian 矩阵定义为：

```
                ┌  ∂f₁/∂x₁   ∂f₁/∂x₂   ···   ∂f₁/∂xₙ  ┐
                │  ∂f₂/∂x₁   ∂f₂/∂x₂   ···   ∂f₂/∂xₙ  │
J(x)  =  ∂f/∂x │     ⋮          ⋮        ⋱       ⋮      │  ∈ ℝᵐˣⁿ
                └  ∂fₘ/∂x₁   ∂fₘ/∂x₂   ···   ∂fₘ/∂xₙ  ┘
```

即 Jᵢⱼ = ∂fᵢ/∂xⱼ。

直觉：**J 是函数 f 在某一点处最好的线性近似**（一阶 Taylor 展开）：

```
f(x + δx) ≈ f(x) + J(x) · δx
```

### 1.2 在机器人学中的含义

正运动学函数 FK 将关节角度 **q** 映射到末端位置 **x**：

```
x = FK(q),    FK: ℝⁿ → ℝ³
```

对时间求导，得到速度层面的关系：

```
ẋ = J(q) · q̇
```

即 **末端速度 = Jacobian × 关节速度**。

J(q) 的每一列是"仅该关节运动时，末端产生的瞬时速度方向和大小"。

---

## 2. 机器人学中的两种 Jacobian

### 2.1 位置 Jacobian Jₚ

映射关节速度到末端**线速度**：

```
v = Jₚ · q̇       Jₚ ∈ ℝ³ˣⁿ
```

### 2.2 旋转 Jacobian Jᵣ

映射关节速度到末端**角速度**：

```
ω = Jᵣ · q̇       Jᵣ ∈ ℝ³ˣⁿ
```

### 2.3 完整 Jacobian

将两者堆叠得到 6×n 的完整 Jacobian：

```
┌ v ┐     ┌ Jₚ ┐
│   │  =  │    │ · q̇       J ∈ ℝ⁶ˣⁿ
└ ω ┘     └ Jᵣ ┘
```

### 2.4 MuJoCo API

```python
jacp = np.zeros((3, nv))   # 位置 Jacobian
jacr = np.zeros((3, nv))   # 旋转 Jacobian
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
# 也可用 mj_jacBody / mj_jacBodyCom / mj_jacGeom
```

> 参见 `chapter_08_advanced/01_kinematics.py` 中 `compute_jacobian()` 的实现。

---

## 3. 不同关节类型对 Jacobian 的贡献

| 关节类型 | 对 Jₚ 列的贡献 | 对 Jᵣ 列的贡献 | nq | nv |
|:---------|:---------------|:---------------|:--:|:--:|
| hinge (转动) | zᵢ × (p_ee − pᵢ) | zᵢ | 1 | 1 |
| slide (平动) | zᵢ | **0** | 1 | 1 |
| ball (球) | — (需 3 列) | — (需 3 列) | 4 (四元数) | 3 |
| free (自由) | I₃ (前 3 列) + 旋转贡献 | 旋转贡献 | 7 | 6 |

其中 **zᵢ** 是关节 i 的轴方向，**pᵢ** 是关节 i 的位置。

注意：**nq ≠ nv** 对于 ball/free 关节——这是 MuJoCo 中 qpos 和 qvel 维度不同的根源。Jacobian 的列数 = nv（不是 nq）。

> 参见 `chapter_03_qpos_deep_dive/04_qpos_qvel_relationship.py` 中关于 nq vs nv 的讨论。

---

## 4. Jacobian 的几何直觉

### 4.1 列空间 = 末端能动的方向

```
J = [ j₁ | j₂ | ··· | jₙ ]

jᵢ = 仅关节 i 运动时，末端的瞬时速度
```

**列空间 Col(J)**：所有可能的末端速度的集合。若 rank(J) < m（任务维度），说明有些方向末端**动不了** → 奇异。

### 4.2 零空间 = "内部运动"

若 n > m（冗余机器人，关节数 > 任务维度），Jacobian 有非平凡零空间：

```
J · q̇_null = 0
```

关节在动，但末端不动。这是冗余自由度的核心利用方式——在保持末端不动的前提下，优化关节姿态（避障、远离限位等）。

```
q̇ = J⁺ ẋ  +  (I − J⁺J) q̇₀
     ─────     ──────────────
     任务空间     零空间分量
      分量      （不影响末端）
```

### 4.3 可操作性椭球

```
ẋᵀ (J Jᵀ)⁻¹ ẋ  ≤  1
```

这个椭球描述了"单位关节速度"能产生的末端速度集合：
- **长轴方向**：末端最容易运动的方向
- **短轴方向**：末端最难运动的方向
- **椭球退化为低维**：奇异配置

可操作性指标（Yoshikawa, 1985）：

```
w = √det(J Jᵀ) = σ₁ · σ₂ ··· σₘ    （奇异值之积）
```

w = 0 → 奇异；w 越大 → 操作性越好。

---

## 5. 奇异性

### 5.1 什么是奇异

当 rank(J) < min(m, n) 时，某些末端速度方向**无法实现**。

### 5.2 物理例子

以 3 连杆平面臂为例：

| 配置 | 现象 | 条件数 |
|:-----|:-----|------:|
| 完全伸展 (0°, 0°, 0°) | 沿臂方向无法运动 | 很大 |
| 完全折叠 (0°, 180°, 0°) | 退化为一个点 | → ∞ |
| 弯曲 45° | 各方向都能运动 | 较小 |

### 5.3 条件数

```
κ(J) = σ_max / σ_min
```

```python
cond = np.linalg.cond(J)
```

| 条件数 | 含义 |
|------:|:-----|
| < 10 | 状态良好 |
| > 100 | 接近奇异 |
| → ∞ | 完全奇异 |

> `chapter_08_advanced/01_kinematics.py` 中的工作空间分析用条件数着色来可视化奇异区域。

---

## 6. Jacobian 的逆问题：从任务空间到关节空间

这是 IK / 控制的核心问题：已知期望末端速度 ẋ_d，求关节速度 q̇。

### 6.1 方阵 & 满秩（n = m，非奇异）

直接求逆：

```
q̇ = J⁻¹ ẋ_d
```

### 6.2 欠约束（n > m，冗余机器人）

有无穷多解，用**伪逆**取最小范数解：

```
q̇ = J⁺ ẋ_d       其中 J⁺ = Jᵀ(J Jᵀ)⁻¹
```

### 6.3 过约束（n < m，关节不够）

没有精确解，用伪逆取最小误差解：

```
q̇ = J⁺ ẋ_d       其中 J⁺ = (JᵀJ)⁻¹ Jᵀ
```

### 6.4 阻尼最小二乘 (DLS / Levenberg-Marquardt)

当接近奇异时，J⁺ 数值不稳定。加阻尼项 λ：

```
q̇ = Jᵀ (J Jᵀ + λ²I)⁻¹ ẋ_d
```

λ 的效果：

| λ | 行为 |
|:--|:-----|
| λ → 0 | 趋近于真正的伪逆（精确但可能爆炸） |
| λ 较大 | 牺牲精度换稳定性（动作变慢但安全） |
| 自适应 λ | 根据条件数动态调整（Nakamura & Hanafusa, 1986） |

> 这正是 `chapter_08_advanced/01_kinematics.py` 中 `JacobianIKSolver` 使用的方法。

---

## 7. Jacobian 的计算方式

### 7.1 解析法

根据 DH 参数或运动链结构，用公式直接推导。对于 hinge 关节：

```
Jₚ 第 i 列 = zᵢ × (p_ee − pᵢ)
Jᵣ 第 i 列 = zᵢ
```

### 7.2 MuJoCo 内置

MuJoCo 直接提供高效的 Jacobian 计算，不需要手动推导：

```python
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)    # 对 site
mujoco.mj_jacBody(model, data, jacp, jacr, body_id)     # 对 body origin
mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)  # 对 body COM
```

### 7.3 有限差分（数值验证用）

中心差分公式：

```
J 的第 j 列 ≈ [ FK(q + ε·eⱼ) − FK(q − ε·eⱼ) ] / 2ε
```

```python
eps = 1e-6
for j in range(nv):
    q[j] += eps;  x_plus = FK(q)
    q[j] -= 2*eps;  x_minus = FK(q)
    q[j] += eps
    J[:, j] = (x_plus - x_minus) / (2 * eps)
```

> `chapter_08_advanced/01_kinematics.py` 中的 `demo_jacobian()` 做了解析 vs 有限差分的交叉验证。

### 7.4 自动微分

JAX / PyTorch 等框架可以自动计算 Jacobian：

```python
import jax
J = jax.jacobian(fk_function)(q)
```

---

## 8. Jacobian 在各场景中的应用

### 8.1 逆运动学 (IK)

```
迭代:
  Δx = x_target − FK(q)          ← 位置误差
  Δq = J⁺ Δx  (或 DLS 版)       ← Jacobian 伪逆求关节增量
  q  ← q + α · Δq               ← 更新关节角度
```

> 详见 `notes/ik_methods_comparison.md`

### 8.2 速度控制 (Resolved Rate Control)

直接用 Jacobian 做末端速度 → 关节速度的实时映射：

```
q̇_cmd = J⁺(q) · ẋ_desired
```

### 8.3 力映射 (静力学)

关节力矩与末端力的关系（虚功原理）：

```
τ = Jᵀ · F

关节力矩 = Jacobian转置 × 末端力
```

这在力控制 / 阻抗控制中至关重要。注意方向是反过来的：
- **速度**：关节 → 末端，用 **J**
- **力**：末端 → 关节，用 **Jᵀ**

### 8.4 动力学

Jacobian 出现在操作空间动力学中（Khatib, 1987）：

```
操作空间惯量:  Λ = (J M⁻¹ Jᵀ)⁻¹
操作空间力:    F = Λ ẍ_d + μ + p

其中 M = 关节空间质量矩阵
```

### 8.5 碰撞检测与回避

Jacobian 可以计算"某个 geom 的速度对关节速度的灵敏度"，用于碰撞回避约束：

```
d(碰撞距离)/dt = J_col · q̇  ≥  −α · (距离 − 安全边距)
```

> mink 库中的 `CollisionAvoidanceLimit` 内部就是这个原理。

### 8.6 可操作性分析与工作空间

- 遍历关节空间，计算每个配置的 w = √det(JJᵀ) → 操作性热力图
- 条件数 → 奇异区域标记
- 椭球可视化 → 各方向的运动能力

> `chapter_08_advanced/01_kinematics.py` 中的 `analyze_workspace()` 做了完整的工作空间分析。

---

## 9. 常见坑

### 9.1 Jacobian 列数是 nv，不是 nq

MuJoCo 中对于 ball/free 关节，qpos 用四元数（4/7 维），但 qvel 用角速度（3/6 维）。
Jacobian 矩阵的列数 = nv（速度空间维度），而非 nq。

### 9.2 配置依赖

J 是关节配置 q 的函数，**每换一个 q 都要重新计算**。使用前必须确保 `mj_forward()` 已经用当前 qpos 执行过。

### 9.3 奇异点附近不要用裸伪逆

裸 J⁺ 在奇异点附近会产生巨大的关节速度。务必使用 DLS（加阻尼）或 QP 方法。

### 9.4 MuJoCo Jacobian 的坐标系

`mj_jacSite` 返回的是**世界坐标系**下的 Jacobian。如果需要末端坐标系下的，需要额外旋转：

```python
R = data.site_xmat[site_id].reshape(3, 3)
jacp_local = R.T @ jacp
```

---

## 10. 速查总结

| 概念 | 公式 | 说明 |
|:-----|:-----|:-----|
| 定义 | J = ∂**f**/∂**x** | 向量函数的一阶导数矩阵 |
| 速度映射 | ẋ = J q̇ | 关节 → 末端 |
| 力映射 | τ = Jᵀ F | 末端 → 关节 |
| 位置 Jacobian | Jₚ ∈ ℝ³ˣⁿᵥ | 线速度 |
| 旋转 Jacobian | Jᵣ ∈ ℝ³ˣⁿᵥ | 角速度 |
| 阻尼最小二乘 | Δq = Jᵀ(JJᵀ + λ²I)⁻¹ Δx | IK 核心公式 |
| 条件数 | κ = σ_max / σ_min | 奇异性度量 |
| 可操作性 | w = √det(JJᵀ) | 操作性能指标 |

**MuJoCo API**: `mj_jacSite` / `mj_jacBody` / `mj_jacBodyCom`（列数 = nv，不是 nq）

**教程中的实践**:
- `chapter_08_advanced/01_kinematics.py` — Jacobian 计算 + 数值验证
- `chapter_03_qpos_deep_dive/04_*.py` — nq vs nv 的关系
- `notes/ik_methods_comparison.md` — IK 方案中的 Jacobian 应用
