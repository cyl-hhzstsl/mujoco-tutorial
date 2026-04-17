# 第 8 章 · 01 — 正运动学与逆运动学 (FK / IK)

> **目标**: 理解关节空间与笛卡尔空间的映射关系，掌握 Jacobian 矩阵在 IK 和工作空间分析中的应用。

## 核心知识点

1. 正运动学 (FK): qpos → 末端执行器位置
2. Jacobian 矩阵: 速度映射 ẋ = J · q̇
3. 逆运动学 (IK): 基于 Jacobian 伪逆的迭代求解
4. 工作空间分析: 可达区域与奇异性

---

## 1. 正运动学 (Forward Kinematics)

给定关节角度 → 计算末端执行器位置。

```python
data.qpos[:] = [θ1, θ2, θ3]
mujoco.mj_forward(model, data)
ee_pos = data.site_xpos[site_id]  # (x, y, z)
```

**关键调用**: `mj_forward()` 执行所有正运动学计算，更新所有 body/site 的位置和姿态。

### 验证

零位时（所有关节 0°），3 连杆臂末端位置：
\[x = L_1 + L_2 + L_3 = 0.3 + 0.25 + 0.2 = 0.75 \text{ m}\]

---

## 2. Jacobian 矩阵

物理含义：末端速度 = Jacobian × 关节速度

\[\dot{x} = J \cdot \dot{q}\]

```python
jacp = np.zeros((3, nv))  # 位置 Jacobian
jacr = np.zeros((3, nv))  # 旋转 Jacobian
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
```

### 条件数

```python
cond = np.linalg.cond(jacp)
```

| 条件数 | 含义 |
| -----: | :--- |
| < 10 | 状态良好，操作性好 |
| > 100 | 接近奇异，末端运动受限 |
| → ∞ | 奇异配置（如完全伸展/折叠） |

### 数值验证

解析 Jacobian 与有限差分对比，误差应 < 1e-4。

---

## 3. 逆运动学 (Inverse Kinematics)

给定目标位置 → 求解关节角度。

### 算法：阻尼最小二乘 (Damped Least Squares)

```
迭代:
  1. Δx = target - current_pos          (位置误差)
  2. Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ Δx       (阻尼伪逆)
  3. q = q + α · Δq                     (更新关节角度)
  4. 限位约束: clip(q, q_min, q_max)
```

### JacobianIKSolver 参数

| 参数 | 默认值 | 含义 |
| :--- | -----: | :--- |
| max_iter | 200 | 最大迭代次数 |
| tol | 1e-4 | 收敛容差 (m) |
| step_size | 0.5 | 学习率 α |
| damping | 1e-4 | 阻尼系数 λ（防止奇异） |

### IK 的特殊情况

- **多解**: 不同初始值可能收敛到不同解
- **无解**: 目标超出工作空间
- **奇异**: Jacobian 退化，阻尼项提供数值稳定性

---

## 4. 工作空间分析

遍历关节空间，通过 FK 映射到笛卡尔空间：

```python
for θ1 in linspace(range1):
  for θ2 in linspace(range2):
    for θ3 in linspace(range3):
      FK(θ1, θ2, θ3) → (x, y, z)
```

### 统计指标

- X/Z 范围、最大可达距离
- 奇异点比例（条件数 > 100 的配置）

### 可视化

3 子图：XZ 平面工作空间（条件数着色）、可达距离分布、条件数分布。

---

## 5. 数据工程视角

| 场景 | FK/IK 的作用 |
| :--- | :----------- |
| 数据清洗 | 用 FK 验证 qpos 是否产生合理的末端位置 |
| 轨迹校验 | 检查末端轨迹是否在工作空间内 |
| 数据可视化 | 将关节空间数据映射到直观的笛卡尔空间 |
| 异常检测 | 高条件数区域的数据可能质量较差 |

---

## 6. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              运动学 — 核心要点                                  │
│                                                              │
│  FK: qpos → 末端位置 (确定性映射)                              │
│    API: mj_forward() + data.site_xpos                        │
│                                                              │
│  Jacobian: ẋ = J · q̇ (速度映射)                              │
│    API: mj_jacSite()                                         │
│    条件数衡量操作性能                                          │
│                                                              │
│  IK: 目标位置 → qpos (迭代求解)                               │
│    阻尼最小二乘: Δq = Jᵀ(JJᵀ + λ²I)⁻¹Δx                    │
│    可能多解 / 无解 / 接近奇异                                  │
│                                                              │
│  工作空间: 遍历关节空间 → 可达区域 + 奇异区域                   │
└──────────────────────────────────────────────────────────────┘
```
