# 第 3 章：qpos 深度解析

> **本章是整个教程的核心。** 理解 `qpos` 是理解一切机器人仿真数据的基础。
>
> 如果你只能深入学一章，就学这章。

---

## 为什么 qpos 如此重要？

在机器人数据平台中，你接触到的每一条轨迹数据，核心就是 **qpos 序列**。
无论是遥操作采集、仿真生成还是实机回放，最终存储的都是每一帧的 `qpos`。

如果你不能准确回答以下问题，你的数据平台就可能埋下 bug：

- `qpos[7]` 代表什么？哪个关节的什么分量？
- 为什么 `len(qpos) != len(qvel)`？差了多少？为什么？
- 四元数 `[0, 0, 0, 1]` 和 `[1, 0, 0, 0]` 哪个是单位旋转？在 MuJoCo 里呢？
- 怎样正确地在两个 qpos 之间插值？能直接做线性插值吗？
- 如何从 qpos 推算出末端执行器的笛卡尔坐标？

学完本章，你将能自信地回答所有这些问题。

---

## 本章内容

| 文件 | 主题 | 关键概念 |
|------|------|----------|
| `01_qpos_structure.py` | qpos 内存布局 | 每种关节类型的贡献、索引映射、nq vs nv |
| `02_quaternion_deep_dive.py` | 四元数深度教程 | MuJoCo 四元数顺序、旋转、插值、归一化 |
| `03_qpos_manipulation.py` | qpos 实战操作 | 读写关节、正运动学、积分、差值、快照 |
| `04_qpos_qvel_relationship.py` | qpos/qvel 关系 | 维度不匹配、数值微分、雅可比矩阵、相空间 |
| `05_exercises.py` | 综合练习 | 带 assert 验证的填空题 |

---

## 核心概念预览

### 1. qpos = 广义位置向量

`qpos`（generalized position）完整描述了机器人所有关节的位置状态。

```
qpos = [x, y, z, qw, qx, qy, qz, θ₁, θ₂, θ₃, ...]
         ↑ 自由关节 (7维)          ↑ 铰链关节 (各1维)
```

### 2. 每种关节对 qpos 的贡献

| 关节类型 | qpos 维度 | qvel 维度 | 说明 |
|----------|-----------|-----------|------|
| `free`   | 7 (pos3 + quat4) | 6 (vel3 + angvel3) | 自由浮动体 |
| `ball`   | 4 (quat4) | 3 (angvel3) | 球铰（纯旋转） |
| `hinge`  | 1 (angle) | 1 (angular vel) | 单轴旋转 |
| `slide`  | 1 (displacement) | 1 (linear vel) | 单轴平移 |

**关键洞察：** `free` 和 `ball` 关节使用四元数表示旋转，4 维四元数对应 3 维角速度，
这就是 `nq ≠ nv` 的根本原因。

### 3. 四元数在 MuJoCo 中的顺序

```
MuJoCo:  [w, x, y, z]   ← w 在前！
scipy:   [x, y, z, w]   ← w 在后！
```

**这是最常见的 bug 来源之一**，切换库时务必注意！

### 4. qpos 索引映射

```
model.jnt_qposadr[i]  → 关节 i 在 qpos 中的起始索引
model.jnt_dofadr[i]   → 关节 i 在 qvel 中的起始索引
```

---

## 前置知识

- 第 1 章：MjModel 和 MjData 基本概念
- 第 2 章：MJCF 模型中的关节定义
- 第 0 章：NumPy 数组操作基础

## 预计学习时间

- 通读 + 运行所有脚本：3-4 小时
- 完成练习：1-2 小时
- 建议反复回来查阅，直到完全内化

---

**开始 → [01_qpos_structure.py](01_qpos_structure.py)**
