# 第 4 章 · 01 — 加载真实机器人模型

> **目标**: 学会从 `robot_descriptions` 库加载工业级机器人模型，逐个分析其关节结构、执行器配置和 qpos 布局。

## 核心知识点

1. `robot_descriptions` 库的使用方法
2. 不同机器人的 nq/nv/nu 差异
3. 关节类型映射与 qpos 索引
4. 执行器与关节的绑定关系

---

## 1. robot_descriptions 库

`robot_descriptions` 是一个 Python 包，提供了常见机器人模型的 MJCF / URDF 文件路径。

```bash
pip install robot_descriptions
```

### 支持的模型（本脚本涉及）

| 名称 | 模块路径 | 属性 |
| :--- | :------- | :--- |
| UR5e | `robot_descriptions.ur5e_mj_description` | `MJCF_PATH` |
| Franka Panda | `robot_descriptions.panda_mj_description` | `MJCF_PATH` |
| Unitree Go2 | `robot_descriptions.go2_mj_description` | `MJCF_PATH` |
| Unitree H1 | `robot_descriptions.h1_mj_description` | `MJCF_PATH` |
| ALOHA | `robot_descriptions.aloha_mj_description` | `MJCF_PATH` |

### 加载流程

```python
import importlib
mod = importlib.import_module("robot_descriptions.ur5e_mj_description")
mjcf_path = mod.MJCF_PATH        # 返回 .xml 文件的绝对路径
model = mujoco.MjModel.from_xml_path(mjcf_path)
```

> **回退机制**：脚本在 `robot_descriptions` 不可用时，自动使用内置的简化 MJCF 字符串创建模型，确保在任何环境下都能运行。

---

## 2. 涉及的机器人及其特征

| 机器人 | 类型 | 基座 | 关节 DOF | 执行器 |
| :----- | :--- | :--- | :------: | :----: |
| UR5e | 6 轴协作机械臂 | 固定 | 6 (hinge) | 6 |
| Franka Panda | 7 轴机械臂 + 手指 | 固定 | 7 hinge + 2 slide | 9 |
| Unitree Go2 | 四足机器人 | 浮动 (free) | 12 (hinge) | 12 |
| Unitree H1 | 人形机器人 | 浮动 (free) | 19 (hinge) | 19 |
| ALOHA | 双臂遥操作 | 固定 | 12 hinge + 2 slide | 14 |

### 固定基座 vs 浮动基座

- **固定基座**（UR5e, Franka, ALOHA）：基座焊死在世界坐标系，qpos 直接从关节角开始
- **浮动基座**（Go2, H1）：有 `free joint`，qpos 前 7 位是基座位姿 `[x, y, z, qw, qx, qy, qz]`

---

## 3. 关节类型常量

```python
JNT_TYPE_NAMES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
JNT_QPOS_DIM   = {0: 7,      1: 4,      2: 1,       3: 1}
JNT_QVEL_DIM   = {0: 6,      1: 3,      2: 1,       3: 1}
```

这三个字典是贯穿全章的基础工具——给定关节类型编号，即可查询：
- 关节名称
- 在 qpos 中占用的维度
- 在 qvel 中占用的维度

---

## 4. 模型分析：关键 API

### 基本维度

```python
model.nq     # 广义坐标维度（qpos 长度）
model.nv     # 广义速度维度（qvel 长度，= DOF 数）
model.nu     # 执行器数量（ctrl 长度）
model.nbody  # 刚体数量
model.njnt   # 关节数量
model.ngeom  # 几何体数量
```

### 遍历关节

```python
for j in range(model.njnt):
    name     = model.joint(j).name          # 关节名称
    jnt_type = int(model.jnt_type[j])       # 关节类型 (0/1/2/3)
    qpos_adr = model.jnt_qposadr[j]        # 在 qpos 中的起始索引
    dof_adr  = model.jnt_dofadr[j]         # 在 qvel 中的起始索引
```

### 遍历执行器

```python
for a in range(model.nu):
    act_name = model.actuator(a).name          # 执行器名称
    trntype  = model.actuator_trntype[a]       # 传动类型 (0=joint)
    jnt_id   = model.actuator_trnid[a][0]      # 关联的关节 ID
    jnt_name = model.joint(jnt_id).name        # 关联的关节名
```

`actuator_trntype == 0` 表示执行器直接驱动关节（最常见情况）。

---

## 5. 初始 qpos 解读

### UR5e（固定基座，6 hinge）

```
qpos 长度 = 6
qpos[0..5] = 全部 hinge 关节角（弧度），初始均为 0
```

### Franka Panda（固定基座，7 hinge + 2 slide）

```
qpos 长度 = 9
qpos[0..6] = 7 个 hinge 关节角
qpos[7..8] = 2 个 slide 手指位移
```

### Unitree Go2（浮动基座，free + 12 hinge）

```
qpos 长度 = 19        ← 7 (free) + 12 (hinge)
qpos[0:3]  = 基座位置 [x, y, z]
qpos[3:7]  = 基座四元数 [qw, qx, qy, qz]
qpos[7:19] = 12 个腿关节角
```

### Unitree H1（浮动基座，free + 19 hinge）

```
qpos 长度 = 26        ← 7 (free) + 19 (hinge)
qpos[0:3]  = 基座位置 [x, y, z]
qpos[3:7]  = 基座四元数 [qw, qx, qy, qz]
qpos[7:26] = 19 个关节角（躯干 + 双腿 + 双臂）
```

### ALOHA（固定基座，12 hinge + 2 slide）

```
qpos 长度 = 14
qpos[0..5]   = 左臂 6 个 hinge
qpos[6]      = 左手 gripper (slide)
qpos[7..12]  = 右臂 6 个 hinge
qpos[13]     = 右手 gripper (slide)
```

---

## 6. 汇总对比

脚本最终会打印所有机器人的对比表：

```
  机器人              nq    nv    nu   njnt  浮动基座
  ──────────────────  ────  ────  ────  ─────  ────────
  UR5e (fallback)        6     6     6      6  否 (fixed)
  Franka (fallback)      9     9     9      9  否 (fixed)
  Go2 (fallback)        19    18    12     13  是 (free)
  H1 (fallback)         26    25    19     20  是 (free)
  ALOHA (fallback)      14    14    14     14  否 (fixed)
```

### 关键观察

- **固定基座**：nq == nv（只有 hinge 和 slide 关节）
- **浮动基座**：nq == nv + 1（四元数 4 维 vs 角速度 3 维）
- **nu ≤ nv**：浮动基座的 6 DOF 没有直接执行器

---

## 7. 执行器类型

本脚本涉及两种执行器类型：

| 类型 | XML 标签 | 含义 | 典型用途 |
| :--- | :------- | :--- | :------- |
| position | `<position>` | 位置伺服，`kp` 为比例增益 | 机械臂精确控制 |
| motor | `<motor>` | 力矩/力直接输出，`gear` 为传动比 | 腿部力控 |

```xml
<!-- 位置控制：ctrl 是目标角度 -->
<position name="act1" joint="joint1" kp="1000"/>

<!-- 力矩控制：ctrl 是目标力矩 -->
<motor name="FL_hip_act" joint="FL_hip_joint" gear="25"/>
```

---

## 8. 总结

```
┌─────────────────────────────────────────────────────────────┐
│              加载真实机器人模型 — 核心流程                     │
│                                                             │
│  1. robot_descriptions 库提供 MJCF 文件路径                   │
│  2. from_xml_path / from_xml_string 创建 MjModel             │
│  3. 分析 nq/nv/nu/njnt 掌握模型维度                          │
│  4. 遍历关节：jnt_type, jnt_qposadr, jnt_dofadr             │
│  5. 遍历执行器：actuator_trntype, actuator_trnid             │
│  6. 区分固定基座 vs 浮动基座对 qpos 布局的影响                 │
│                                                             │
│  关键 API:                                                  │
│    model.joint(j).name       → 关节名                        │
│    model.jnt_type[j]         → 关节类型                      │
│    model.jnt_qposadr[j]      → qpos 中的起始索引              │
│    model.actuator(a).name    → 执行器名                      │
│    model.actuator_trnid[a]   → 关联关节 ID                   │
└─────────────────────────────────────────────────────────────┘
```
