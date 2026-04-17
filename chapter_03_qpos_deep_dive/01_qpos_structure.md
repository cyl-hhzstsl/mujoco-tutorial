# 第 3 章 · 01 — qpos 内存布局全解析

> **目标**: 彻底搞懂 `qpos` 的每一个元素是什么意思，掌握从 qpos 索引反查关节名称的方法。

## 核心知识点

1. 每种关节类型对 `qpos` 的维度贡献
2. `jnt_qposadr` / `jnt_dofadr` 索引映射
3. `nq != nv` 的本质原因
4. 反向查找：`qpos[i]` → 哪个关节的哪个分量

### Java 类比

`qpos` 就像一个扁平的 `byte[]` 缓冲区，`jnt_qposadr` 就是各字段的 offset——你需要知道每个 offset 对应的字段名和长度。

---

## 1. 关节类型常量与元信息

MuJoCo 定义了四种关节类型，每种在 `qpos`（广义坐标）和 `qvel`（广义速度）中占用不同的维度：

| 关节类型 | 编号 | qpos 维度 (nq) | qvel 维度 (nv) |
| :------: | :--: | :------------: | :------------: |
| free     | 0    | 7              | 6              |
| ball     | 1    | 4              | 3              |
| slide    | 2    | 1              | 1              |
| hinge    | 3    | 1              | 1              |

### qpos 各分量的语义标签

| 关节类型 | qpos 分量                                                        |
| :------: | :--------------------------------------------------------------- |
| free     | `pos_x`, `pos_y`, `pos_z`, `quat_w`, `quat_x`, `quat_y`, `quat_z` |
| ball     | `quat_w`, `quat_x`, `quat_y`, `quat_z`                          |
| slide    | `displacement`（位移）                                            |
| hinge    | `angle_rad`（弧度角）                                             |

### qvel 各分量的语义标签

| 关节类型 | qvel 分量                                                |
| :------: | :------------------------------------------------------- |
| free     | `vel_x`, `vel_y`, `vel_z`, `angvel_x`, `angvel_y`, `angvel_z` |
| ball     | `angvel_x`, `angvel_y`, `angvel_z`                        |
| slide    | `velocity`（速度）                                        |
| hinge    | `angular_vel`（角速度）                                    |

---

## 2. 逐个关节类型分析

### 2a. Free Joint（自由关节）

```xml
<body name="floating_box" pos="0 0 1">
  <joint name="free_jnt" type="free"/>
  <geom type="box" size="0.1 0.1 0.1" mass="1"/>
</body>
```

- **nq = 7**（qpos 长度）
- **nv = 6**（qvel 长度）
- 默认值：`[0, 0, 1, 1, 0, 0, 0]`

各分量含义：

| 索引 | 默认值 | 含义    |
| :--: | -----: | :------ |
| 0    | 0.0    | pos_x   |
| 1    | 0.0    | pos_y   |
| 2    | 1.0    | pos_z   |
| 3    | 1.0    | quat_w  |
| 4    | 0.0    | quat_x  |
| 5    | 0.0    | quat_y  |
| 6    | 0.0    | quat_z  |

**解读**：

- **位置部分** `qpos[0:3]` = `[0, 0, 1]` → 物体在世界坐标系中的 (x, y, z) 位置
- **旋转部分** `qpos[3:7]` = `[1, 0, 0, 0]` → 四元数 (w, x, y, z)，`[1,0,0,0]` 是单位旋转（无旋转）

### 2b. Ball Joint（球铰关节）

```xml
<body name="arm" pos="0 0 1">
  <joint name="ball_jnt" type="ball"/>
  <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3" mass="1"/>
</body>
```

- **nq = 4**（qpos 长度）
- **nv = 3**（qvel 长度）
- 默认值：`[1, 0, 0, 0]`（单位四元数）

各分量含义：

| 索引 | 默认值 | 含义    |
| :--: | -----: | :------ |
| 0    | 1.0    | quat_w  |
| 1    | 0.0    | quat_x  |
| 2    | 0.0    | quat_y  |
| 3    | 0.0    | quat_z  |

> **注意**：nq=4 但 nv=3，四元数 4 维 → 角速度 3 维

### 2c. Hinge Joint（铰链关节）

```xml
<body name="door" pos="0 0 1">
  <joint name="hinge_jnt" type="hinge" axis="0 0 1"/>
  <geom type="box" size="0.3 0.02 0.4" mass="2"/>
</body>
```

- **nq = 1**（qpos 长度）
- **nv = 1**（qvel 长度）
- 默认值：`[0.0]`

各分量含义：

| 索引 | 默认值 | 含义             |
| :--: | -----: | :--------------- |
| 0    | 0.0    | angle_rad（弧度） |

> nq=nv=1，铰链关节最简单。

### 2d. Slide Joint（滑动关节）

```xml
<body name="slider" pos="0 0 1">
  <joint name="slide_jnt" type="slide" axis="1 0 0"/>
  <geom type="box" size="0.1 0.1 0.1" mass="1"/>
</body>
```

- **nq = 1**（qpos 长度）
- **nv = 1**（qvel 长度）
- 默认值：`[0.0]`

各分量含义：

| 索引 | 默认值 | 含义              |
| :--: | -----: | :---------------- |
| 0    | 0.0    | displacement（米） |

> nq=nv=1，滑动关节也很简单。

---

## 3. 组合模型：完整 qpos 内存布局

以一个包含所有关节类型的组合模型为例：

```xml
<mujoco>
  <worldbody>
    <!-- 自由浮动基座 -->
    <body name="base" pos="0 0 1">
      <joint name="base_free" type="free"/>
      <geom type="box" size="0.2 0.2 0.05" mass="5"/>

      <!-- 肩部球铰 -->
      <body name="shoulder" pos="0.2 0 0">
        <joint name="shoulder_ball" type="ball"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0" mass="1"/>

        <!-- 肘部铰链 -->
        <body name="elbow" pos="0.3 0 0">
          <joint name="elbow_hinge" type="hinge" axis="0 1 0"
                 range="-2.0 2.0" limited="true"/>
          <geom type="capsule" size="0.025" fromto="0 0 0 0.25 0 0" mass="0.8"/>

          <!-- 腕部铰链 -->
          <body name="wrist" pos="0.25 0 0">
            <joint name="wrist_hinge" type="hinge" axis="0 0 1"
                   range="-1.57 1.57" limited="true"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.3"/>

            <!-- 手指滑动 -->
            <body name="finger" pos="0.1 0 0">
              <joint name="finger_slide" type="slide" axis="0 1 0"
                     range="0 0.05" limited="true"/>
              <geom type="box" size="0.01 0.02 0.03" mass="0.1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
```

### 总维度

- **nq = 14**（qpos 总长度）
- **nv = 12**（qvel 总长度）
- **njnt = 5**（关节数）
- **nq - nv = 2**（因为 free 和 ball 各有一个四元数）

### 关节布局表

| 关节名          | 类型   | qpos 起始 | qpos 长度 | qvel 起始 | qvel 长度 |
| :-------------- | :----- | --------: | --------: | --------: | --------: |
| base_free       | free   |         0 |         7 |         0 |         6 |
| shoulder_ball   | ball   |         7 |         4 |         6 |         3 |
| elbow_hinge     | hinge  |        11 |         1 |         9 |         1 |
| wrist_hinge     | hinge  |        12 |         1 |        10 |         1 |
| finger_slide    | slide  |        13 |         1 |        11 |         1 |

### 完整 qpos 索引映射表

| 索引       | 默认值 | 关节名          | 含义    |
| :--------: | -----: | :-------------- | :------ |
| qpos[0]    | 0.0    | base_free       | pos_x   |
| qpos[1]    | 0.0    | base_free       | pos_y   |
| qpos[2]    | 1.0    | base_free       | pos_z   |
| qpos[3]    | 1.0    | base_free       | quat_w  |
| qpos[4]    | 0.0    | base_free       | quat_x  |
| qpos[5]    | 0.0    | base_free       | quat_y  |
| qpos[6]    | 0.0    | base_free       | quat_z  |
| qpos[7]    | 1.0    | shoulder_ball   | quat_w  |
| qpos[8]    | 0.0    | shoulder_ball   | quat_x  |
| qpos[9]    | 0.0    | shoulder_ball   | quat_y  |
| qpos[10]   | 0.0    | shoulder_ball   | quat_z  |
| qpos[11]   | 0.0    | elbow_hinge     | angle_rad |
| qpos[12]   | 0.0    | wrist_hinge     | angle_rad |
| qpos[13]   | 0.0    | finger_slide    | displacement |

### 完整 qvel 索引映射表

| 索引       | 默认值 | 关节名          | 含义       |
| :--------: | -----: | :-------------- | :--------- |
| qvel[0]    | 0.0    | base_free       | vel_x      |
| qvel[1]    | 0.0    | base_free       | vel_y      |
| qvel[2]    | 0.0    | base_free       | vel_z      |
| qvel[3]    | 0.0    | base_free       | angvel_x   |
| qvel[4]    | 0.0    | base_free       | angvel_y   |
| qvel[5]    | 0.0    | base_free       | angvel_z   |
| qvel[6]    | 0.0    | shoulder_ball   | angvel_x   |
| qvel[7]    | 0.0    | shoulder_ball   | angvel_y   |
| qvel[8]    | 0.0    | shoulder_ball   | angvel_z   |
| qvel[9]    | 0.0    | elbow_hinge     | angular_vel |
| qvel[10]   | 0.0    | wrist_hinge     | angular_vel |
| qvel[11]   | 0.0    | finger_slide    | velocity   |

---

## 4. jnt_qposadr 与 jnt_dofadr 详解

```python
model.jnt_qposadr  # 每个关节在 qpos 中的起始索引
model.jnt_dofadr   # 每个关节在 qvel 中的起始索引
```

- `jnt_qposadr[i]` 告诉你：关节 `i` 的数据从 `qpos` 的第几个元素开始
- `jnt_dofadr[i]` 告诉你：关节 `i` 的数据从 `qvel` 的第几个元素开始

### 当前模型的索引

| 关节            | jnt_qposadr | 占 qpos 维度 | jnt_dofadr | 占 qvel 维度 |
| :-------------- | ----------: | -----------: | ---------: | -----------: |
| base_free       |           0 |            7 |          0 |            6 |
| shoulder_ball   |           7 |            4 |          6 |            3 |
| elbow_hinge     |          11 |            1 |          9 |            1 |
| wrist_hinge     |          12 |            1 |         10 |            1 |
| finger_slide    |          13 |            1 |         11 |            1 |

### 验证地址连续性

- `0 + 7 = 7`（base_free 结束 = shoulder_ball 开始）
- `7 + 4 = 11`（shoulder_ball 结束 = elbow_hinge 开始）
- `11 + 1 = 12`（elbow_hinge 结束 = wrist_hinge 开始）
- `12 + 1 = 13`（wrist_hinge 结束 = finger_slide 开始）

---

## 5. nq ≠ nv — 维度不匹配的根本原因

| 关节            | qpos 贡献          | qvel 贡献           | 差异 |
| :-------------- | :----------------- | :------------------ | ---: |
| base_free       | 7 (pos3 + quat4)   | 6 (vel3 + angvel3)  |   +1 |
| shoulder_ball   | 4 (quat4)          | 3 (angvel3)         |   +1 |
| elbow_hinge     | 1                  | 1                   |    0 |
| wrist_hinge     | 1                  | 1                   |    0 |
| finger_slide    | 1                  | 1                   |    0 |
| **总计**        | **nq = 14**        | **nv = 12**         | **+2** |

### 根本原因

- 旋转的「位置」用**四元数**表示：4 个数 `(w, x, y, z)`
- 旋转的「速度」用**角速度**表示：3 个数 `(ωx, ωy, ωz)`
- 四元数有归一化约束 \(w^2 + x^2 + y^2 + z^2 = 1\)，实际只有 3 个自由度
- 所以每个四元数多占 1 维：`nq` 比 `nv` 多出来的就是四元数的个数

### 公式

\[
nq - nv = \text{free关节数} \times 1 + \text{ball关节数} \times 1
\]

本例：`14 - 12 = 1 (free) + 1 (ball) = 2`

---

## 6. 反向查找：qpos 索引 → 关节信息

### 构建 qpos 索引映射

```python
def build_qpos_index_map(model):
    """构建 qpos 索引 → (关节名, 关节类型, 分量标签) 的映射表。"""
    index_map = {}
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = model.jnt_type[j]
        qpos_adr = model.jnt_qposadr[j]
        labels = QPOS_LABELS[jtype]
        for k, label in enumerate(labels):
            index_map[qpos_adr + k] = {
                "joint_name": jname,
                "joint_id": j,
                "joint_type": JOINT_DIMS[jtype]["name"],
                "component": label,
                "offset": k,
            }
    return index_map
```

### 查询函数

```python
def lookup_qpos(model, index):
    """查询 qpos[index] 属于哪个关节的哪个分量"""
    qmap = build_qpos_index_map(model)
    if index in qmap:
        info = qmap[index]
        return f"qpos[{index}] → 关节 '{info['joint_name']}' ({info['joint_type']}) 的 {info['component']}"
    return f"qpos[{index}] → 无效索引 (nq={model.nq})"
```

### 演示输出

```
qpos[ 0] → 关节 'base_free'     (free)  的 pos_x
qpos[ 1] → 关节 'base_free'     (free)  的 pos_y
qpos[ 2] → 关节 'base_free'     (free)  的 pos_z
qpos[ 3] → 关节 'base_free'     (free)  的 quat_w
qpos[ 4] → 关节 'base_free'     (free)  的 quat_x
qpos[ 5] → 关节 'base_free'     (free)  的 quat_y
qpos[ 6] → 关节 'base_free'     (free)  的 quat_z
qpos[ 7] → 关节 'shoulder_ball' (ball)  的 quat_w
qpos[ 8] → 关节 'shoulder_ball' (ball)  的 quat_x
qpos[ 9] → 关节 'shoulder_ball' (ball)  的 quat_y
qpos[10] → 关节 'shoulder_ball' (ball)  的 quat_z
qpos[11] → 关节 'elbow_hinge'   (hinge) 的 angle_rad
qpos[12] → 关节 'wrist_hinge'   (hinge) 的 angle_rad
qpos[13] → 关节 'finger_slide'  (slide) 的 displacement
```

---

## 7. 实用工具：按关节名获取/设置 qpos 切片

### 读取关节 qpos

```python
def get_joint_qpos(model, data, joint_name):
    """获取指定关节在 qpos 中的值（返回数组切片）"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    jtype = model.jnt_type[jid]
    adr = model.jnt_qposadr[jid]
    length = JOINT_DIMS[jtype]["nq"]
    return data.qpos[adr:adr + length]
```

### 设置关节 qpos

```python
def set_joint_qpos(model, data, joint_name, value):
    """设置指定关节在 qpos 中的值"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    jtype = model.jnt_type[jid]
    adr = model.jnt_qposadr[jid]
    length = JOINT_DIMS[jtype]["nq"]
    value = np.asarray(value)
    assert value.shape == (length,)
    data.qpos[adr:adr + length] = value
```

### 使用示例

```python
# 读取各关节值
get_joint_qpos(model, data, "base_free")      # → [0, 0, 1, 1, 0, 0, 0]
get_joint_qpos(model, data, "shoulder_ball")   # → [1, 0, 0, 0]
get_joint_qpos(model, data, "elbow_hinge")     # → [0.0]

# 修改肘部角度为 0.5 弧度
set_joint_qpos(model, data, "elbow_hinge", np.array([0.5]))
```

---

## 8. 总结

```
┌─────────────────────────────────────────────────────────┐
│                    qpos 内存布局                         │
│                                                         │
│  关节类型    qpos维度   qvel维度   默认值                │
│  ────────   ────────   ────────   ──────────            │
│  free       7          6          [0,0,0, 1,0,0,0]     │
│  ball       4          3          [1,0,0,0]             │
│  hinge      1          1          0                     │
│  slide      1          1          0                     │
│                                                         │
│  nq - nv = free关节数 + ball关节数                       │
│                                                         │
│  关键 API:                                              │
│    model.jnt_qposadr[i] → 关节 i 在 qpos 中的起始位     │
│    model.jnt_dofadr[i]  → 关节 i 在 qvel 中的起始位     │
│    model.jnt_type[i]    → 关节 i 的类型 (0/1/2/3)      │
└─────────────────────────────────────────────────────────┘
```
