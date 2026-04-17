# 第 3 章 · 03 — qpos 实战操作

> **目标**: 掌握 qpos 的所有实用操作 — 读取、修改、积分、差值、正运动学、快照保存、插值、关节限位检查。

## 核心技能一览

| 技能 | MuJoCo API / 方法 | 适用场景 |
| :--- | :--- | :--- |
| 按名称读写关节 | `jnt_qposadr` + 数组切片 | 控制特定关节 |
| 正确积分 | `mj_integratePos` | 有 free/ball 关节时 |
| 计算差值 | `mj_differentiatePos` | 对比两个姿态 |
| 正运动学 | `mj_forward` → `site_xpos` | qpos → 笛卡尔坐标 |
| 快照保存 | JSON 序列化 | 数据记录与回放 |
| 插值 | 线性 / `integratePos` | 轨迹生成 |
| 限位检查 | `jnt_limited` + `jnt_range` | 数据质量校验 |

### 使用的模型

3 自由度机械臂：肩关节 (shoulder_yaw) + 肘关节 (elbow_pitch) + 腕关节 (wrist_pitch)，全部是 hinge 关节，nq = nv = 3。

---

## 1. 通过 qpos 设置初始姿态

```python
data.qpos[:] = np.radians([45, -60, 30])
mujoco.mj_forward(model, data)

ee_pos = data.site_xpos[ee_id]
```

流程：**设置 qpos → 调用 `mj_forward` → 读取派生量**。

`mj_forward` 根据 qpos 计算出所有派生量（世界位置 `xpos`、旋转矩阵 `xmat`、站点位置 `site_xpos` 等），不调用它则派生量不会更新。

---

## 2. 按名称读取/修改单个关节

直接用索引操作 `data.qpos` 容易出错，按名称更安全：

```python
def get_joint_qpos(model, data, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    nq = {0: 7, 1: 4, 2: 1, 3: 1}[model.jnt_type[jid]]
    return data.qpos[adr:adr + nq].copy()

def set_joint_qpos(model, data, joint_name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = model.jnt_qposadr[jid]
    nq = {0: 7, 1: 4, 2: 1, 3: 1}[model.jnt_type[jid]]
    data.qpos[adr:adr + nq] = value
```

### 关键映射表

| 关节类型 | `jnt_type` 值 | qpos 维度 (nq) | qvel 维度 (nv) |
| :------: | :---: | :---: | :---: |
| free     | 0 | 7 (pos + quat) | 6 (lin + ang) |
| ball     | 1 | 4 (quat) | 3 (ang) |
| slide    | 2 | 1 | 1 |
| hinge    | 3 | 1 | 1 |

---

## 3. mj_integratePos — 正确的 qpos 积分

### 为什么不能用 `qpos += qvel * dt`？

| 情况 | 问题 |
| :--- | :--- |
| free 关节 | qpos 有 7 维，qvel 有 6 维，**维度不匹配** |
| ball 关节 | qpos 有 4 维（四元数），qvel 有 3 维（角速度），同样不匹配 |
| hinge/slide | 维度相同，简单加法结果正确 |

### 正确做法

```python
qpos_new = qpos.copy()
mujoco.mj_integratePos(model, qpos_new, qvel, dt)
```

`mj_integratePos` 的内部处理：
- **位置部分**：`pos += vel_lin * dt`（普通加法）
- **四元数部分**：用指数映射（exponential map）积分角速度，保证结果仍然是单位四元数
- **hinge/slide 部分**：`q += qvel * dt`（和简单加法一样）

> 只要模型中有 free 或 ball 关节，就**必须**用 `mj_integratePos`。

---

## 4. mj_differentiatePos — 计算 qpos 差值

`mj_integratePos` 的逆操作：给定两个 qpos，计算它们在 qvel 空间中的差。

```python
qvel_diff = np.zeros(model.nv)
mujoco.mj_differentiatePos(model, qvel_diff, dt, qpos1, qpos2)
```

### 参数说明

| 参数 | 含义 |
| :--- | :--- |
| `model` | 模型 |
| `qvel_diff` | 输出，nv 维差值向量 |
| `dt` | 时间步长（返回的是 `差值 / dt`，通常设为 1.0 得到原始差值） |
| `qpos1` | 起始 qpos |
| `qpos2` | 目标 qpos |

### 与简单减法的关系

```
对于纯 hinge/slide: mj_differentiatePos ≡ (qpos2 - qpos1) / dt
对于 free/ball:     简单减法无意义，必须用此函数
```

---

## 5. 正运动学: qpos → 笛卡尔空间

正运动学（Forward Kinematics）：给定关节角度，计算末端在世界坐标系中的位置。

```python
data.qpos[:] = target_qpos
mujoco.mj_forward(model, data)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
ee_pos = data.site_xpos[site_id]  # 世界坐标 [x, y, z]
```

MuJoCo 中不需要自己写 DH 参数或变换矩阵链，`mj_forward` 帮你算好一切。

### 工作空间探索

通过随机采样关节角度，可以估算机械臂的可达工作空间：

```python
def compute_workspace_bounds(model, data, n_samples=5000):
    positions = []
    for _ in range(n_samples):
        for j in range(model.njnt):
            adr = model.jnt_qposadr[j]
            low, high = model.jnt_range[j]
            data.qpos[adr] = np.random.uniform(low, high)
        mujoco.mj_forward(model, data)
        positions.append(data.site_xpos[site_id].copy())
    positions = np.array(positions)
    return {"min": positions.min(axis=0), "max": positions.max(axis=0)}
```

---

## 6. 保存和加载 qpos 快照

### 快照内容

```json
{
  "nq": 3,
  "nv": 3,
  "qpos": [0.785, -1.047, 0.524],
  "qvel": [0.1, -0.2, 0.3],
  "time": 1.5,
  "joints": {
    "shoulder_yaw": {"type": "hinge", "qpos_adr": 0, "nq": 1, "value": [0.785]},
    "elbow_pitch":  {"type": "hinge", "qpos_adr": 1, "nq": 1, "value": [-1.047]},
    "wrist_pitch":  {"type": "hinge", "qpos_adr": 2, "nq": 1, "value": [0.524]}
  },
  "metadata": {"task": "pick_and_place", "step": 42}
}
```

### 保存/加载

```python
save_qpos_snapshot(model, data, "snapshot.json", metadata={"task": "demo"})

mujoco.mj_resetData(model, data)
load_qpos_snapshot(model, data, "snapshot.json")
# data.qpos, data.qvel, data.time 均已恢复
```

### 设计要点

- 保存 `nq`/`nv` 用于加载时校验模型兼容性
- 保存关节详情方便人工检查
- `metadata` 字段存储任务相关信息（任务名、步数等）

---

## 7. qpos 插值 — 生成平滑轨迹

### 纯 hinge/slide 关节：直接线性插值

```python
qpos_t = (1 - t) * qpos_start + t * qpos_end  # t ∈ [0, 1]
```

### 通用插值（支持 free/ball 关节）

利用 `mj_differentiatePos` + `mj_integratePos` 实现：

```python
def interpolate_qpos(model, qpos1, qpos2, t):
    qvel_diff = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, qvel_diff, 1.0, qpos1, qpos2)
    result = qpos1.copy()
    mujoco.mj_integratePos(model, result, qvel_diff, t)
    return result
```

原理：
1. `differentiatePos` 算出从 qpos1 到 qpos2 在 qvel 空间的"方向"
2. `integratePos` 沿这个方向走 t 步（t=0 → qpos1，t=1 → qpos2）
3. 四元数部分自动做了球面插值，无需手动处理

> **注意**：关节空间的线性插值 ≠ 笛卡尔空间的直线路径，末端走的是曲线。

---

## 8. 关节限位检查与裁剪

### 不同关节类型的限位语义

| 关节类型 | `jnt_range` 含义 | 检查方式 |
| :------: | :--------------- | :------- |
| hinge | `[θ_min, θ_max]`，角度范围 | 直接比较 `qpos[adr]` |
| slide | `[d_min, d_max]`，位移范围 | 直接比较 `qpos[adr]` |
| ball | `[0, max_angle]`，**离参考姿态的最大偏转角** | 计算四元数偏转角 `2·arccos(|qw|)` |
| free | 不支持限位 | `jnt_limited` 为 False，自动跳过 |

### 检查

```python
def check_joint_limits(model, qpos):
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        jtype = model.jnt_type[j]
        adr = model.jnt_qposadr[j]
        low, high = model.jnt_range[j]

        if jtype in (2, 3):  # slide or hinge
            val = qpos[adr]
            if val < low or val > high:
                print(f"超限! 值={val}, 范围=[{low}, {high}]")

        elif jtype == 1:  # ball — 限位是最大偏转角
            q = qpos[adr:adr + 4]
            angle = 2.0 * np.arccos(np.clip(np.abs(q[0]), 0, 1))
            if angle > high:
                print(f"超限! 偏转角={angle}, 最大={high}")
```

> ball 关节的 `q[0]`（即 qw）在无偏转时为 1，偏转越大越接近 0，`2·arccos(|qw|)` 即为偏转角。

### 裁剪

```python
def clip_to_joint_limits(model, qpos):
    qpos = qpos.copy()
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        jtype = model.jnt_type[j]
        adr = model.jnt_qposadr[j]
        low, high = model.jnt_range[j]

        if jtype in (2, 3):  # slide or hinge
            qpos[adr] = np.clip(qpos[adr], low, high)

        elif jtype == 1:  # ball — 将四元数裁剪到最大偏转角
            q = qpos[adr:adr + 4]
            angle = 2.0 * np.arccos(np.clip(np.abs(q[0]), 0, 1))
            if angle > high:
                axis = q[1:4]
                axis /= np.linalg.norm(axis)  # 保持旋转轴不变
                half = high / 2.0
                q[0] = np.cos(half) * np.sign(q[0])
                q[1:4] = axis * np.sin(half)  # 缩放到最大允许角度
                qpos[adr:adr + 4] = q
    return qpos
```

> ball 裁剪原理：保持旋转轴方向不变，仅将偏转角缩小到 `max_angle`。

### 示例

```
检查 [200°, -150°, 30°]:
  ❌ shoulder_yaw: 200.00° 超出 [-180.00°, 180.00°]
  ❌ elbow_pitch: -150.00° 超出 [-134.64°, 134.64°]
  ✅ wrist_pitch: 30.00° 在 [-114.59°, 114.59°] 范围内

裁剪后: [180.00°, -134.64°, 30.00°]
```

---

## 9. 数据平台实用工具

### 轨迹验证

`validate_qpos_trajectory` 检查一条 qpos 轨迹的数据质量：

| 检查项 | 说明 |
| :--- | :--- |
| 维度匹配 | `nq` 是否与模型一致 |
| 关节限位 | 每帧的 qpos 是否在 `jnt_range` 内 |
| 四元数归一化 | free/ball 关节的四元数 \|q\| ≈ 1 |
| 速度跳变 | 相邻帧差值是否超过中位数的 5 倍 |

```python
report = validate_qpos_trajectory(model, trajectory)
# report = {
#   "n_frames": 11,
#   "nq_match": True,
#   "limit_violations": 0,
#   "quat_norm_errors": 0,
#   "velocity_jumps": 0,
# }
```

---

## 总结

```
┌──────────────────────────────────────────────────────────────┐
│                    qpos 操作速查                               │
│                                                              │
│  读取关节:  data.qpos[model.jnt_qposadr[jid]:adr+nq]       │
│  修改关节:  设置 qpos 后调用 mj_forward 更新派生量           │
│                                                              │
│  正确积分:  mj_integratePos(model, qpos, qvel, dt)          │
│  计算差值:  mj_differentiatePos(model, diff, dt, q1, q2)    │
│                                                              │
│  正运动学:  设 qpos → mj_forward → 读 site_xpos             │
│  快照保存:  qpos + qvel + time + 关节元信息 → JSON          │
│                                                              │
│  插值:      hinge/slide 线性插值即可                          │
│             free/ball 用 differentiatePos + integratePos     │
│                                                              │
│  限位检查:  model.jnt_limited + model.jnt_range              │
└──────────────────────────────────────────────────────────────┘
```
