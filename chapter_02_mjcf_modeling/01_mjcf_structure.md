# 第 2 章 · 01 — MJCF 结构全解析

> **目标**: 加载一个 MJCF 模型，系统地打印出每一个组成部分，深入理解 MJCF XML 各标签在编译后模型中的对应关系。

---

## 1. 基本信息

加载模型后可查看的核心属性：

| 属性 | 含义 |
| :--- | :--- |
| `model.nq` | qpos 维度 |
| `model.nv` | qvel 维度（自由度数） |
| `model.nbody` | 刚体数（含 world body） |
| `model.njnt` | 关节数 |
| `model.ngeom` | 几何体数 |
| `model.nu` | 执行器数 |
| `model.nsensordata` | 传感器数据总维度 |

---

## 2. 仿真选项 (option)

```python
model.opt.timestep       # 时间步长
model.opt.gravity        # 重力向量 [0, 0, -9.81]
model.opt.integrator     # 积分器: 0=Euler, 1=RK4, 2=implicit
model.opt.cone           # 接触锥: 0=pyramidal, 1=elliptic
```

---

## 3. 刚体层级树

MuJoCo 的刚体组成一棵树，根节点是 world body (id=0)。

```
🌐 world
  ├─ 📦 base       pos=[0, 0, 0.1]
  │   ├─ 📦 link1  pos=[0, 0, 0.05]
  │   │   🔗 关节: shoulder_yaw  type=hinge  axis=[0, 0, 1]
  │   │   🔵 几何: capsule
  │   │   ├─ 📦 link2  ...
```

遍历方法：

```python
for child_id in range(model.nbody):
    if model.body_parentid[child_id] == parent_id:
        # child_id 是 parent_id 的子体
```

---

## 4. 关节详情

```python
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    jtype = model.jnt_type[j]        # 0=free, 1=ball, 2=slide, 3=hinge
    axis = model.jnt_axis[j]          # 旋转/滑动轴
    limited = model.jnt_limited[j]    # 是否有限位
    jnt_range = model.jnt_range[j]    # [下限, 上限]
    damping = model.dof_damping[j]    # 阻尼系数
```

---

## 5. 几何体详情

```python
for g in range(model.ngeom):
    model.geom_type[g]        # 0=plane, 2=sphere, 3=capsule, 6=box, ...
    model.geom_size[g]        # 尺寸参数
    model.geom_bodyid[g]      # 所属 body
    model.geom_contype[g]     # 碰撞类型掩码
    model.geom_conaffinity[g] # 碰撞亲和掩码
```

---

## 6. 执行器详情

通过 `gainprm` 和 `biasprm` 判断执行器类型：

| gain[0] ≠ 0 且... | 类型 |
| :--- | :--- |
| bias[1] ≠ 0 | 位置伺服 (position) |
| bias[2] ≠ 0 | 速度伺服 (velocity) |
| 其他 | 力矩电机 (motor) |

---

## 7. 传感器详情

```python
for s in range(model.nsensor):
    stype = model.sensor_type[s]     # 传感器类型 ID
    dim = model.sensor_dim[s]        # 数据维度
    adr = model.sensor_adr[s]        # 在 sensordata 中的起始地址
    values = data.sensordata[adr:adr + dim]
```

常见传感器类型：jointpos, jointvel, actuatorfrc, accelerometer, gyro, framepos, framequat, touch

---

## 8. 当前状态快照

```python
data.time          # 仿真时间
data.qpos          # 广义位置
data.qvel          # 广义速度
data.ctrl          # 控制输入
data.site_xpos[id] # 站点世界位置（需先调用 mj_forward）
```
