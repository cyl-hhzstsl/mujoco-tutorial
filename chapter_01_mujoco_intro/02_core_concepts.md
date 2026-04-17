# 第 1 章 · 02 — MuJoCo 核心概念

> **目标**: 深入理解 MjModel 和 MjData 的关系，掌握仿真循环。

## Java 类比

| MuJoCo | Java | 说明 |
| :--- | :--- | :--- |
| MjModel | Class 定义 | 编译后不可变的结构 |
| MjData | Object 实例 | 运行时可变的状态 |
| mj_step | tick() / update() | 推进一步仿真 |
| mj_forward | 只计算不推进 | 用于渲染/查询 |

---

## 1. MjModel — 不可变的模型结构

```python
model = mujoco.MjModel.from_xml_path("ball_drop.xml")
```

### 关键属性

| 属性 | 含义 |
| :--- | :--- |
| `model.nq` | qpos 维度 |
| `model.nv` | qvel 维度 |
| `model.nu` | ctrl 控制输入维度 |
| `model.nbody` | 刚体数量（含 world） |
| `model.njnt` | 关节数量 |
| `model.ngeom` | 几何体数量 |
| `model.opt.timestep` | 时间步长 |
| `model.opt.gravity` | 重力向量 |

### 遍历 Body / Joint / Geom

```python
for i in range(model.nbody):
    name = model.body(i).name
    parent = model.body_parentid[i]

for i in range(model.njnt):
    jtype = model.jnt_type[i]    # 0=free, 1=ball, 2=slide, 3=hinge
    adr = model.jnt_qposadr[i]   # 在 qpos 中的起始地址
```

---

## 2. MjData — 可变的运行时状态

```python
data = mujoco.MjData(model)
```

| 属性 | 说明 | 可写 |
| :--- | :--- | :---: |
| `data.time` | 仿真时间 | 是 |
| `data.qpos` | 广义位置 | 是 |
| `data.qvel` | 广义速度 | 是 |
| `data.ctrl` | 控制输入 | 是 |
| `data.xpos` | Body 世界位置 | 否（派生量） |
| `data.xmat` | Body 旋转矩阵 | 否（派生量） |

---

## 3. 仿真循环 — mj_step

```python
for step in range(500):
    mujoco.mj_step(model, data)
    # data.time, data.qpos, data.qvel 都会更新
```

---

## 4. mj_step vs mj_forward

| | mj_forward | mj_step |
| :--- | :--- | :--- |
| 做什么 | 只计算派生量 | 完整仿真一步 |
| 改变 qpos | 不改 | 会改 |
| 改变 time | 不改 | time += dt |
| 用途 | 正运动学、读状态 | 物理仿真 |

---

## 5. 重置状态

```python
mujoco.mj_resetData(model, data)       # 重置到默认
data.qpos[2] = 5.0                      # 手动设置
data.qpos[:] = model.qpos0              # 从模型默认值恢复
mujoco.mj_forward(model, data)          # 别忘了刷新派生量
```
