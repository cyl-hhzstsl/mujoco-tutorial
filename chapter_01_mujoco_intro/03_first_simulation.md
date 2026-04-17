# 第 1 章 · 03 — 第一个完整仿真

> **目标**: 加载单摆模型，施加控制，录制数据，画图分析。完整的"加载 → 控制 → 录制 → 分析"流程。

---

## 1. 加载模型

```python
model = mujoco.MjModel.from_xml_path("models/pendulum.xml")
data = mujoco.MjData(model)
```

---

## 2. 设置初始条件

```python
data.qpos[0] = np.deg2rad(45)   # 初始角度 45°
mujoco.mj_forward(model, data)
```

---

## 3. 仿真循环 + 数据录制

三个实验对比不同控制策略：

| 实验 | ctrl 值 | 效果 |
| :--- | :--- | :--- |
| 无控制（自由摆动） | 0 | 因阻尼逐渐静止 |
| 恒定力矩 | 1.0 | 单向偏转 |
| 正弦控制 | `3.0 * sin(2πt)` | 可能产生共振 |

```python
for step in range(NUM_STEPS):
    data.ctrl[0] = ctrl_fn(step, data)

    record["time"][step] = data.time
    record["angle"][step] = data.qpos[0]
    record["velocity"][step] = data.qvel[0]
    record["energy"][step] = data.energy[0] + data.energy[1]

    mujoco.mj_step(model, data)
```

> 注意：录制数据在 `mj_step` 之前，这样记录的是步进前的状态。

---

## 4. 可视化

生成 `simulation_result.png`，三行子图：
- 角度 vs 时间
- 角速度 vs 时间
- 控制力矩 vs 时间

---

## 5. 数据分析

- 角度范围（min/max）
- 最大角速度
- 振荡频率估算：通过零点交叉（`np.diff(np.sign(angle))`）计算周期
