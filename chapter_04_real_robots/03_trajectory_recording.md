# 第 4 章 · 03 — 仿真轨迹录制

> **目标**: 在仿真中运行机器人，录制完整的轨迹数据，保存为多种格式，对比各格式的优劣。

## 核心知识点

1. 控制信号的生成（正弦波扫频）
2. 仿真循环中的数据采集
3. 多种序列化格式（.npy, .npz, .pkl, .json）的优劣
4. 数据完整性验证

---

## 1. 使用的模型

6 DOF 固定基座机械臂，包含传感器：

```xml
<actuator>
  <position name="act1" joint="joint1" kp="500" kv="50"/>
  ...
</actuator>

<sensor>
  <jointpos name="jpos1" joint="joint1"/>   <!-- 关节位置传感器 -->
  <jointvel name="jvel1" joint="joint1"/>   <!-- 关节速度传感器 -->
  ...
</sensor>
```

| 属性 | 值 |
| :--- | --: |
| nq (qpos 维度) | 6 |
| nv (qvel 维度) | 6 |
| nu (执行器数) | 6 |
| nsensor (传感器数) | 12 |
| timestep | 0.002 s (500 Hz) |

---

## 2. 控制信号生成

脚本使用**正弦波扫频**为每个关节生成不同频率和幅度的控制信号：

```python
def generate_control_signal(t, nu):
    ctrl = np.zeros(nu)
    for i in range(nu):
        freq  = 0.3 + i * 0.15      # 递增频率
        amp   = 0.5 / (1 + i * 0.3) # 递减幅度（远端关节幅度小）
        phase = i * np.pi / 4        # 各关节相位差
        ctrl[i] = amp * np.sin(2 * np.pi * freq * t + phase)
    return ctrl
```

### 设计思路

| 参数 | 策略 | 目的 |
| :--- | :--- | :--- |
| 频率 | 越远端越高 | 激发不同动态特性 |
| 幅度 | 越远端越小 | 避免远端关节过大摆动 |
| 相位差 | π/4 递增 | 各关节不同步运动，覆盖更多构型 |

---

## 3. 仿真循环与数据采集

### 核心循环

```python
for step in range(n_steps):
    data.ctrl[:] = generate_control_signal(data.time, model.nu)
    mujoco.mj_step(model, data)           # 物理仿真前进一步

    if step % record_every == 0:          # 降采样
        times[idx]    = data.time
        qpos_log[idx] = data.qpos.copy()  # 必须 copy！
        qvel_log[idx] = data.qvel.copy()
        ctrl_log[idx] = ctrl.copy()
        sensor_log[idx] = data.sensordata.copy()
        ee_pos_log[idx] = data.site_xpos[ee_site_id].copy()
```

### 关键细节

1. **必须 `.copy()`**：`data.qpos` 是 MuJoCo 内部缓冲区的视图，不 copy 的话所有帧都指向同一内存
2. **降采样**：仿真在 500 Hz 运行，但每 10 步记录一次 → 50 Hz 录制频率
3. **末端执行器位置**：通过 `data.site_xpos` 获取，需要先用 `mj_name2id` 查找 site ID

### 录制参数

```
仿真频率    = 500 Hz (dt = 0.002s)
录制频率    = 50 Hz  (每 10 步记录)
仿真时长    = 5 s
总仿真步数  = 2500
录制帧数    = 250
```

---

## 4. 轨迹数据结构

完整的轨迹数据是一个字典：

```python
trajectory = {
    "time":        np.ndarray (T,),          # 时间戳
    "qpos":        np.ndarray (T, nq),       # 广义坐标
    "qvel":        np.ndarray (T, nv),       # 广义速度
    "ctrl":        np.ndarray (T, nu),       # 控制信号
    "sensor_data": np.ndarray (T, nsensor),  # 传感器数据
    "ee_pos":      np.ndarray (T, 3),        # 末端执行器位置 (x,y,z)
    "metadata": {
        "model_name":     "6dof_arm",
        "nq":             6,
        "nv":             6,
        "nu":             6,
        "dt":             0.002,             # 仿真时间步长
        "record_dt":      0.02,              # 录制时间步长
        "duration":       5.0,
        "n_frames":       250,
        "record_hz":      50.0,
        "joint_names":    ["joint1", ...],
        "actuator_names": ["act1", ...],
    },
}
```

> **metadata 的重要性**：没有 metadata 的轨迹数据是"裸数据"，无法被正确解析。metadata 应至少包含 nq、nv、nu 和采样率。

---

## 5. 多种保存格式

### 5.1 NumPy `.npy` — 单数组

```python
np.save("trajectory_qpos.npy", trajectory["qpos"])
loaded = np.load("trajectory_qpos.npy")
```

- 只能存**一个**数组
- 读写极快，零序列化开销
- 不支持 metadata

### 5.2 NumPy `.npz` — 多数组压缩包

```python
np.savez_compressed("trajectory.npz",
    time=trajectory["time"],
    qpos=trajectory["qpos"],
    qvel=trajectory["qvel"],
    ctrl=trajectory["ctrl"],
)

with np.load("trajectory.npz") as npz:
    qpos = npz["qpos"]   # 支持懒加载
```

- 多个数组打包成一个文件
- 支持压缩 (`savez_compressed`)
- 支持懒加载（按需读取某个数组）
- 不支持嵌套字典、任意 Python 对象

### 5.3 Pickle `.pkl` — 完整 Python 对象

```python
import pickle
with open("trajectory.pkl", "wb") as f:
    pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("trajectory.pkl", "rb") as f:
    loaded = pickle.load(f)
```

- 可保存任意 Python 对象（包括嵌套字典、metadata）
- 仅 Python 可读，跨语言不友好
- 有安全风险（反序列化可执行任意代码）

### 5.4 JSON — 元数据 / 配置

```python
import json
json_data = {
    "metadata": trajectory["metadata"],
    "statistics": {
        "qpos_mean": trajectory["qpos"].mean(axis=0).tolist(),
        "qpos_std":  trajectory["qpos"].std(axis=0).tolist(),
        ...
    },
}
with open("trajectory_meta.json", "w") as f:
    json.dump(json_data, f, indent=2)
```

- 人类可读，跨语言通用
- 不适合存储大型数组（体积大、精度丢失）
- 适合存 metadata 和统计摘要

### 格式对比总结

| 格式 | 适合场景 | 优点 | 缺点 |
| :--- | :------- | :--- | :--- |
| `.npy` | 单个数组快速读写 | 极快，零开销 | 只能存单个数组 |
| `.npz` | 多个数组打包 | 可压缩，懒加载 | 不支持任意对象 |
| `.pkl` | 完整 Python 对象 | 灵活，保留类型 | 仅 Python，安全风险 |
| `.json` | 元数据/配置 | 人类可读，跨语言 | 不适合大数组 |
| `.hdf5` | 大规模数据集（第 5 章） | 分层存储，部分读取 | 需要 h5py 库 |

---

## 6. 数据完整性验证

保存后应立即重新加载并验证：

```python
loaded_qpos = np.load("trajectory_qpos.npy")
assert np.allclose(loaded_qpos, trajectory["qpos"])  # 数值一致性

with np.load("trajectory.npz") as npz:
    assert np.allclose(npz["qpos"], trajectory["qpos"])

with open("trajectory.pkl", "rb") as f:
    loaded = pickle.load(f)
    assert np.allclose(loaded["qpos"], trajectory["qpos"])
    assert loaded["metadata"]["model_name"] == "6dof_arm"
```

验证要点：
- 数组形状 (shape) 一致
- 数值精度 (`np.allclose`) 通过
- metadata 完整且正确

---

## 7. 轨迹统计摘要

脚本会打印每个关节的统计信息：

```
    qpos 统计 (shape (250, 6)):
      joint1: mean=  0.015  std= 0.254  range=[-0.456,  0.474]
      joint2: mean=  0.008  std= 0.190  range=[-0.348,  0.361]
      ...
```

这些统计有助于快速判断轨迹质量：
- **均值偏离 0**：可能存在系统偏移
- **标准差过小**：关节几乎没动
- **范围超限**：可能违反关节限位

---

## 8. 总结

```
┌───────────────────────────────────────────────────────────────┐
│              仿真轨迹录制 — 核心流程                            │
│                                                               │
│  录制流程:                                                     │
│    1. 加载模型，创建 MjData                                     │
│    2. 预分配 numpy 数组（避免动态 append）                       │
│    3. 仿真循环: ctrl → mj_step → 采集 qpos/qvel/ctrl           │
│    4. 降采样记录（仿真 500Hz → 记录 50Hz）                      │
│    5. 所有数据必须 .copy()！                                    │
│                                                               │
│  保存策略:                                                     │
│    • 数据数组 → .npz (压缩，多数组)                             │
│    • 完整轨迹 → .pkl (含 metadata)                              │
│    • 元数据   → .json (人类可读)                                │
│    • 大规模   → .hdf5 (第 5 章)                                 │
│                                                               │
│  最佳实践:                                                     │
│    • 始终保存 metadata（nq, nv, nu, 采样率, 关节名）             │
│    • 保存后立即验证数据完整性                                    │
│    • 预分配数组而非 list.append                                  │
└───────────────────────────────────────────────────────────────┘
```
