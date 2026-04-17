# 第 4 章 · 04 — 轨迹回放

> **目标**: 加载已录制的轨迹数据，在仿真中逐帧回放，掌握格式自动检测、速度控制和统计分析。

## 核心知识点

1. 从 `.pkl` 文件加载完整轨迹
2. 通过设置 `qpos` + `mj_forward` 实现回放（无需物理仿真）
3. 轨迹数据格式的自动检测
4. 回放速度控制
5. 轨迹质量统计

---

## 1. 轨迹加载与格式自动检测

### 支持的格式

```python
def load_trajectory(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pkl":    # Pickle → 最完整
    elif ext == ".npz":  # NumPy 压缩包
    elif ext == ".npy":  # 纯数组
    elif ext == ".json": # 元数据
```

### 归一化为标准格式

无论输入什么格式，都归一化为统一结构：

```python
{
    "time":     ndarray (T,),           # 必须有（无则自动生成）
    "qpos":     ndarray (T, nq),        # 必须有
    "qvel":     ndarray (T, nv) | None, # 可选
    "ctrl":     ndarray (T, nu) | None, # 可选
    "ee_pos":   ndarray (T, 3)  | None, # 可选
    "metadata": dict | None,            # 可选但推荐
}
```

### 自动检测三种数据结构

| 结构 | 检测方式 | 示例 |
| :--- | :------- | :--- |
| 标准字典 | `isinstance(data, dict)` 且有 `"qpos"` 键 | `.pkl` / `.npz` 加载结果 |
| 帧列表 | `isinstance(data, list)` 且元素是 dict | `[{"qpos": [...], "time": 0.0}, ...]` |
| 纯数组 | `isinstance(data, ndarray)` | `.npy` 加载的 qpos 数组 |

### 自动补全 time

如果数据中没有 time 字段，会自动生成：

```python
if result["time"] is None:
    dt = metadata.get("record_dt", 0.02)  # 默认 50 Hz
    result["time"] = np.arange(n_frames) * dt
```

---

## 2. 回放原理：qpos 设置 vs 物理仿真

### 方式 A：设置 qpos + mj_forward（推荐）

```python
for i in range(n_frames):
    data.qpos[:] = qpos_traj[i]      # 直接设置关节位置
    data.qvel[:] = qvel_traj[i]      # 设置速度（可选）
    mujoco.mj_forward(model, data)   # 只做运动学计算，不做物理仿真
```

- **精确重现**：末端位置与录制时完全一致
- **无累积误差**：每帧独立设置，不依赖前一帧
- **适用场景**：回放已录制的轨迹数据

### 方式 B：设置 ctrl + mj_step（物理仿真）

```python
for i in range(n_frames):
    data.ctrl[:] = ctrl_traj[i]      # 设置控制信号
    mujoco.mj_step(model, data)      # 执行物理仿真
```

- **有累积误差**：仿真结果可能偏离录制数据
- **适用场景**：验证控制策略是否可复现

### mj_forward vs mj_step

| | `mj_forward` | `mj_step` |
| :--- | :--- | :--- |
| 功能 | 只做正运动学 + 碰撞检测 | 完整物理仿真（含力学积分） |
| 修改 qpos | 否（保持你设置的值） | 是（按物理规律更新） |
| 用途 | 回放、可视化 | 仿真、控制 |
| 误差 | 零 | 有累积误差 |

---

## 3. 速度控制

```python
if speed > 0 and i < n_frames - 1:
    dt_data = time_traj[i + 1] - time_traj[i]
    sleep_time = dt_data / speed
    if sleep_time > 0.001:
        time.sleep(sleep_time)
```

| 速度 | sleep_time | 效果 |
| ---: | ---------: | :--- |
| ∞ | 0 | 尽快回放（计算速度） |
| 2.0x | dt/2 | 两倍速 |
| 1.0x | dt | 实时 |
| 0.5x | dt×2 | 慢放 |

### 多速度回放测试

脚本会以 4 种速度回放前 100 帧，打印实际达到的速度：

```
  模式          目标速度  墙钟时间    实际速度
  ──────────   ────────  ──────────  ──────────
  最快 (∞)           ∞     0.005s     400.0x
  2x 倍速        2.0x     1.001s       2.0x
  1x 实时        1.0x     2.001s       1.0x
  0.5x 慢放      0.5x     4.001s       0.5x
```

---

## 4. 回放精度验证

回放后对比末端执行器位置：

```python
error = np.sqrt(np.sum(
    (replay_ee[:min_len] - orig_ee[:min_len]) ** 2, axis=1
))
print(f"平均误差: {error.mean():.6f} m")
print(f"最大误差: {error.max():.6f} m")
```

使用 `qpos + mj_forward` 方式回放时，误差应为 0（或浮点精度级别 < 1e-6 m）。

---

## 5. 轨迹统计分析

### 5.1 基本信息

```
总帧数:      250
时间跨度:    4.980 s
平均帧间隔:  0.020000 s (50.0 Hz)
帧间隔标准差: 0.000000 s             ← 均匀采样
```

### 5.2 各关节 qpos 统计

```
  关节          均值      标准差      最小      最大    变化范围
  ────────     ──────    ──────    ──────    ──────    ──────
  joint1        0.0150    0.2540   -0.4560    0.4740    0.9300
  joint2        0.0080    0.1900   -0.3480    0.3610    0.7090
  ...
```

### 5.3 数值差分速度

通过 \(\dot{q} \approx \frac{q_{t+1} - q_t}{\Delta t}\) 计算每个关节的近似速度：

```python
dq = np.diff(qpos, axis=0)                    # (T-1, nq)
dt = np.diff(time_arr)[:, None]                # (T-1, 1)
velocity = dq / dt                             # (T-1, nq)
smoothness = np.abs(np.diff(velocity, axis=0)).mean()  # 加速度 ≈ 平滑度
```

**平滑度**越低，轨迹越平滑（加速度变化小）。

### 5.4 关节相关性

```python
corr = np.corrcoef(qpos.T)  # (nq, nq) 相关性矩阵
```

如果两个关节的 |r| > 0.7，说明它们高度相关——可能存在耦合运动或冗余。

### 5.5 末端执行器统计

```
总行程距离:  1.2345 m
工作空间 X:  [-0.15, 0.18] m
工作空间 Y:  [-0.12, 0.14] m
工作空间 Z:  [ 0.35, 0.85] m
```

---

## 6. 可视化回放

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(n_frames):
        if not viewer.is_running():
            break
        data.qpos[:] = qpos_traj[i]
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(dt / speed)
```

- 使用 `launch_passive` 启动非阻塞 viewer
- 每帧设置 qpos → mj_forward → viewer.sync()
- 支持在 macOS / Linux (有 DISPLAY) / Windows 上运行
- headless 服务器上自动跳过

---

## 7. 本章总结

```
┌───────────────────────────────────────────────────────────────┐
│              轨迹回放 — 核心要点                                │
│                                                               │
│  回放方式:                                                     │
│    ✓ qpos + mj_forward → 精确回放，零误差                      │
│    △ ctrl + mj_step    → 物理仿真，有累积误差                   │
│                                                               │
│  格式检测:                                                     │
│    • 自动识别 .pkl / .npz / .npy / .json                       │
│    • 归一化为标准字典格式                                       │
│    • 缺失 time 时自动生成                                      │
│                                                               │
│  速度控制:                                                     │
│    • sleep_time = dt_data / speed                              │
│    • speed=∞ 最快, speed=1 实时, speed=0.5 慢放                 │
│                                                               │
│  统计分析:                                                     │
│    • 各关节 mean/std/min/max                                   │
│    • 数值差分速度与平滑度                                       │
│    • 关节相关性矩阵                                            │
│    • 末端执行器工作空间范围                                      │
│                                                               │
│  第 4 章完整回顾:                                               │
│    01 → 加载多种真实机器人，分析关节结构                          │
│    02 → 对比固定基座 vs 浮动基座                                 │
│    03 → 仿真录制轨迹，多格式保存                                 │
│    04 → 加载回放轨迹，速度控制，统计分析                          │
└───────────────────────────────────────────────────────────────┘
```
