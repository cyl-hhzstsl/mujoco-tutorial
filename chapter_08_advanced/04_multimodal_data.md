# 第 8 章 · 04 — 多模态机器人数据处理

> **目标**: 理解如何同步采集、对齐、存储来自不同传感器的数据。

## 核心知识点

1. MuJoCo 离屏渲染
2. MultiModalRecorder: 多模态数据采集
3. 不同频率数据的对齐策略
4. HDF5 多模态存储

---

## 1. 数据模态

| 模态 | 典型频率 | 数据形状 | 特点 |
| :--- | -------: | :------- | :--- |
| qpos / qvel | 1000 Hz | (nq,) / (nv,) | 小、连续 |
| 图像 (RGB) | 30 Hz | (H, W, 3) uint8 | 大、离散 |
| 末端位置 | 200 Hz | (3,) | 小、由 FK 导出 |
| 传感器数据 | 1000 Hz | (nsensordata,) | 力、加速度等 |
| 控制信号 | 1000 Hz | (nu,) | 执行器输出 |

---

## 2. 离屏渲染 (ImageRenderer)

```python
renderer = ImageRenderer(model, width=320, height=240)
img = renderer.render(data, camera_name="overhead")  # (240, 320, 3) uint8
```

两种模式：
- **真实渲染**: MuJoCo OpenGL 离屏渲染（需要 GPU/EGL）
- **Mock 渲染**: 基于 qpos 生成伪图像（无 GPU 降级方案）

---

## 3. MultiModalRecorder

### 配置

```python
recorder = MultiModalRecorder(sim_dt=0.001)
recorder \
    .add_modality("qpos", frequency=1000, shape=(2,)) \
    .add_modality("image", frequency=30, shape=(240, 320, 3), dtype="uint8") \
    .add_modality("ee_pos", frequency=200, shape=(3,))
```

### 录制流程

```python
recorder.start_episode()
for step in range(n_steps):
    if recorder.should_record("qpos", step):
        recorder.record("qpos", step, data.qpos)
    if recorder.should_record("image", step):
        recorder.record("image", step, renderer.render(data))
episode = recorder.end_episode()
```

### 采样间隔

`should_record()` 根据频率自动计算：
- qpos 1000Hz / sim_dt 0.001 → 每步采样
- image 30Hz / sim_dt 0.001 → 每 33 步采样
- ee_pos 200Hz / sim_dt 0.001 → 每 5 步采样

---

## 4. 数据对齐 (DataAligner)

不同频率的数据需要对齐到统一时间轴：

| 策略 | 方法 | 适用场景 |
| :--- | :--- | :------- |
| 最近邻 | `align_nearest` | 图像等离散数据 |
| 线性插值 | `align_interpolate` | 连续数值（qpos、力） |
| 零阶保持 | `align_zero_order_hold` | 控制信号（步进式） |

```python
ee_at_img_times = DataAligner.align_interpolate(
    img_timestamps, ee_timestamps, ee_data
)
```

---

## 5. HDF5 存储

### 文件结构

```
/metadata
    episode_id, sim_dt, save_time
/qpos
    data (T, nq), timestamps (T,)
/image_overhead
    data (T_img, H, W, 3) [gzip 压缩], timestamps (T_img,)
/ee_pos
    data (T_ee, 3), timestamps (T_ee,)
```

图像数据使用 gzip 压缩减小文件大小。不同模态的帧数不同（频率不同），各自带独立时间戳。

### 降级方案

h5py 不可用时自动切换到 `np.savez_compressed`。

---

## 6. 数据量估算

3 秒录制，sim_dt=0.001：

| 模态 | 帧数 | 大小估算 |
| :--- | ---: | -------: |
| qpos (2 joints, 1kHz) | 3000 | ~47 KB |
| image (320×240, 30Hz) | 90 | ~21 MB |
| ee_pos (3D, 200Hz) | 600 | ~7 KB |
| sensor (8 channels, 1kHz) | 3000 | ~94 KB |

图像数据占绝对主导（>99%）。

---

## 7. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              多模态数据 — 核心要点                              │
│                                                              │
│  采集:                                                        │
│    MultiModalRecorder 按频率自动决定何时采样                   │
│    ImageRenderer 支持真实渲染和 mock 降级                      │
│                                                              │
│  对齐:                                                        │
│    最近邻 → 图像（不可插值）                                   │
│    线性插值 → 连续数值                                        │
│    零阶保持 → 控制信号                                        │
│                                                              │
│  存储:                                                        │
│    HDF5: 各模态独立存储 + 独立时间戳                          │
│    图像用 gzip 压缩，占总大小 > 99%                           │
│                                                              │
│  设计要点:                                                    │
│    • 不同模态帧数不同，靠时间戳关联                            │
│    • 图像和数值分开考虑压缩策略                                │
│    • 无 GPU 时有降级方案保持管道完整                           │
└──────────────────────────────────────────────────────────────┘
```
