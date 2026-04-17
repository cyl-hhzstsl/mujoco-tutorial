# 第 5 章 · 04 — 业界标准数据集格式

> **目标**: 了解机器人学习领域的主流数据集标准（ALOHA、LeRobot、RLDS、Open X-Embodiment），学会创建符合标准的数据并验证合规性。

## 核心知识点

1. ALOHA / ACT 数据集格式 (HDF5)
2. LeRobot 数据集格式 (Parquet + 视频)
3. RLDS 格式概念 (TFRecord)
4. Open X-Embodiment 格式
5. Schema 验证器

---

## 1. ALOHA / ACT 数据集格式

ALOHA 是 Google DeepMind 开源的双臂遥操作系统，ACT 是其配套的模仿学习算法。其数据格式已成为**事实标准之一**。

### HDF5 文件结构

```
episode_XXXX.hdf5
├── /observations/
│   ├── qpos                (T, 14)        float64
│   ├── qvel                (T, 14)        float64
│   └── /images/
│       ├── cam_high        (T, 480, 640, 3) uint8
│       ├── cam_left_wrist  (T, 480, 640, 3) uint8
│       └── cam_right_wrist (T, 480, 640, 3) uint8
├── /action                 (T, 14)        float64
└── [attributes]
    ├── sim: bool
    ├── compress: bool
    └── num_timesteps: int
```

- **14** = 7 (左臂 6 关节 + 夹爪) + 7 (右臂)
- **T** = episode 的时间步数（通常 300-500）

### 目录结构

```
dataset/
├── episode_0.hdf5
├── episode_1.hdf5
├── ...
└── episode_49.hdf5
```

### 图像数据的压缩配置

```python
imgs.create_dataset(
    "cam_high", data=cam_data,
    chunks=(1, H, W, 3),              # 每帧一个 chunk
    compression="gzip",
    compression_opts=4,
)
```

---

## 2. ALOHA Schema 验证器

脚本提供了 `validate_aloha_episode()` 函数，检查以下内容：

| 检查项 | 说明 |
| :----- | :--- |
| 必需 Group | `/observations`, `/observations/images` |
| 必需 Dataset | `/observations/qpos`, `/observations/qvel`, `/action` |
| 维度一致性 | 所有数据的 T 维度必须统一 |
| 数据类型 | qpos/qvel/action 为 float64，图像为 uint8 |
| 属性 | `num_timesteps` 必须与实际 T 一致 |
| ALOHA 特有 | nq=14 为标准双臂，nq=7 为单臂 |

```python
result = validate_aloha_episode("episode_0.hdf5")
result.report()
# ✓ 通过 / ✗ 失败（附详细错误和警告）
```

---

## 3. LeRobot 数据集格式

LeRobot 是 Hugging Face 推出的机器人学习框架，其 v2 格式设计用于大规模分发和高效训练。

### 目录结构

```
dataset/
├── meta/
│   ├── info.json            数据集元信息
│   ├── episodes.jsonl       每集的元数据
│   ├── stats.json           数据统计信息
│   └── tasks.jsonl          任务描述
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   └── chunk-001/
│       └── ...
└── videos/  (可选)
    └── chunk-000/
        └── observation.images.cam_high/
            ├── episode_000000.mp4
            └── ...
```

### Parquet 文件的列

| 列名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `observation.state` | `[nq] float` | 关节状态 |
| `observation.images.cam_high` | 视频帧引用 | 指向 mp4 文件 |
| `action` | `[nu] float` | 控制动作 |
| `episode_index` | int | 所属 episode |
| `frame_index` | int | 帧序号 |
| `timestamp` | float | 时间戳 |
| `task_index` | int | 任务 ID |
| `next.done` | bool | 是否为 episode 最后一帧 |

### 关键设计决策

- **Parquet** 存储表格化数据（高效列式存储）
- **视频独立存储为 MP4**（比逐帧 uint8 高效 10-100x）
- **按 chunk 分片**（支持大规模数据集的分布式存储）
- **元数据集中管理**（info.json, episodes.jsonl）

---

## 4. RLDS 格式

RLDS (Reinforcement Learning Datasets) 是 Google 设计的强化学习数据集标准，底层使用 TFRecord。

### 核心结构

```
Dataset = [Episode, Episode, ...]

Episode = {
    steps: [Step, Step, ...],
    metadata: {...}
}

Step = {
    observation: {image, state},
    action,
    reward,
    discount,
    is_first, is_last, is_terminal,
    language_instruction,
}
```

### 特点

- 基于 TFRecord — 高效的顺序读取
- Step-centric — 每个时间步是独立记录
- 标准化字段 — `is_first` / `is_last` / `is_terminal` 管理 episode 边界
- 与 TensorFlow Datasets (TFDS) 深度集成

### RLDS → 标准 array 格式转换

```python
states  = np.stack([s["observation"]["state"] for s in episode["steps"]])
actions = np.stack([s["action"] for s in episode["steps"]])
rewards = np.array([s["reward"] for s in episode["steps"]])
```

---

## 5. Open X-Embodiment (OXE)

Google DeepMind 主导的大规模跨机器人数据集项目。

### 架构

- 底层格式: RLDS (TFRecord)
- 标准化 observation: `image`, `wrist_image`, `state`
- 标准化 action: `[x, y, z, rx, ry, rz, gripper]`
  - (x,y,z) = 末端执行器平移
  - (rx,ry,rz) = 末端执行器旋转
  - gripper = 夹爪开合 (0=关, 1=开)

### 数据集规模

| 指标 | 规模 |
| :--- | ---: |
| 数据集数量 | 60+ |
| 机器人形态 | 22 |
| 轨迹条数 | 100K+ |
| 任务类型 | 抓取、推动、开关门、叠衣服等 |

---

## 6. 跨标准格式转换

### ALOHA HDF5 → LeRobot Parquet

```python
# 读取 ALOHA 数据
with h5py.File("episode_0.hdf5", "r") as f:
    qpos  = f["observations/qpos"][:]
    action = f["action"][:]
    T = qpos.shape[0]

# 转换为 LeRobot 表格
table = pa.table({
    "observation.state": [qpos[i].tolist() for i in range(T)],
    "action":            [action[i].tolist() for i in range(T)],
    "episode_index":     [0] * T,
    "frame_index":       list(range(T)),
    "timestamp":         np.arange(T, dtype=np.float32) / 50.0,
    "next.done":         [False] * (T - 1) + [True],
})
pq.write_table(table, "episode.parquet")
```

> **注意**: 图像/视频转换需要额外的编码工具（如 ffmpeg），这里只处理数值数据。

---

## 7. 格式选择决策树

```
你要做什么？
│
├─ 模仿学习 / 离线 RL 训练
│  ├─ 使用 ACT / Diffusion Policy  →  ALOHA HDF5
│  ├─ 使用 LeRobot 框架            →  LeRobot Parquet + 视频
│  ├─ 使用 RT-X / Octo             →  RLDS / Open X-Embodiment
│  └─ 自定义训练管线                →  HDF5（最灵活）
│
├─ 数据采集 / 快速迭代
│  └─ PKL 或 NPZ（之后转为标准格式）
│
├─ 数据共享 / 分发
│  ├─ Hugging Face Hub              →  LeRobot 格式
│  ├─ 学术发表                      →  RLDS 或 HDF5
│  └─ 团队内部                      →  HDF5
│
└─ 数据分析 / 可视化
   └─ Parquet (配合 pandas/DuckDB)
```

### 数据平台开发者建议

1. **内部存储**标准化为 HDF5（最通用、功能最全）
2. 提供到 LeRobot/RLDS 的**自动转换管线**
3. 元数据索引用 Parquet 或数据库
4. 数据采集端可以用 PKL（然后自动转换）

---

## 8. 总结

```
┌──────────────────────────────────────────────────────────┐
│          业界标准数据集格式 — 核心要点                      │
│                                                          │
│  ALOHA/ACT:                                              │
│    • HDF5, 每集一个文件                                   │
│    • observations/qpos + qvel + images + action           │
│    • 机器人模仿学习的事实标准                              │
│                                                          │
│  LeRobot:                                                │
│    • Parquet + MP4 视频                                   │
│    • 表格化数据 + 独立视频文件                             │
│    • Hugging Face 生态集成                                │
│                                                          │
│  RLDS:                                                   │
│    • TFRecord, Episode/Step 嵌套结构                      │
│    • Google 机器人数据的标准格式                           │
│    • Open X-Embodiment 的底层格式                         │
│                                                          │
│  选择原则:                                               │
│    • 内部存储 → HDF5                                     │
│    • HuggingFace → LeRobot                               │
│    • Google 生态 → RLDS                                  │
│    • 快速原型 → PKL/NPZ → 后转标准格式                    │
└──────────────────────────────────────────────────────────┘
```
