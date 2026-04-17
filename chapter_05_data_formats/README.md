# 第 5 章：机器人数据格式

> **从仿真产出到数据工程 —— 掌握机器人学习领域的核心数据格式与转换流水线**

---

## 本章定位

前四章让你学会了如何在 MuJoCo 中建模、仿真、录制机器人轨迹。
但对于数据平台开发者来说，**真正的工作从数据落盘之后才开始**：

- 一条轨迹应该存成什么格式？
- HDF5 的分块（chunking）和压缩（compression）怎么选？
- 业界的标准数据集（ALOHA、LeRobot、RLDS）长什么样？
- 怎么在不同格式之间无损转换？

本章就是 **MuJoCo 知识** 与 **数据工程** 之间的桥梁。

---

## 本章目标

| # | 目标 | 对应脚本 |
|---|------|---------|
| 1 | 深入理解 HDF5：层级结构、压缩、分块、切片读取 | `01_hdf5_deep_dive.py` |
| 2 | 掌握 PKL / NPZ 格式的使用场景与局限性 | `02_pkl_and_npz.py` |
| 3 | 在 HDF5 / PKL / NPZ 之间自由转换 | `03_data_conversion.py` |
| 4 | 了解业界标准数据集格式（ALOHA、LeRobot、RLDS） | `04_dataset_standards.py` |
| 5 | 动手练习：构建数据处理流水线 | `05_exercises.py` |

---

## 格式速查表

| 格式 | 扩展名 | 优势 | 劣势 | 典型场景 |
|------|--------|------|------|---------|
| **HDF5** | `.hdf5` / `.h5` | 层级结构、压缩、切片读取、元数据 | API 较重、并发写受限 | 大规模数据集、ALOHA/ACT |
| **PKL** | `.pkl` | 灵活、任意 Python 对象 | 安全风险、非跨语言 | 快速原型、临时缓存 |
| **NPZ** | `.npz` | 轻量、NumPy 原生 | 无层级、无元数据 | 纯数组数据、小规模实验 |
| **Parquet** | `.parquet` | 列式存储、生态丰富 | 不适合大张量 | 元数据表、索引文件 |

---

## 标准数据集格式概览

| 标准 | 来源 | 核心格式 | 特点 |
|------|------|---------|------|
| **ALOHA/ACT** | Google DeepMind | HDF5 | 每集一个文件，固定层级结构 |
| **LeRobot** | Hugging Face | Parquet + 视频 | 表格化元数据 + 独立视频文件 |
| **RLDS** | Google | TFRecord | Steps/Episodes 嵌套结构 |
| **Open X-Embodiment** | Google et al. | RLDS | 统一多机器人数据的元标准 |

---

## 前置知识

- 第 3 章：qpos/qvel 的含义
- 第 4 章：轨迹录制的基本概念
- Python: NumPy 数组操作

## 环境准备

```bash
# 必需
pip install numpy h5py

# 可选（脚本对这些做了 try/except 处理）
pip install pyarrow pandas matplotlib tabulate
```

## 核心概念图

```
┌──────────────────── 机器人数据生命周期 ────────────────────┐
│                                                           │
│  仿真/遥操作  ──→  原始轨迹  ──→  标准格式  ──→  训练     │
│                                                           │
│  MuJoCo         dict/array     HDF5/RLDS     PyTorch     │
│  真实硬件       PKL/NPZ        LeRobot        JAX        │
│                                                           │
│              ← 本章覆盖范围 →                              │
└───────────────────────────────────────────────────────────┘
```

## 典型机器人轨迹数据结构

```
episode_0000.hdf5
├── observations/
│   ├── qpos          (T, nq)    关节位置
│   ├── qvel          (T, nv)    关节速度
│   └── images/
│       ├── cam_high  (T, H, W, C)  顶部相机
│       └── cam_wrist (T, H, W, C)  腕部相机
├── action            (T, nu)    控制指令
└── [attributes]
    ├── sim            True
    ├── robot_type     "aloha"
    ├── num_timesteps  400
    └── compress_type  "gzip"
```

---

## 文件列表

| 文件 | 行数 | 说明 |
|------|------|------|
| `01_hdf5_deep_dive.py` | ~500 | HDF5 全方位实战：创建、压缩、分块、切片、性能测试 |
| `02_pkl_and_npz.py` | ~300 | PKL 与 NPZ 格式详解、安全警告、格式对比 |
| `03_data_conversion.py` | ~400 | 格式互转、DatasetConverter 类、批量转换 |
| `04_dataset_standards.py` | ~500 | ALOHA/LeRobot/RLDS/OpenX 标准详解与验证器 |
| `05_exercises.py` | ~250 | 动手练习：数据构建、合并、索引、流水线 |
