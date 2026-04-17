# 第 5 章 · 01 — HDF5 深度实战

> **目标**: 全面掌握 HDF5 格式在机器人数据领域的应用，包括创建、压缩、分块、切片读取、追加写入和性能调优。

## 核心知识点

1. HDF5 是什么：文件内的层级文件系统
2. 标准机器人轨迹数据结构
3. 压缩选项与文件体积影响
4. 分块（chunking）策略
5. 切片读取与追加写入
6. 性能基准测试

---

## 1. HDF5 是什么？

HDF5 = Hierarchical Data Format version 5

核心思想：**一个文件就是一个「文件系统」**

```
HDF5 文件
├── Group（目录）
│   ├── Dataset（数组）
│   ├── Dataset
│   └── Group（子目录）
│       └── Dataset
├── Group
│   └── Dataset
└── Attributes（元数据，附着在任何节点）
```

### 类比

| HDF5 概念 | Python 类比 | Java 类比 |
| :--------- | :---------- | :-------- |
| File | 嵌套 dict（支持磁盘存储） | `Map<String, Object>` 持久化到磁盘 |
| Group | dict | 嵌套的 Map |
| Dataset | `numpy.ndarray`（支持懒加载） | NDArray |
| Attribute | dict 上附加的 metadata | Map 上的 metadata |

### 为什么机器人领域偏爱 HDF5？

1. **层级结构** — 天然适配 observations / action / images
2. **压缩** — 图像数据压缩后体积锐减
3. **切片读取** — 不用加载整个文件就能读取第 100-200 步
4. **元数据** — 可以记录机器人类型、采样率等信息
5. **跨语言** — C/C++/Java/MATLAB/Python 都有库

---

## 2. 创建 HDF5 文件

```python
import h5py
import numpy as np

with h5py.File("demo.hdf5", "w") as f:
    # 创建 Group（目录）
    obs_group = f.create_group("observations")
    img_group = obs_group.create_group("images")

    # 创建 Dataset（数组）
    obs_group.create_dataset("qpos", data=qpos_array)
    obs_group.create_dataset("qvel", data=qvel_array)
    f.create_dataset("action", data=action_array)
    img_group.create_dataset("cam_high", data=image_array)

    # 添加 Attributes（元数据）
    f.attrs["sim"] = True
    f.attrs["robot_type"] = "franka_panda"
    f.attrs["num_timesteps"] = 100
```

### 读取与遍历

```python
with h5py.File("demo.hdf5", "r") as f:
    # 遍历文件结构
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"  {name}/")
    f.visititems(print_structure)

    # 读取元数据
    for key, val in f.attrs.items():
        print(f"  {key}: {val}")
```

---

## 3. 标准机器人轨迹数据结构

以 ALOHA/ACT 为例的 episode 文件结构：

```
episode_0000.hdf5
├── observations/
│   ├── qpos          (T, nq)        float64   关节位置
│   ├── qvel          (T, nv)        float64   关节速度
│   └── images/
│       ├── cam_high  (T, 480, 640, 3) uint8   顶部相机 RGB
│       └── cam_wrist (T, 480, 640, 3) uint8   腕部相机 RGB
├── action            (T, nu)        float64   控制指令
└── [attrs]
    ├── sim=True/False
    ├── robot_type="aloha"
    └── num_timesteps=400
```

| 符号 | 含义 |
| :--- | :--- |
| T | 时间步数（一集的长度） |
| nq | 关节位置维度 |
| nv | 关节速度维度 |
| nu | 动作维度 |
| H, W, C | 图像高、宽、通道数 |

---

## 4. 压缩选项

HDF5 支持多种压缩算法，在创建 Dataset 时指定：

```python
f.create_dataset("data", data=array,
                 compression="gzip",       # 压缩算法
                 compression_opts=4)       # 压缩级别 (gzip: 0-9)
```

### 常用压缩算法

| 算法 | 特点 | compression_opts |
| :--- | :--- | :--- |
| `gzip` | 通用，压缩比好，速度中等 | 0(无) ~ 9(最强) |
| `lzf` | 速度快，压缩比稍低 | 无 |
| `szip` | 科学计算常用 | 需额外安装 |

### 压缩效果经验法则

| 数据类型 | 压缩效果 | 推荐 |
| :------- | :------- | :--- |
| 关节数据 (float64) | 一般（随机性强） | gzip-1 或 lzf |
| 图像数据 (uint8) | 非常好（像素间有空间相关性） | gzip-4 |
| 读取速度优先 | — | lzf 或不压缩 |
| 存储成本优先 | — | gzip-9 |

---

## 5. 分块策略（Chunking）

HDF5 默认把数据连续存储。启用 chunking 后，数据被切分成固定大小的「块」独立存储。

### 为什么需要 chunking？

1. **启用压缩必须先启用 chunking**
2. 不同的 chunk 大小影响读取模式的性能
3. 支持数据追加（resize）

### chunk 大小选择策略

| 访问模式 | 推荐 chunk shape |
| :------- | :--------------- |
| 顺序读整条轨迹 | `(T, nq)` 或不分块 |
| 随机读单个时间步 | `(1, nq)` |
| 读取时间窗口 | `(window_size, nq)` |
| 图像顺序读 | `(1, H, W, C)` 一帧一 chunk |
| 图像批量读 | `(batch, H, W, C)` |

```python
f.create_dataset("qpos", data=qpos,
                 chunks=(100, 7),         # 每 100 步为一个 chunk
                 compression="gzip")

f.create_dataset("cam_high", data=images,
                 chunks=(1, 480, 640, 3), # 每帧一个 chunk
                 compression="gzip", compression_opts=4)
```

> **经验**: `chunk=(100, nq)` 通常是关节数据的好折中。图像建议 `chunk=(1, H, W, C)`。

---

## 6. 切片读取 — 只读你需要的数据

```python
with h5py.File("data.hdf5", "r") as f:
    qpos = f["observations/qpos"]      # Dataset 对象（懒加载）

    first_10 = qpos[:10]               # 前 10 步
    last_5   = qpos[-5:]               # 最后 5 步
    window   = qpos[100:200]           # 第 100-200 步
    sampled  = qpos[::10]              # 每隔 10 步
    partial  = qpos[:, :3]             # 只读前 3 个关节
    fancy    = qpos[[0, 50, 100]]      # 高级索引
```

### 关键区别

```python
f["qpos"]       # → h5py.Dataset（懒加载引用，不占内存）
f["qpos"][:]    # → numpy.ndarray（实际加载到内存）
f["qpos"][0:10] # → numpy.ndarray（只加载 10 步到内存）
```

### 性能优势

对于 T=2000 的轨迹，切片读取 200 步通常比全量读取快 5-10x。

---

## 7. 追加数据到现有文件

在在线数据采集场景中，需要边采集边写入：

```python
# 第一步：创建文件，设置 maxshape 允许时间轴扩展
with h5py.File("data.hdf5", "w") as f:
    f.create_dataset(
        "qpos",
        shape=(0, 7),             # 初始为空
        maxshape=(None, 7),       # None = 该维度可以无限扩展
        dtype=np.float64,
        chunks=(100, 7),
        compression="gzip",
    )

# 第二步：分批追加
with h5py.File("data.hdf5", "a") as f:  # "a" = append 模式
    ds = f["qpos"]
    old_len = ds.shape[0]
    new_len = old_len + batch_size
    ds.resize(new_len, axis=0)           # 扩展
    ds[old_len:new_len] = new_data       # 写入新数据
```

---

## 8. 遍历 episode 目录

### 方法 1：简单遍历收集元信息

```python
episode_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))
for fname in episode_files:
    with h5py.File(os.path.join(data_dir, fname), "r") as f:
        T = f.attrs["num_timesteps"]
        qpos_shape = f["observations/qpos"].shape
```

### 方法 2：构建全局索引（用于训练时随机采样）

```python
index = []
cumulative = 0
for fname in episode_files:
    with h5py.File(path, "r") as f:
        T = int(f.attrs["num_timesteps"])
        index.append({
            "file": path,
            "num_timesteps": T,
            "global_start": cumulative,
            "global_end": cumulative + T,
        })
        cumulative += T

# 通过全局索引定位：global_idx → (file, local_idx)
```

---

## 9. 实用技巧汇总

| 技巧 | 说明 |
| :--- | :--- |
| 文件模式 | `"r"` 只读, `"w"` 覆盖写, `"a"` 追加, `"r+"` 读写 |
| 始终用 `with` | 自动关闭文件句柄 |
| 字符串属性 | `str(f.attrs["name"])` 安全读取（h5py 可能返回 bytes） |
| 检查键 | `if "observations/qpos" in f:` |
| 并发写入 | HDF5 不支持多进程同时写同一文件 |
| 并发读取 | 多进程只读是安全的 |
| 大文件策略 | 图像必开压缩; 数值数据压缩收益有限但无害 |

---

## 10. 总结

```
┌────────────────────────────────────────────────────────────┐
│                HDF5 深度实战 — 核心要点                      │
│                                                            │
│  创建: h5py.File → create_group → create_dataset           │
│  元数据: f.attrs["key"] = value                            │
│  压缩: compression="gzip", compression_opts=4              │
│  分块: chunks=(100, nq) 或 (1, H, W, C)                   │
│  切片: f["qpos"][100:200] 只读需要的部分                    │
│  追加: maxshape=(None, nq) + resize + "a" 模式             │
│                                                            │
│  最佳实践:                                                 │
│    • 数值数据 → gzip-1 + chunk=(100, nq)                   │
│    • 图像数据 → gzip-4 + chunk=(1, H, W, C)                │
│    • 始终保存 metadata 为 attributes                        │
│    • 大数据集用切片读取，不要全量加载                         │
└────────────────────────────────────────────────────────────┘
```
