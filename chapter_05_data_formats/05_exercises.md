# 第 5 章 · 05 — 动手练习

> 在 py 文件的 TODO 处填写代码，运行后所有 assert 通过即完成。

## 练习概览

| # | 练习 | 核心技能 |
| --: | :--- | :------- |
| 1 | 创建标准 HDF5 Episode 文件 | HDF5 创建、压缩、元数据 |
| 2 | 合并多个 Episode 文件 | HDF5 读写、递归复制 |
| 3 | 构建数据集索引 | 目录遍历、全局索引映射 |
| 4 | 格式转换 Pipeline | PKL → HDF5 → NPZ 三步转换 |

---

## 练习 1: 创建标准 HDF5 Episode 文件

创建一个符合 ALOHA 标准的 episode 文件。

### 要求

| 路径 | shape | dtype | 说明 |
| :--- | :---- | :---- | :--- |
| `/observations/qpos` | (300, 14) | float64 | 关节位置 |
| `/observations/qvel` | (300, 14) | float64 | 关节速度 |
| `/observations/images/cam_high` | (300, 60, 80, 3) | uint8 | 需要压缩 |
| `/action` | (300, 14) | float64 | 控制指令 |

属性: `sim=True`, `robot_type="aloha"`, `num_timesteps=300`

### 验证项

- 文件结构完整
- shape 和 dtype 正确
- 图像数据已启用压缩
- 属性值正确
- qpos 数值有物理意义（使用正弦波生成）

---

## 练习 2: 合并多个 Episode 文件

编写 `merge_episodes(input_dir, output_path)` 函数。

### 输入

目录中有多个 `episode_XXXX.hdf5` 文件，每个包含不同长度的轨迹。

### 输出结构

```
merged_dataset.hdf5
├── episode_0000/
│   ├── observations/
│   │   ├── qpos    (T0, nq)
│   │   └── qvel    (T0, nq)
│   └── action      (T0, nq)
├── episode_0001/
│   └── ...
└── [attrs]
    ├── total_episodes
    └── total_timesteps
```

### 关键实现

```python
def merge_episodes(input_dir, output_path):
    with h5py.File(output_path, "w") as out_f:
        for fname in sorted(episode_files):
            with h5py.File(fpath, "r") as in_f:
                ep_group = out_f.create_group(ep_name)
                # 递归复制所有 dataset
                in_f.visititems(copy_item)
                # 复制 episode 属性
                for key, val in in_f.attrs.items():
                    ep_group.attrs[key] = val
        out_f.attrs["total_episodes"] = total
        out_f.attrs["total_timesteps"] = total_steps
```

---

## 练习 3: 构建数据集索引

编写 `build_dataset_index(dataset_dir)` 和 `lookup_global_index(index, global_idx)` 函数。

### 索引结构

每个 episode 的索引包含：

```python
{
    "filename": "episode_0003.hdf5",
    "filepath": "/full/path/...",
    "episode_idx": 3,
    "num_timesteps": 250,
    "nq": 7,
    "has_images": True,
    "robot_type": "franka_panda",
    "file_size_bytes": 12345,
    "global_start": 680,      # 全局起始索引
    "global_end": 930,        # 全局结束索引
}
```

### 全局索引查找

```python
# 全局索引 700 → episode_0003.hdf5 的第 20 步
entry, local_idx = lookup_global_index(index, 700)
# entry["filename"] == "episode_0003.hdf5"
# local_idx == 20
```

### 汇总统计

```python
summary = {
    "total_episodes": 10,
    "total_timesteps": 2500,
    "avg_episode_length": 250,
    "min_episode_length": 80,
    "max_episode_length": 400,
    "has_images_count": 4,
    "robot_types": ["franka_panda", "aloha"],
}
```

---

## 练习 4: 格式转换 Pipeline

实现 PKL → HDF5 → NPZ 的三步转换流水线。

### 流程

```
raw_episode.pkl  →  converted.hdf5  →  converted.npz
                     (带压缩和元数据)
```

### 每步要求

1. **PKL → HDF5**
   - 读取 PKL 中的嵌套 dict
   - 创建 HDF5 文件，启用 gzip 压缩
   - 元数据存为 attributes
   - 验证 qpos 数值一致

2. **HDF5 → NPZ**
   - 读取 HDF5 中的数组和元数据
   - 展平嵌套结构为 `"observations.qpos"` 格式
   - 字符串元数据加 `"__str__"` 前缀
   - 使用 `savez_compressed` 保存
   - 验证数值一致

3. **端到端验证**
   - 最终 NPZ 中的数据与原始 PKL 数值完全一致
   - 记录每步的耗时和文件大小

### Pipeline 日志格式

```
[PKL → HDF5]
  输入: 125.3 KB
  输出: 89.7 KB
  耗时: 12.5ms
  ✓ 数据一致性验证通过

[HDF5 → NPZ]
  输入: 89.7 KB
  输出: 76.2 KB
  耗时: 8.3ms
  ✓ 数据一致性验证通过
```
