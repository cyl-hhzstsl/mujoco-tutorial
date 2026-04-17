# 第 5 章 · 03 — 数据格式互转

> **目标**: 构建通用的 DatasetConverter，实现 HDF5、PKL、NPZ 之间的无损转换，包括批量转换和元数据保留。

## 核心知识点

1. HDF5 ↔ PKL ↔ NPZ 双向转换
2. DatasetConverter 统一接口
3. 批量目录转换
4. 元数据保留策略
5. 转换 Pipeline 实战

---

## 1. 基础转换函数

### HDF5 读写

```python
def _hdf5_group_to_dict(group) -> dict:
    """递归把 HDF5 Group 转为嵌套 dict"""
    result = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            result[key] = _hdf5_group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[:]              # 加载为 ndarray
    return result

def _dict_to_hdf5_group(group, data: dict, compression="gzip"):
    """递归把嵌套 dict 写入 HDF5 Group"""
    for key, val in data.items():
        if isinstance(val, dict):
            sub = group.create_group(key)
            _dict_to_hdf5_group(sub, val, compression)
        elif isinstance(val, np.ndarray):
            group.create_dataset(key, data=val, compression=compression)
        else:
            group.attrs[key] = val             # 标量/字符串存为 attribute
```

### NPZ 的层级编码

NPZ 不支持层级结构，需要将嵌套 dict 展平：

```python
# 展平：嵌套 key 用 "." 连接
"observations.qpos" → ndarray
"observations.qvel" → ndarray
"metadata.robot_type" → "__str__metadata.robot_type"  # 字符串加前缀

# 还原：按 "." 拆分重建嵌套
```

| 原始数据 | 展平后的 key |
| :------- | :----------- |
| `data["observations"]["qpos"]` | `"observations.qpos"` |
| `data["metadata"]["robot_type"]` (str) | `"__str__metadata.robot_type"` |
| `data["metadata"]["hz"]` (int) | `"metadata.hz"` (0-d array) |

---

## 2. DatasetConverter 统一转换器

```python
class DatasetConverter:
    READERS = {"hdf5": read_hdf5, "pkl": read_pkl, "npz": read_npz}
    WRITERS = {"hdf5": write_hdf5, "pkl": write_pkl, "npz": write_npz}

    def convert(self, input_path, output_path):
        """单文件转换：自动检测格式"""
        data = self.read(input_path)
        self.write(output_path, data)

    def batch_convert(self, input_dir, output_dir, target_format="hdf5"):
        """批量转换目录中的所有文件"""
        ...

    def verify_conversion(self, original_path, converted_path):
        """验证转换后数据与原始数据的一致性"""
        ...
```

### 使用示例

```python
converter = DatasetConverter()
converter.convert("episode.hdf5", "episode.pkl")
converter.convert("episode.pkl", "episode.npz")
converter.batch_convert("pkl_dir/", "hdf5_dir/", target_format="hdf5")
```

---

## 3. 批量目录转换

```python
# PKL → HDF5 批量转换
converter.batch_convert(
    input_dir="raw_episodes/",
    output_dir="standardized/",
    target_format="hdf5",
    source_extensions=(".pkl",),
)
```

转换过程中会：
- 自动跳过已是目标格式的文件
- 打印每个文件的输入/输出大小
- 计算总压缩比

---

## 4. 元数据保留策略

不同格式对元数据的支持程度不同：

| 格式 | 层级结构 | 属性/元数据 | 类型保真 |
| :--- | :------: | :---------: | :------: |
| HDF5 | ✓ | ✓ | ✓ |
| PKL | ✓ | ✓ | ✓ |
| NPZ | ✗ | △ (通过编码) | △ |

### NPZ 的元数据编码方案

1. 嵌套 dict 的 key 用 `"."` 连接展平
2. 字符串值加 `"__str__"` 前缀
3. 标量值转为 0-d 数组
4. 还原时按前缀和维度恢复原始类型

> **注意**: 经过 NPZ 中转后，某些类型信息可能丢失（如 bool → int）。HDF5 和 PKL 之间的往返转换通常是无损的。

---

## 5. 转换性能对比

| 数据规模 | 格式 | 写入 | 读取 | 文件大小 |
| :------- | :--- | ---: | ---: | :------- |
| 小 (T=100, nq=7) | hdf5 | 中 | 中 | 小 |
| | pkl | 快 | 快 | 中 |
| | npz | 中 | 中 | 最小 |
| 大 (T=5000, nq=14) | hdf5 | 中 | 中 | 最小 |
| | pkl | 快 | 快 | 最大 |
| | npz | 慢 | 中 | 小 |

### 总结

- **PKL** 读写最快（内存序列化零开销）
- **NPZ** 压缩版文件最小（zip 压缩）
- **HDF5** 是速度和功能的最佳平衡
- 数据量越大，HDF5 的压缩优势越明显

---

## 6. 转换 Pipeline 实战

实际场景：数据采集系统产出 PKL 文件 → 需要转换为标准 HDF5 格式。

```
Pipeline 步骤:
  1. 扫描源目录
  2. 验证每个文件的数据完整性
  3. 转换格式
  4. 验证转换结果
  5. 生成转换报告
```

### 转换报告示例

```
────── 转换报告 ──────
总文件数:  6
成功:      6
失败:      0
总时间步:  1842
输入总量:  285.3 KB
输出总量:  198.7 KB
压缩比:    69.65%
```

---

## 7. 总结

```
┌────────────────────────────────────────────────────────────┐
│              数据格式互转 — 核心要点                         │
│                                                            │
│  转换链:                                                   │
│    HDF5 ←→ PKL ←→ NPZ                                     │
│    任意两种格式之间可以双向转换                               │
│                                                            │
│  关键实现:                                                 │
│    • HDF5: 递归遍历 Group/Dataset 与嵌套 dict 互转         │
│    • NPZ: 展平嵌套 dict 为 "key1.key2" 形式               │
│    • PKL: 直接 dump/load 嵌套 dict                         │
│                                                            │
│  DatasetConverter:                                         │
│    • convert() — 单文件转换                                 │
│    • batch_convert() — 批量目录转换                         │
│    • verify_conversion() — 一致性验证                       │
│                                                            │
│  最佳实践:                                                 │
│    • 转换后立即验证数据一致性                                │
│    • 元数据经过 NPZ 中转可能有类型损失                       │
│    • 正式数据集统一存为 HDF5                                 │
│    • 批量转换时生成转换报告                                  │
└────────────────────────────────────────────────────────────┘
```
