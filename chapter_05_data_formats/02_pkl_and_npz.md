# 第 5 章 · 02 — PKL 与 NPZ 格式详解

> **目标**: 掌握 pickle 和 NumPy 原生格式在机器人数据场景中的用法，理解各自的优劣和安全风险。

## 核心知识点

1. `pickle.dump` / `pickle.load` — 任意 Python 对象的序列化
2. 安全警告 — 为什么不应该加载不信任来源的 pkl 文件
3. `np.save` / `np.load` / `np.savez` / `np.savez_compressed`
4. 格式对比 — 文件大小、读写速度、灵活性
5. 格式选择指南

---

## 1. Pickle 基础

pickle 是 Python 内置的序列化模块，可以把几乎任意 Python 对象保存到磁盘。

```python
import pickle

# 写入
with open("trajectory.pkl", "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# 读取
with open("trajectory.pkl", "rb") as f:
    loaded = pickle.load(f)
```

### 优势与劣势

| 优势 | 劣势 |
| :--- | :--- |
| 几乎支持任意 Python 对象 | **不安全**（可执行任意代码） |
| 使用极其简单 | 仅限 Python（不跨语言） |
| 快速原型开发时非常方便 | 版本敏感（不同 Python 版本可能不兼容） |
| | 没有切片读取（必须全量加载） |

### Protocol 版本

```python
pickle.DEFAULT_PROTOCOL   # 当前默认
pickle.HIGHEST_PROTOCOL   # 最高可用（推荐）
```

使用 `pickle.HIGHEST_PROTOCOL` 可获得最佳性能和最小文件体积。

---

## 2. Pickle 安全警告

```
┌─────────────────── 严重安全风险 ───────────────────┐
│                                                     │
│  pickle.load() 可以执行任意代码！                    │
│                                                     │
│  恶意 .pkl 文件可以:                                 │
│    - 删除你的文件                                    │
│    - 安装后门                                        │
│    - 窃取密钥/密码                                   │
│    - 下载并运行恶意软件                               │
│                                                     │
│  原理: pickle 反序列化时调用 __reduce__ 方法，       │
│  攻击者可构造恶意对象让 __reduce__ 执行任意命令。    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 安全准则

1. **绝对不要**加载来自不信任来源的 `.pkl` 文件
2. 从网上下载的数据集，如果是 `.pkl` 格式，要格外小心
3. 团队内部传输 `.pkl` 文件时，确保来源可信
4. 优先使用 HDF5/NPZ 等更安全的格式分享数据
5. 如果必须用 pkl，考虑用 `fickling` 库扫描恶意内容

---

## 3. NumPy 的 .npy 和 .npz 格式

### .npy — 单个数组

```python
np.save("qpos.npy", qpos_array)
loaded = np.load("qpos.npy")
```

### .npz — 多个数组的归档

```python
# 未压缩
np.savez("data.npz", qpos=qpos, qvel=qvel, action=action)

# 压缩
np.savez_compressed("data.npz", qpos=qpos, qvel=qvel)

# 读取
data = np.load("data.npz")
qpos = data["qpos"]
data.close()    # 记得关闭！或者用 with 语句
```

### .npy 的 memory-mapped 读取

```python
mmap_data = np.load("qpos.npy", mmap_mode="r")
# 数据不全部加载到内存，按需从磁盘读取
subset = mmap_data[0:10]  # 只读前 10 步
```

### 优势与劣势

| 优势 | 劣势 |
| :--- | :--- |
| 轻量、快速 | 只能存 NumPy 数组 |
| NumPy 原生，无需额外依赖 | 没有层级结构（扁平 key-value） |
| 比 pkl 更安全 | 没有元数据支持 |
| | 不支持切片读取（默认全量加载） |

---

## 4. 存储多集轨迹数据

| 方式 | 做法 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| 一个 pkl | `pickle.dump(all_episodes)` | 简单直接 | 必须全量加载 |
| 每集一个 pkl | `episode_0000.pkl` ... | 可按需加载 | 文件数量多 |
| 每集一个 npz | `episode_0000.npz` ... | 安全、高效 | 不能存元数据 |

---

## 5. 格式性能对比

对于 T=1000, nq=14 的轨迹数据（约 336 KB 原始数据）：

| 格式 | 文件大小 | 写入速度 | 读取速度 |
| :--- | :------- | :------- | :------- |
| pkl | 中等 | 最快 | 最快 |
| npz | 较大 | 中等 | 中等 |
| npz_compressed | 最小 | 较慢（压缩开销） | 中等 |
| hdf5_gzip | 较小 | 中等 | 中等 |

---

## 6. 格式选择指南

```
┌──────────────────── 格式选择指南 ────────────────────┐
│                                                       │
│  场景                          推荐格式                │
│  ────                          ────────                │
│  快速原型 / 临时缓存           pkl                     │
│  纯数组小规模实验               npz_compressed          │
│  正式数据集 / 需要元数据        HDF5                    │
│  需要跨语言访问                 HDF5                    │
│  需要切片读取大文件             HDF5                    │
│  分享给他人                     HDF5 或 npz（更安全）   │
│  包含复杂 Python 对象           pkl（注意安全风险）     │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## 7. 常见陷阱

| 陷阱 | 说明 | 建议 |
| :--- | :--- | :--- |
| pkl 版本兼容性 | Python 3.8 的 pkl 在 3.12 中可能无法加载 | 长期存储不用 pkl |
| `allow_pickle` | `np.load` 默认 `allow_pickle=False` | 避免在 npz 中存对象数组 |
| pkl 大小膨胀 | 大批量小 dict 时开销可观 | 用 json 或 parquet |
| 内存溢出 | `pickle.load` 全量加载 | 大数据集用 HDF5 |
| 文件句柄泄漏 | `np.load` 返回的 NpzFile 需要关闭 | 用 `with` 或 `.close()` |
| 自定义类序列化 | 加载时需要类定义在作用域内 | 只序列化基础类型 |

---

## 8. 总结

```
┌──────────────────────────────────────────────────────┐
│            PKL 与 NPZ — 核心要点                      │
│                                                      │
│  PKL:                                                │
│    • 万能序列化，任意 Python 对象                      │
│    • ⚠ 严重安全风险：不要加载不信任的 pkl              │
│    • 适合快速原型和临时缓存                            │
│    • 用 HIGHEST_PROTOCOL 获得最佳性能                  │
│                                                      │
│  NPZ:                                                │
│    • NumPy 原生，只存数组                             │
│    • 比 pkl 安全，比 HDF5 轻量                        │
│    • savez_compressed 可压缩                          │
│    • .npy 支持 mmap_mode 按需读取                     │
│                                                      │
│  正式项目 → HDF5                                     │
│  快速实验 → npz_compressed                           │
│  临时缓存 → pkl (注意安全)                            │
└──────────────────────────────────────────────────────┘
```
