# 第 0 章 · 01 — NumPy 核心操作

> **目标**: 掌握 NumPy 数组的创建、索引、运算，为后续处理 qpos 数据打基础。

## Java 类比

| NumPy | Java 对应 |
| :--- | :--- |
| `np.array` | `double[]`，但支持向量化运算 |
| `np.zeros` | `new double[n]` |
| 切片 `a[0:3]` | `Arrays.copyOfRange()`，但更强大 |
| 广播机制 | 无直接对应，NumPy 杀手级特性 |

---

## 1. 创建数组

```python
a = np.array([1.0, 2.0, 3.0])       # 一维，shape=(3,), dtype=float64
m = np.zeros((3, 4))                 # 3×4 零矩阵
np.ones(5)                           # [1, 1, 1, 1, 1]
np.arange(0, 1, 0.2)                # [0.0, 0.2, 0.4, 0.6, 0.8]
np.linspace(0, 1, 5)                # [0.0, 0.25, 0.5, 0.75, 1.0]
np.random.randn(5)                   # 标准正态分布随机数
np.array([1.0, 0.0, 0.0, 0.0])      # 单位四元数（后面频繁使用）
```

---

## 2. 索引与切片

模拟一个 qpos 数组：7 (free joint) + 6 (hinge) = 13 维

```python
position    = qpos[:3]      # 位置 xyz
quaternion  = qpos[3:7]     # 四元数 wxyz
joint_angles = qpos[7:]     # 6 个关节角度
```

| 操作 | 语法 | 说明 |
| :--- | :--- | :--- |
| 单个元素 | `qpos[2]` | z 坐标 |
| 切片 | `qpos[0:3]` 或 `qpos[:3]` | 前 3 个元素 |
| 负索引 | `qpos[-1]` | 最后一个 |
| 二维切片 | `traj[:, 2]` | 所有帧的第 3 维 |
| 子矩阵 | `traj[:5, 7:]` | 前 5 帧的关节角度 |

---

## 3. 向量化运算（告别 for 循环）

```python
a * 2         # 每个元素 ×2
a + 10        # 每个元素 +10
a ** 2        # 每个元素平方
np.sin(a)     # 每个元素求 sin
```

### 弧度 ↔ 角度转换

```python
np.rad2deg(angles_rad)    # 弧度 → 角度
np.deg2rad(angles_deg)    # 角度 → 弧度
```

### 数组间运算

```python
a + b         # 逐元素加
a * b         # 逐元素乘（不是矩阵乘）
a @ b         # 点积（内积）
np.dot(a, b)  # 同上
```

---

## 4. 统计函数

```python
data.mean()             # 全局平均
data.mean(axis=0)       # 沿时间轴，每个关节的平均值
data.std(axis=0)        # 每个关节的标准差
data.min(axis=0)        # 每个关节的最小值
data.max(axis=0)        # 每个关节的最大值
```

`axis=0` 表示沿行（时间轴）聚合，`axis=1` 表示沿列（关节轴）聚合。

---

## 5. Reshape 与转置

```python
flat = np.arange(12)          # [0, 1, ..., 11]
reshaped = flat.reshape(3, 4) # 3 行 4 列
auto = flat.reshape(4, -1)    # -1 自动推断列数
reshaped.T                    # 转置
```

---

## 6. 布尔索引（数据过滤）

```python
heights = np.array([0.95, 1.02, 0.30, 1.05, 0.88, 0.10, 1.01])
fallen = heights < 0.5                 # 布尔数组
np.where(fallen)[0]                    # 摔倒帧索引
heights[fallen]                        # 摔倒帧高度
heights[~fallen]                       # 没摔倒的帧
```

---

## 7. 拼接与分割

```python
qpos = np.concatenate([pos, quat, joints])     # 拼接成 qpos
combined = np.concatenate([traj1, traj2], axis=0)  # 沿时间轴拼接

pos, quat, joints = np.split(qpos, [3, 7])     # 分割
```

---

## 8. copy 的重要性

```python
view = original[:2]     # 切片是视图，修改 view 会影响 original！
safe = original[:2].copy()  # .copy() 才是独立副本
```

> 在 MuJoCo 中，`data.qpos` 是内部内存的引用。保存数据时一定要 `data.qpos.copy()`，否则后续 `mj_step` 会覆盖！
