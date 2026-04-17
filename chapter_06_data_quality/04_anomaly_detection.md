# 第 6 章 · 04 — 异常检测 (Anomaly Detection)

> **目标**: 掌握多种异常检测方法，从简单的统计方法到机器学习方法，自动识别机器人轨迹中的异常片段。

## 核心知识点

1. Z-Score 异常检测
2. IQR 异常检测
3. 滑动窗口分析
4. Isolation Forest（多变量异常检测）
5. 物理一致性检查
6. 时序连续性检查
7. 综合检测器 AnomalyDetector

---

## 1. 异常类型分类

```python
class AnomalyType(Enum):
    STATISTICAL_OUTLIER     # 统计离群点
    FRAME_JUMP              # 帧间跳变
    PHYSICS_VIOLATION       # 物理违规
    TEMPORAL_INCONSISTENCY  # 时序不一致
    MULTIVARIATE_OUTLIER    # 多变量离群
```

每条异常记录包含：帧范围、涉及关节、严重度 (0-1)、描述、详情。

---

## 2. 六种检测方法

### 2.1 Z-Score 检测

\[
z = \frac{x - \mu}{\sigma}
\]

当 \(|z| > 3\) 时判定为异常（对应正态分布的 99.7%）。

| 优点 | 缺点 |
| :--- | :--- |
| 简单直观，计算快速 | 对均值/标准差敏感 |
| | 大量异常值会"污染"统计量 |

### 2.2 IQR 检测

```
IQR = Q3 - Q1
下界 = Q1 - 1.5 × IQR
上界 = Q3 + 1.5 × IQR
```

| 优点 | 缺点 |
| :--- | :--- |
| 对极端值比 z-score 更鲁棒 | 假设单峰分布 |

### 2.3 滑动窗口检测

在每个窗口内计算局部均值，与全局均值比较。能检测出**局部正常但整体异常**的片段。

```python
local_means = convolve(qpos, window)
deviations = |local_means - global_mean| / global_std
```

### 2.4 Isolation Forest

多变量异常检测方法。核心思想：

- 异常点在特征空间中"与众不同"
- 随机划分时，异常点更容易被隔离
- 路径长度越短 = 越异常

脚本提供了简化版实现（不依赖 sklearn），优先使用 sklearn 的高性能版本。

### 2.5 物理一致性检查

基于物理定律检测异常：

| 检查项 | 原理 |
| :----- | :--- |
| 加速度合理性 | \(\|a\| < a_{max}\)，电机有扭矩上限 |
| 能量突变 | 动能不应在一帧内突变（除非有外力） |

### 2.6 时序连续性检查

```python
velocity = np.diff(qpos) * hz
excessive = np.where(|velocity| > max_velocity)
```

速度超限 = 帧间跳变或丢帧。

---

## 3. 综合检测器 AnomalyDetector

整合所有检测方法，生成统一报告：

```python
config = {
    "hz": 50.0,
    "z_threshold": 3.0,
    "iqr_factor": 1.5,
    "window_size": 30,
    "contamination": 0.05,
    "max_accel": 50.0,
    "max_velocity": 5.0,
}

detector = AnomalyDetector(config)
report = detector.detect(episode_data)
detector.print_report(report)
detector.visualize(episode_data, report, "anomaly_detection.png")
```

### 报告结构

```python
{
    "total_anomalies": 15,
    "anomaly_ratio": 0.08,          # 8% 的帧被标记为异常
    "by_type": {
        "statistical_outlier": 5,
        "frame_jump": 3,
        "physics_violation": 4,
        "multivariate_outlier": 3,
    },
    "by_detector": { ... },
    "all_anomalies": [Anomaly, ...],
    "anomaly_frame_mask": np.ndarray,  # (T,) bool
}
```

---

## 4. 可视化输出

生成的 `anomaly_detection.png` 包含：

1. **前 5 个关节的时间序列** + 红色异常点高亮
2. **异常帧掩码热图** — 红色条 = 异常帧
3. **按检测器分类的异常时间条** — 不同颜色 = 不同检测器

---

## 5. 方法选择指南

| 方法 | 适用场景 | 速度 |
| :--- | :------- | :--- |
| Z-Score | 快速筛查，简单数据 | 最快 |
| IQR | 有极端值时 | 快 |
| 滑动窗口 | 检测局部模式变化 | 中 |
| Isolation Forest | 多变量关联异常 | 较慢 |
| 物理检查 | 有物理约束时 | 快 |
| 时序检查 | 帧间连续性 | 快 |

> **推荐**: 使用综合检测器 `AnomalyDetector` 一次性运行所有方法，然后根据 `by_detector` 分别查看结果。

---

## 6. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              异常检测 — 核心要点                                │
│                                                              │
│  6 种检测方法:                                                │
│    统计: Z-Score, IQR                                        │
│    时序: 滑动窗口, 时序连续性                                  │
│    物理: 加速度 / 能量守恒                                     │
│    ML: Isolation Forest (多变量)                              │
│                                                              │
│  每个异常记录:                                                │
│    类型 + 帧范围 + 关节 + 严重度 + 描述                        │
│                                                              │
│  AnomalyDetector:                                            │
│    • detect() → 运行所有检测器                                │
│    • print_report() → 汇总打印                               │
│    • visualize() → 时间序列 + 异常高亮 + 检测器分类            │
└──────────────────────────────────────────────────────────────┘
```
