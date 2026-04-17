# 第 6 章 · 02 — 数据统计分析 (Data Statistics)

> **目标**: 全面计算数据集的统计特征，为数据质量评估和模型训练提供定量依据。

## 核心知识点

1. 关节级统计: 均值、标准差、百分位数、偏度、峰度
2. 轨迹级统计: 时长、平滑度、路径长度、覆盖范围
3. 数据集级统计: episode 数量、总帧数、宏观指标
4. 相关性分析: 关节间的联动关系
5. 频谱分析: FFT 揭示周期性运动模式
6. 分布偏移检测: KS 检验、Cohen's d

---

## 1. 关节级统计 (JointStatistics)

对每个关节计算的描述性统计：

| 统计量 | 含义 | 用途 |
| :----- | :--- | :--- |
| mean / std | 中心与分散度 | 归一化参数 |
| min / max / range | 极值与范围 | 限位验证 |
| median | 中位数 | 对异常值鲁棒 |
| p5 / p25 / p75 / p95 | 百分位数 | 分布描述 |
| IQR (Q3 - Q1) | 四分位距 | 异常值检测 |
| skewness | 偏度 | 0=对称, >0右偏, <0左偏 |
| kurtosis | 峰度 | 0=正态, >0尖峰, <0扁平 |

---

## 2. 轨迹级统计 (TrajectoryStatistics)

每条轨迹的宏观指标：

| 指标 | 计算方式 | 意义 |
| :--- | :------- | :--- |
| num_frames | `qpos.shape[0]` | 轨迹长度 |
| duration_seconds | `num_frames / hz` | 时长 |
| smoothness | `mean(‖d²q/dt²‖)` | 越小越平滑 |
| total_path_length | `Σ‖Δq‖` | 关节空间总位移 |
| joint_coverage | `mean(range / 2π)` | 关节范围利用率 |

### 平滑度公式

\[
\text{smoothness} = \frac{1}{T-2} \sum_{t=1}^{T-2} \|q_{t+1} - 2q_t + q_{t-1}\| \cdot f_s^2
\]

---

## 3. DatasetStatistics 使用流程

```python
ds_stats = DatasetStatistics(hz=50, joint_names=["J1", "J2", ...])

for ep_id, data in episodes:
    ds_stats.add_episode(ep_id, data)

ds_stats.compute()
ds_stats.print_summary()
ds_stats.export_json("stats.json")
ds_stats.save_visualization("data_statistics.png")
```

---

## 4. 相关性分析

```python
corr = np.corrcoef(all_qpos.T)  # (nq, nq)
```

| 相关系数 | 含义 |
| -------: | :--- |
| +1.0 | 完全正相关（同步运动） |
| 0.0 | 无相关（独立运动） |
| -1.0 | 完全负相关（反向运动） |
| \|r\| > 0.7 | 强相关（可能有机械耦合或协调控制） |

---

## 5. 频谱分析 (FFT)

对每个关节做快速傅里叶变换，揭示运动中的周期性模式：

```python
sig = qpos[:, j] - qpos[:, j].mean()  # 去均值
window = signal.windows.hann(n)         # Hann 窗减少频谱泄漏
spectrum = np.abs(np.fft.rfft(sig * window))
freqs = np.fft.rfftfreq(n, d=1.0/hz)
```

### 解读

- **主频** = 最大幅值对应的频率
- 步态运动: 主频 ≈ 步频 (如 2 Hz)
- 高频成分: 可能是噪声或不稳定的振荡
- 可用于对比不同策略的运动特征

---

## 6. 分布偏移检测

比较两个数据集 A 和 B 的分布差异：

```python
comparison = DatasetStatistics.compare_distributions(ds_a, ds_b)
```

### 检测方法

| 方法 | 含义 | 判定偏移 |
| :--- | :--- | :------- |
| KS 检验 | 两组分布是否来自同一总体 | p < 0.05 |
| Cohen's d | 效应量（均值差 / 合并标准差） | d > 0.5 |
| 均值差异 | 中心位置偏移 | — |
| 标准差比率 | 分散度变化 | — |

### 常见偏移场景

- Sim-to-real gap（仿真 → 真实）
- 不同操作员采集的数据
- 不同机器人硬件
- 数据采集条件变化

---

## 7. 可视化输出

`save_visualization()` 生成包含 6 个子图的统计大图：

1. **关节均值 ± 标准差** — 条形图
2. **关节值分布** — 箱线图
3. **相关性热图** — 红蓝色图
4. **episode 时长分布** — 直方图
5. **频谱** — 频率-幅值曲线
6. **平滑度 vs 路径长度** — 散点图

---

## 8. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              数据统计分析 — 核心要点                            │
│                                                              │
│  三个层次:                                                    │
│    关节级 → 均值/标准差/百分位/偏度/峰度                       │
│    轨迹级 → 时长/平滑度/路径长度/覆盖率                        │
│    数据集级 → episode 数/总帧数/宏观汇总                       │
│                                                              │
│  高级分析:                                                    │
│    相关性矩阵 → 发现关节联动关系                               │
│    FFT 频谱 → 揭示周期性运动模式                               │
│    分布偏移检测 → KS 检验 / Cohen's d                         │
│                                                              │
│  输出形式:                                                    │
│    • print_summary() → 终端打印                              │
│    • export_json() → 供数据平台消费                           │
│    • save_visualization() → 6 子图统计大图                    │
└──────────────────────────────────────────────────────────────┘
```
