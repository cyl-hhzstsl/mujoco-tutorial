# 第 0 章 · 03 — Matplotlib 数据可视化

> **目标**: 学会画机器人数据最常用的几种图表。

运行后在当前目录生成 7 张 PNG 图片（`plot_01` ~ `plot_07`）。

---

## 图 1: 基础折线图 — 单关节角度

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time, knee_angle, color='#2196F3', linewidth=1.5)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("角度 (°)")
ax.grid(True, alpha=0.3)
```

---

## 图 2: 多曲线 — 所有关节角度对比

```python
for i, (name, color) in enumerate(zip(joint_names, colors)):
    ax.plot(time, angles[:, i], label=name, color=color)
ax.legend(ncol=3)
```

---

## 图 3: 子图 (subplots) — 分开展示

```python
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
```

- `sharex=True`：所有子图共享 X 轴
- `fill_between`：填充曲线下方区域
- `annotate`：标注最大/最小值

---

## 图 4: 直方图 — 关节角度分布

```python
ax.hist(angles, bins=40, alpha=0.7, edgecolor='white')
ax.axvline(x=mean, color='black', linestyle='--')   # 均值线
```

---

## 图 5: 热力图 — 关节角度相关性

```python
corr = np.corrcoef(joint_angles.T)   # Pearson 相关系数矩阵
ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
```

- 红色 → 正相关，蓝色 → 负相关
- 对角线恒为 1（自相关）

---

## 图 6: 相空间图 — 角度 vs 角速度

```python
ax.scatter(angle, velocity, c=time, cmap='viridis', s=3)
```

颜色映射时间，闭合轨迹表示周期运动。

---

## 图 7: 3D 轨迹图 — 基座运动轨迹

```python
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.scatter(x[0], y[0], z[0], color='green', label='起点')
```

---

## 常用技巧速查

| 技巧 | 代码 |
| :--- | :--- |
| 设置分辨率 | `fig.savefig("out.png", dpi=150)` |
| 自动调整布局 | `fig.tight_layout()` |
| 参考线 | `ax.axhline(y=0, linestyle='--')` |
| 网格 | `ax.grid(True, alpha=0.3)` |
| 图例 | `ax.legend(loc='upper right')` |
| 颜色条 | `fig.colorbar(scatter, label="...")` |
