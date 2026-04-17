# 第 0 章 · 04 — 综合练习

> 在 py 文件的 TODO 处填写代码，运行后所有断言通过即完成。

---

## 练习 1: 创建 free joint 初始 qpos

要求：位置 (0, 0, 1.2)，姿态为单位四元数 (1, 0, 0, 0)，shape=(7,)

```python
qpos_free = np.array([0.0, 0.0, 1.2, 1.0, 0.0, 0.0, 0.0])
```

---

## 练习 2: 合并 qpos

将 free joint (7 维) 和 6 个 hinge 关节角度合并为完整 qpos：

```python
full_qpos = np.concatenate([free_part, hinge_angles])   # shape=(13,)
```

---

## 练习 3: 四元数归一化

对一批四元数批量归一化，使 ||q|| = 1：

```python
norms = np.linalg.norm(quats, axis=1, keepdims=True)
normalized = quats / norms
```

`keepdims=True` 保持形状为 (N, 1)，广播除法。

---

## 练习 4: 轨迹数据分析

- **4a**: 每个关节的平均角度 → `traj.mean(axis=0)`
- **4b**: 绝对值最大的帧和关节 → `np.unravel_index(np.abs(traj).argmax(), traj.shape)`
- **4c**: 帧间角度变化（数值微分）→ `np.diff(traj, axis=0) / dt`

---

## 练习 5: 数据过滤

找出所有关节角度都在 ±1 弧度内的"安全帧"：

```python
safe_mask = np.all(np.abs(traj) < 1.0, axis=1)   # shape=(1000,), dtype=bool
safe_traj = traj[safe_mask]
```

---

## 练习 6: 限位检查

给定每个关节的 [下限, 上限]，检查每帧是否在范围内：

```python
low = joint_limits[:, 0]
high = joint_limits[:, 1]
in_limits = (traj >= low) & (traj <= high)   # shape=(1000, 6), dtype=bool
```

利用广播：`low` 的 shape 是 (6,)，`traj` 是 (1000, 6)，自动逐行比较。
