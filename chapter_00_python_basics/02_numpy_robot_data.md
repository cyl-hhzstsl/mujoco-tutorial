# 第 0 章 · 02 — 用 NumPy 处理机器人数据

> **目标**: 模拟真实的机器人数据处理场景，为后续章节做准备。

---

## 场景 1: 模拟人形机器人 qpos 数据

- NQ = 33（free(7) + 26 个 hinge）
- 500 帧，步长 2ms，总时长 1.0s
- 基座位置：x 方向前进，y 微小摆动，z 上下起伏
- 四元数：保持直立 + 微小扰动，需归一化
- 关节角度：周期性正弦运动模拟行走

```python
quat_norms = np.linalg.norm(qpos_traj[:, 3:7], axis=1, keepdims=True)
qpos_traj[:, 3:7] /= quat_norms   # 批量归一化四元数
```

---

## 场景 2: 数据统计分析

```python
z = qpos_traj[:, 2]
z.mean(), z.std(), z.min(), z.max(), z.argmin(), z.argmax()

joint_angles = qpos_traj[:, 7:]
ranges = joint_angles.max(axis=0) - joint_angles.min(axis=0)  # 每个关节的活动幅度
ranges.argmax()  # 最活跃的关节
```

---

## 场景 3: 数值微分 → 角速度

对于 hinge 关节，角速度可以通过差分近似：

```python
joint_vel = np.diff(joint_angles, axis=0) / DT   # shape: (499, 26)
```

注意：`np.diff` 结果比输入少一帧。

---

## 场景 4: 异常帧检测

| 检测项 | 方法 |
| :--- | :--- |
| NaN 值 | `np.isnan(data).any()` + `np.where` |
| 高度骤降（摔倒） | `z < 0.5` |
| 关节超限 | `np.abs(joint_data) > np.pi` |
| 关节突变 | `np.abs(np.diff(joint_data, axis=0)) > 1.0` |

---

## 场景 5: 四元数基础操作

```python
def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_multiply(q1, q2):
    """Hamilton 积，MuJoCo [w,x,y,z] 格式"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
```

---

## 场景 6: 数据保存与加载

| 格式 | 方法 | 说明 |
| :--- | :--- | :--- |
| `.npy` | `np.save` / `np.load` | 单个数组 |
| `.npz` | `np.savez_compressed` / `np.load` | 多个数组打包，支持压缩 |

```python
np.save("trajectory.npy", qpos_traj)
np.savez_compressed("episode.npz", qpos=qpos_traj, qvel=joint_vel)
```
