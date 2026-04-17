# 第 3 章 · 05 — 综合练习

> 在 py 文件的 TODO 处填写代码，运行后所有 assert 通过即完成。

## 使用的模型

混合关节机器人：free + ball + hinge + hinge + slide

```
base (free) → shoulder (ball) → elbow (hinge) → wrist (hinge) → gripper (slide)
```

| 关节 | 类型 | nq | nv | qpos 起始索引 |
| :--- | :---: | :---: | :---: | :---: |
| base_free | free | 7 | 6 | 0 |
| shoulder_ball | ball | 4 | 3 | 7 |
| elbow_hinge | hinge | 1 | 1 | 11 |
| wrist_hinge | hinge | 1 | 1 | 12 |
| gripper_slide | slide | 1 | 1 | 13 |
| **合计** | | **14** | **12** | |

nq - nv = 2（1 个 free + 1 个 ball = 2 个四元数）

---

## 练习 1: 理解 qpos 维度

计算 nq 和 nv：

```
nq = free(7) + ball(4) + hinge(1) + hinge(1) + slide(1) = 14
nv = free(6) + ball(3) + hinge(1) + hinge(1) + slide(1) = 12
```

---

## 练习 2: 解析 qpos 向量

给定 qpos，说出每个索引对应的含义：

```
qpos[0:3]   → base_free 位置 (x, y, z)
qpos[3:7]   → base_free 四元数 (w, x, y, z)
qpos[7:11]  → shoulder_ball 四元数 (w, x, y, z)
qpos[11]    → elbow_hinge 角度 (rad)
qpos[12]    → wrist_hinge 角度 (rad)
qpos[13]    → gripper_slide 位移 (m)
```

关键 API：`model.jnt_qposadr[j]` 返回关节 j 在 qpos 中的起始索引。

---

## 练习 3: 四元数归一化

实现 `normalize_quaternion(q)`：

- 计算模长 `np.linalg.norm(q)`
- 如果接近 0，返回单位四元数 `[1, 0, 0, 0]`
- 否则返回 `q / norm`

---

## 练习 4: 归一化轨迹中的所有四元数

实现 `normalize_trajectory_quaternions(model, trajectory)`：

1. 遍历所有关节，找出 free (type=0) 和 ball (type=1)
2. free 关节：四元数在 `qposadr+3` 到 `qposadr+7`
3. ball 关节：四元数在 `qposadr` 到 `qposadr+4`
4. 对每帧的每个四元数做归一化
5. 不修改原数组（先 `.copy()`）

---

## 练习 5: 检测 qpos 数据中的异常

实现 `detect_qpos_anomalies(model, trajectory, dt)`，检测三类异常：

| 异常类型 | 检测条件 |
| :--- | :--- |
| 四元数归一化偏差 | \|q\| 偏离 1 超过 0.01 |
| 关节限位违规 | hinge/slide 的值超出 `jnt_range` |
| 速度跳变 | 相邻帧变化量 / dt > 100 rad/s |

返回格式：

```python
{
    "quat_errors":      [(frame, joint_name, norm_value), ...],
    "limit_violations": [(frame, joint_name, value, (low, high)), ...],
    "velocity_jumps":   [(frame, joint_name, velocity), ...],
}
```

---

## 练习 6: 正运动学 — qpos → 末端位置

实现 `qpos_to_cartesian(model, data, qpos, site_name)`：

```python
data.qpos[:] = qpos
mujoco.mj_forward(model, data)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
position = data.site_xpos[site_id].copy()
orientation = data.site_xmat[site_id].reshape(3, 3).copy()
return position, orientation
```

验证：旋转矩阵应满足 R·R^T = I（正交性）。

---

## 练习 7: 构建 qpos 索引查询器

实现 `build_qpos_decoder(model)`，返回每个 qpos 索引到关节信息的映射：

```
qpos[ 0] → base_free       (free  ) pos_x
qpos[ 1] → base_free       (free  ) pos_y
qpos[ 2] → base_free       (free  ) pos_z
qpos[ 3] → base_free       (free  ) quat_w
qpos[ 4] → base_free       (free  ) quat_x
qpos[ 5] → base_free       (free  ) quat_y
qpos[ 6] → base_free       (free  ) quat_z
qpos[ 7] → shoulder_ball   (ball  ) quat_w
qpos[ 8] → shoulder_ball   (ball  ) quat_x
qpos[ 9] → shoulder_ball   (ball  ) quat_y
qpos[10] → shoulder_ball   (ball  ) quat_z
qpos[11] → elbow_hinge     (hinge ) angle_rad
qpos[12] → wrist_hinge     (hinge ) angle_rad
qpos[13] → gripper_slide   (slide ) displacement
```

每种关节的分量命名：

| 类型 | 分量 |
| :--- | :--- |
| free | pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z |
| ball | quat_w, quat_x, quat_y, quat_z |
| hinge | angle_rad |
| slide | displacement |
