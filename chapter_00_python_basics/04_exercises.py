"""
第 0 章 · 04 - 综合练习

在 TODO 标记处填写代码，然后运行此文件验证。
所有断言通过 = 练习完成。

运行: python 04_exercises.py
"""

import numpy as np

passed = 0
total = 0


def check(name, condition):
    global passed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}")


# ============================================================
# 练习 1: 创建一个 free joint 的初始 qpos
# ============================================================
print("练习 1: 创建 free joint 初始 qpos")
print("  要求: 位置 (0, 0, 1.2), 姿态为单位四元数 (1, 0, 0, 0)")

# TODO: 创建 qpos_free, shape=(7,)
qpos_free = np.array([0.0, 0.0, 1.2, 1.0, 0.0, 0.0, 0.0])  # ← 参考答案

check("shape 正确", qpos_free.shape == (7,))
check("位置正确", np.allclose(qpos_free[:3], [0, 0, 1.2]))
check("四元数正确", np.allclose(qpos_free[3:7], [1, 0, 0, 0]))

# ============================================================
# 练习 2: 合并 qpos
# ============================================================
print("\n练习 2: 合并 qpos")
print("  要求: 将 free joint(7维) 和 6 个 hinge 关节角度合并成完整 qpos")

free_part = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
hinge_angles = np.array([0.1, -0.3, 0.5, 0.8, -0.2, 0.0])

# TODO: 合并成 full_qpos, shape=(13,)
full_qpos = np.concatenate([free_part, hinge_angles])  # ← 参考答案

check("shape 正确", full_qpos.shape == (13,))
check("前 7 维正确", np.allclose(full_qpos[:7], free_part))
check("后 6 维正确", np.allclose(full_qpos[7:], hinge_angles))

# ============================================================
# 练习 3: 四元数归一化
# ============================================================
print("\n练习 3: 四元数归一化")
print("  要求: 对一批四元数进行归一化")

quats = np.array([
    [2.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.5, 0.5, 0.5, 0.5],
])

# TODO: 归一化每个四元数，使 ||q|| = 1
norms = np.linalg.norm(quats, axis=1, keepdims=True)  # ← 参考答案
normalized = quats / norms

check("shape 不变", normalized.shape == (3, 4))
check("第 1 个归一化",
      np.allclose(np.linalg.norm(normalized[0]), 1.0))
check("第 2 个归一化",
      np.allclose(np.linalg.norm(normalized[1]), 1.0))
check("全部归一化",
      np.allclose(np.linalg.norm(normalized, axis=1), 1.0))

# ============================================================
# 练习 4: 轨迹数据分析
# ============================================================
print("\n练习 4: 轨迹数据分析")
print("  要求: 分析关节角度轨迹")

np.random.seed(123)
traj = np.random.randn(1000, 6) * 0.5  # 1000 帧, 6 个关节

# TODO 4a: 计算每个关节的平均角度, shape=(6,)
means = traj.mean(axis=0)  # ← 参考答案

check("means shape", means.shape == (6,))
check("means 值正确", np.allclose(means, traj.mean(axis=0)))

# TODO 4b: 找出角度绝对值最大的帧和关节
abs_max_idx = np.unravel_index(np.abs(traj).argmax(), traj.shape)  # ← 参考答案
max_frame, max_joint = abs_max_idx

check("找到最大值帧", isinstance(max_frame, (int, np.intp)))
check("找到最大值关节", isinstance(max_joint, (int, np.intp)))
print(f"    (帧={max_frame}, 关节={max_joint}, "
      f"值={np.rad2deg(traj[max_frame, max_joint]):.1f}°)")

# TODO 4c: 计算帧间角度变化 (数值微分), shape=(999, 6)
dt = 0.002
velocity = np.diff(traj, axis=0) / dt  # ← 参考答案

check("velocity shape", velocity.shape == (999, 6))

# ============================================================
# 练习 5: 数据过滤
# ============================================================
print("\n练习 5: 数据过滤")
print("  要求: 找出所有关节角度都在 ±1 弧度内的'安全帧'")

# TODO: 生成布尔掩码, shape=(1000,)
safe_mask = np.all(np.abs(traj) < 1.0, axis=1)  # ← 参考答案
safe_traj = traj[safe_mask]

check("safe_mask shape", safe_mask.shape == (1000,))
check("safe_mask 类型", safe_mask.dtype == bool)
check("过滤后数据合法",
      np.all(np.abs(safe_traj) < 1.0))
print(f"    安全帧数: {safe_mask.sum()} / {len(traj)}")

# ============================================================
# 练习 6: 弧度/角度 互转与限位检查
# ============================================================
print("\n练习 6: 限位检查")
print("  要求: 检查关节角度是否在给定限位范围内")

joint_limits = np.array([
    [-1.57, 1.57],   # joint 0: ±90°
    [-0.52, 0.52],   # joint 1: ±30°
    [-2.09, 0.52],   # joint 2: -120° ~ 30°
    [0.0, 2.62],     # joint 3: 0° ~ 150° (膝盖不能反弯)
    [-1.05, 1.05],   # joint 4: ±60°
    [-0.52, 0.52],   # joint 5: ±30°
])

# TODO: 检查 traj 的每一帧每个关节是否在限位内
# in_limits shape = (1000, 6), dtype=bool
low = joint_limits[:, 0]   # ← 参考答案
high = joint_limits[:, 1]
in_limits = (traj >= low) & (traj <= high)

check("in_limits shape", in_limits.shape == (1000, 6))

violations_per_joint = (~in_limits).sum(axis=0)
for i, (name, count) in enumerate(
        zip(["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_p", "ankle_r"],
            violations_per_joint)):
    lo_deg = np.rad2deg(joint_limits[i, 0])
    hi_deg = np.rad2deg(joint_limits[i, 1])
    print(f"    {name}: {count} 帧超限 "
          f"(限位 [{lo_deg:.0f}°, {hi_deg:.0f}°])")

# ============================================================
# 结果
# ============================================================
print(f"\n{'=' * 40}")
print(f"通过: {passed}/{total}")
if passed == total:
    print("🎉 全部通过！可以进入第 1 章了！")
else:
    print("继续加油，修改 TODO 部分再试一次。")
