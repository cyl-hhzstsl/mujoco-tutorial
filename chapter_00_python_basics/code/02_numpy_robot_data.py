"""
第 0 章 · 02 - 用 NumPy 处理机器人数据

目标: 模拟真实的机器人数据处理场景，为后续章节做准备。

运行: python 02_numpy_robot_data.py
"""

import numpy as np

# ============================================================
# 场景 1: 模拟一个人形机器人的 qpos 数据
# ============================================================
print("=" * 60)
print("场景 1: 模拟人形机器人 qpos 数据")
print("=" * 60)

NQ = 33         # free(7) + 26 个 hinge
NUM_FRAMES = 500
DT = 0.002      # 2ms 步长

# 生成模拟轨迹
np.random.seed(42)
qpos_traj = np.zeros((NUM_FRAMES, NQ))

# 基座位置: 从 (0,0,1) 开始，x 方向缓慢前进
qpos_traj[:, 0] = np.linspace(0, 2, NUM_FRAMES)    # x: 0 → 2m
qpos_traj[:, 1] = np.sin(np.linspace(0, 4*np.pi, NUM_FRAMES)) * 0.05  # y: 微小摆动
qpos_traj[:, 2] = 1.0 + np.sin(np.linspace(0, 8*np.pi, NUM_FRAMES)) * 0.02  # z: 上下起伏

# 基座四元数: 保持直立 (w=1)，加微小扰动
qpos_traj[:, 3] = 1.0                                # qw
qpos_traj[:, 4:7] = np.random.randn(NUM_FRAMES, 3) * 0.01  # qx,qy,qz 微小扰动

# 归一化四元数
quat_norms = np.linalg.norm(qpos_traj[:, 3:7], axis=1, keepdims=True)
qpos_traj[:, 3:7] /= quat_norms

# 关节角度: 模拟行走的周期性运动
t = np.linspace(0, 4 * np.pi, NUM_FRAMES)
for i in range(7, NQ):
    freq = 1 + (i % 3) * 0.5
    amp = 0.3 + (i % 5) * 0.1
    qpos_traj[:, i] = amp * np.sin(freq * t + i * 0.5)

print(f"轨迹 shape: {qpos_traj.shape}")
print(f"时长: {NUM_FRAMES * DT:.1f}s")
print(f"第 0 帧 qpos: {qpos_traj[0, :10]}...")

# ============================================================
# 场景 2: 数据统计分析
# ============================================================
print("\n" + "=" * 60)
print("场景 2: 数据统计分析")
print("=" * 60)

# 基座高度分析
z = qpos_traj[:, 2]
print(f"基座高度 z:")
print(f"  mean = {z.mean():.4f} m")
print(f"  std  = {z.std():.4f} m")
print(f"  min  = {z.min():.4f} m (第 {z.argmin()} 帧)")
print(f"  max  = {z.max():.4f} m (第 {z.argmax()} 帧)")

# 所有关节角度的统计
joint_angles = qpos_traj[:, 7:]  # 去掉 free joint
print(f"\n关节角度统计 (弧度):")
print(f"  全局 min = {joint_angles.min():.3f} ({np.rad2deg(joint_angles.min()):.1f}°)")
print(f"  全局 max = {joint_angles.max():.3f} ({np.rad2deg(joint_angles.max()):.1f}°)")

# 每个关节的变化幅度
ranges = joint_angles.max(axis=0) - joint_angles.min(axis=0)
print(f"\n关节活动范围 (变化幅度):")
print(f"  最活跃的关节: joint[{ranges.argmax()}], 幅度 = {np.rad2deg(ranges.max()):.1f}°")
print(f"  最不活跃的关节: joint[{ranges.argmin()}], 幅度 = {np.rad2deg(ranges.min()):.1f}°")

# ============================================================
# 场景 3: 计算关节角速度 (数值微分)
# ============================================================
print("\n" + "=" * 60)
print("场景 3: 数值微分 → 角速度")
print("=" * 60)

# qvel ≈ Δqpos / Δt (对 hinge 关节适用)
joint_vel = np.diff(joint_angles, axis=0) / DT
print(f"角速度 shape: {joint_vel.shape}")  # (499, 26) — 少了一帧
print(f"最大角速度: {np.rad2deg(joint_vel.max()):.1f} °/s")
print(f"平均角速度: {np.rad2deg(np.abs(joint_vel).mean()):.1f} °/s")

# ============================================================
# 场景 4: 检测异常帧
# ============================================================
print("\n" + "=" * 60)
print("场景 4: 异常帧检测")
print("=" * 60)

# 人为注入一些异常
qpos_with_issues = qpos_traj.copy()
qpos_with_issues[100, 10] = np.nan          # NaN
qpos_with_issues[200, 2] = 0.1              # 高度骤降 (摔倒)
qpos_with_issues[300, 15] = 5.0             # 关节超限
qpos_with_issues[301, 15] = -5.0            # 关节突变

# 检测 NaN
nan_mask = np.isnan(qpos_with_issues)
if nan_mask.any():
    nan_frames, nan_dims = np.where(nan_mask)
    print(f"发现 NaN: 帧 {nan_frames}, 维度 {nan_dims}")

# 检测高度异常 (< 0.5m 视为摔倒)
z = qpos_with_issues[:, 2]
fall_frames = np.where(z < 0.5)[0]
if len(fall_frames) > 0:
    print(f"疑似摔倒: 帧 {fall_frames}, 高度 {z[fall_frames]}")

# 检测关节超限 (假设限位 ±π)
joint_data = qpos_with_issues[:, 7:]
over_limit = np.abs(joint_data) > np.pi
if over_limit.any():
    frames, joints = np.where(over_limit)
    for f, j in zip(frames[:5], joints[:5]):
        print(f"关节超限: 帧 {f}, joint[{j}] = {np.rad2deg(joint_data[f, j]):.1f}°")

# 检测突变 (帧间变化 > 1 弧度)
diff = np.abs(np.diff(joint_data, axis=0))
jumps = np.where(diff > 1.0)
if len(jumps[0]) > 0:
    for f, j in zip(jumps[0][:5], jumps[1][:5]):
        print(f"关节突变: 帧 {f}→{f+1}, joint[{j}], "
              f"Δ = {np.rad2deg(diff[f, j]):.1f}°")

# ============================================================
# 场景 5: 四元数操作
# ============================================================
print("\n" + "=" * 60)
print("场景 5: 四元数基础操作")
print("=" * 60)

def quat_normalize(q):
    """归一化四元数"""
    return q / np.linalg.norm(q)

def quat_multiply(q1, q2):
    """四元数乘法 (Hamilton 积)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_conjugate(q):
    """四元数共轭 (用于求逆)"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

# 绕 Z 轴旋转 90°
angle = np.pi / 2
qz90 = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
print(f"绕 Z 轴旋转 90° 的四元数: {qz90}")

# 两次旋转 90° = 旋转 180°
qz180 = quat_multiply(qz90, qz90)
print(f"旋转 90° × 2 = {qz180}")
print(f"  (应接近 [0, 0, 0, 1] 即绕 Z 轴 180°)")

# 检查归一化
quat_norms = np.linalg.norm(qpos_traj[:, 3:7], axis=1)
print(f"\n轨迹中四元数的模:")
print(f"  min = {quat_norms.min():.6f}")
print(f"  max = {quat_norms.max():.6f}")
print(f"  全部归一化: {np.allclose(quat_norms, 1.0)}")

# ============================================================
# 场景 6: 保存与加载
# ============================================================
print("\n" + "=" * 60)
print("场景 6: 数据保存与加载")
print("=" * 60)

# NumPy 原生格式
np.save("demo_qpos_trajectory.npy", qpos_traj)
loaded = np.load("demo_qpos_trajectory.npy")
print(f"保存并加载 .npy: shape={loaded.shape}, 一致性={np.allclose(qpos_traj, loaded)}")

# 多个数组保存为 .npz
np.savez_compressed(
    "demo_episode.npz",
    qpos=qpos_traj,
    qvel=joint_vel,
    metadata=np.array([NQ, NUM_FRAMES, DT]),
)
with np.load("demo_episode.npz") as npz:
    print(f"加载 .npz 中的 keys: {list(npz.keys())}")
    print(f"  qpos shape: {npz['qpos'].shape}")
    print(f"  qvel shape: {npz['qvel'].shape}")

# 清理临时文件
import os
os.remove("demo_qpos_trajectory.npy")
os.remove("demo_episode.npz")
print("临时文件已清理")

print("\n✅ 第 02 节完成！")
