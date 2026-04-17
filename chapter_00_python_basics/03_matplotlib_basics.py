"""
第 0 章 · 03 - Matplotlib 数据可视化

目标: 学会画机器人数据最常用的几种图表。

运行: python 03_matplotlib_basics.py
输出: 在当前目录生成 PNG 图片文件
"""

import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
NUM_FRAMES = 500
DT = 0.002
time = np.arange(NUM_FRAMES) * DT

# 模拟 6 个关节的角度轨迹
joint_names = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
t_wave = np.linspace(0, 4 * np.pi, NUM_FRAMES)
joint_angles = np.column_stack([
    0.2 * np.sin(t_wave),
    0.1 * np.sin(t_wave * 1.5 + 0.5),
    -0.4 * np.sin(t_wave + 1.0),
    0.8 * np.sin(t_wave + 1.0) + 0.4,  # 膝盖: 偏置 + 振荡
    -0.3 * np.sin(t_wave + 2.0),
    0.05 * np.sin(t_wave * 2.0),
])

# ============================================================
# 图 1: 基础折线图 — 单关节角度随时间变化
# ============================================================
print("绘制图 1: 基础折线图...")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time, np.rad2deg(joint_angles[:, 3]), color='#2196F3', linewidth=1.5)
ax.set_xlabel("时间 (s)", fontsize=12)
ax.set_ylabel("膝关节角度 (°)", fontsize=12)
ax.set_title("膝关节角度随时间变化", fontsize=14)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig("plot_01_single_joint.png", dpi=150)
plt.close()
print("  → 保存到 plot_01_single_joint.png")

# ============================================================
# 图 2: 多曲线 — 所有关节角度
# ============================================================
print("绘制图 2: 多关节对比...")

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#F44336', '#E91E63', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800']
for i, (name, color) in enumerate(zip(joint_names, colors)):
    ax.plot(time, np.rad2deg(joint_angles[:, i]),
            label=name, color=color, linewidth=1.2, alpha=0.8)
ax.set_xlabel("时间 (s)", fontsize=12)
ax.set_ylabel("关节角度 (°)", fontsize=12)
ax.set_title("左腿各关节角度", fontsize=14)
ax.legend(ncol=3, fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plot_02_all_joints.png", dpi=150)
plt.close()
print("  → 保存到 plot_02_all_joints.png")

# ============================================================
# 图 3: 子图 (subplots) — 分开展示
# ============================================================
print("绘制图 3: 子图布局...")

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

for i, (name, ax, color) in enumerate(zip(joint_names, axes, colors)):
    deg = np.rad2deg(joint_angles[:, i])
    ax.plot(time, deg, color=color, linewidth=1.2)
    ax.fill_between(time, deg, alpha=0.1, color=color)
    ax.set_title(name, fontsize=11)
    ax.set_ylabel("角度 (°)")
    ax.grid(True, alpha=0.3)

    # 标注最大最小值
    max_idx = deg.argmax()
    min_idx = deg.argmin()
    ax.annotate(f"max: {deg[max_idx]:.1f}°",
                xy=(time[max_idx], deg[max_idx]),
                fontsize=8, color='red')

axes[-2].set_xlabel("时间 (s)")
axes[-1].set_xlabel("时间 (s)")
fig.suptitle("左腿关节角度详细视图", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("plot_03_subplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → 保存到 plot_03_subplots.png")

# ============================================================
# 图 4: 直方图 — 关节角度分布
# ============================================================
print("绘制图 4: 角度分布直方图...")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, (name, ax, color) in enumerate(zip(joint_names, axes, colors)):
    deg = np.rad2deg(joint_angles[:, i])
    ax.hist(deg, bins=40, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(x=deg.mean(), color='black', linestyle='--', linewidth=1.5,
               label=f'mean={deg.mean():.1f}°')
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("角度 (°)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

fig.suptitle("关节角度分布", fontsize=14)
fig.tight_layout()
fig.savefig("plot_04_histogram.png", dpi=150)
plt.close()
print("  → 保存到 plot_04_histogram.png")

# ============================================================
# 图 5: 热力图 — 关节角度相关性
# ============================================================
print("绘制图 5: 关节相关性热力图...")

corr = np.corrcoef(joint_angles.T)  # (6, 6) 相关性矩阵

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(joint_names)))
ax.set_yticks(range(len(joint_names)))
ax.set_xticklabels(joint_names, rotation=45, ha='right')
ax.set_yticklabels(joint_names)

for i in range(len(joint_names)):
    for j in range(len(joint_names)):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha='center', va='center',
                color='white' if abs(corr[i,j]) > 0.5 else 'black', fontsize=10)

fig.colorbar(im, label="Pearson 相关系数")
ax.set_title("关节角度相关性矩阵", fontsize=14)
fig.tight_layout()
fig.savefig("plot_05_correlation.png", dpi=150)
plt.close()
print("  → 保存到 plot_05_correlation.png")

# ============================================================
# 图 6: 相空间图 — 角度 vs 角速度
# ============================================================
print("绘制图 6: 相空间图...")

knee_angle = np.rad2deg(joint_angles[:, 3])
knee_velocity = np.rad2deg(np.gradient(joint_angles[:, 3], DT))

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(knee_angle, knee_velocity, c=time, cmap='viridis',
                     s=3, alpha=0.7)
ax.set_xlabel("膝关节角度 (°)", fontsize=12)
ax.set_ylabel("膝关节角速度 (°/s)", fontsize=12)
ax.set_title("膝关节相空间轨迹", fontsize=14)
ax.grid(True, alpha=0.3)
fig.colorbar(scatter, label="时间 (s)")
fig.tight_layout()
fig.savefig("plot_06_phase_space.png", dpi=150)
plt.close()
print("  → 保存到 plot_06_phase_space.png")

# ============================================================
# 图 7: 3D 轨迹图 — 基座运动轨迹
# ============================================================
print("绘制图 7: 3D 基座轨迹...")

base_x = np.linspace(0, 2, NUM_FRAMES)
base_y = np.sin(np.linspace(0, 4*np.pi, NUM_FRAMES)) * 0.1
base_z = 1.0 + np.sin(np.linspace(0, 8*np.pi, NUM_FRAMES)) * 0.02

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(base_x, base_y, base_z, color='#2196F3', linewidth=1.5)
ax.scatter(base_x[0], base_y[0], base_z[0], color='green', s=100, label='起点')
ax.scatter(base_x[-1], base_y[-1], base_z[-1], color='red', s=100, label='终点')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("机器人基座运动轨迹", fontsize=14)
ax.legend()
fig.tight_layout()
fig.savefig("plot_07_3d_trajectory.png", dpi=150)
plt.close()
print("  → 保存到 plot_07_3d_trajectory.png")

print(f"\n共生成 7 张图片，请查看当前目录下的 plot_*.png 文件")
print("✅ 第 03 节完成！")
