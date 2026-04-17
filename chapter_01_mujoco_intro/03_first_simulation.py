"""
第 1 章 · 03 - 第一个完整仿真

目标: 加载单摆模型，施加控制，录制数据，画图分析。
这是一个完整的"加载 → 控制 → 录制 → 分析"流程。

运行: python 03_first_simulation.py
输出: 在当前目录生成 simulation_result.png
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pendulum.xml")

# ============================================================
# 1. 加载模型
# ============================================================
print("=" * 60)
print("1. 加载单摆模型")
print("=" * 60)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"模型: {MODEL_PATH}")
print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
print(f"timestep={model.opt.timestep}s")
print(f"关节: {model.joint(0).name} (type=hinge)")
print(f"执行器: {model.actuator(0).name}")

# ============================================================
# 2. 设置初始条件
# ============================================================
print("\n" + "=" * 60)
print("2. 设置初始条件")
print("=" * 60)

mujoco.mj_resetData(model, data)

initial_angle = np.deg2rad(45)  # 初始角度 45°
data.qpos[0] = initial_angle
mujoco.mj_forward(model, data)

print(f"初始角度: {np.rad2deg(data.qpos[0]):.1f}°")
print(f"初始角速度: {data.qvel[0]:.4f} rad/s")

# ============================================================
# 3. 仿真循环 + 数据录制
# ============================================================
print("\n" + "=" * 60)
print("3. 运行仿真 (3 个实验)")
print("=" * 60)

SIM_DURATION = 5.0  # 仿真 5 秒
NUM_STEPS = int(SIM_DURATION / model.opt.timestep)

experiments = {
    "无控制 (自由摆动)": lambda step, data: 0.0,
    "恒定力矩 (ctrl=1)": lambda step, data: 1.0,
    "正弦控制": lambda step, data: 3.0 * np.sin(2 * np.pi * data.time),
}

results = {}

for exp_name, ctrl_fn in experiments.items():
    print(f"\n实验: {exp_name}")

    mujoco.mj_resetData(model, data)
    data.qpos[0] = initial_angle

    record = {
        "time": np.zeros(NUM_STEPS),
        "angle": np.zeros(NUM_STEPS),
        "velocity": np.zeros(NUM_STEPS),
        "control": np.zeros(NUM_STEPS),
        "energy": np.zeros(NUM_STEPS),
    }

    for step in range(NUM_STEPS):
        # 设置控制信号
        data.ctrl[0] = ctrl_fn(step, data)

        # 录制数据（在 step 之前）
        record["time"][step] = data.time
        record["angle"][step] = data.qpos[0]
        record["velocity"][step] = data.qvel[0]
        record["control"][step] = data.ctrl[0]
        record["energy"][step] = data.energy[0] + data.energy[1]  # 动能+势能

        # 仿真一步
        mujoco.mj_step(model, data)

    results[exp_name] = record
    final_angle = np.rad2deg(record["angle"][-1])
    max_angle = np.rad2deg(np.max(np.abs(record["angle"])))
    print(f"  最终角度: {final_angle:.1f}°, 最大角度: {max_angle:.1f}°")

# ============================================================
# 4. 可视化结果
# ============================================================
print("\n" + "=" * 60)
print("4. 绘制结果")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
colors = ['#2196F3', '#F44336', '#4CAF50']

for (exp_name, record), color in zip(results.items(), colors):
    t = record["time"]
    axes[0].plot(t, np.rad2deg(record["angle"]),
                 label=exp_name, color=color, linewidth=1.2)
    axes[1].plot(t, np.rad2deg(record["velocity"]),
                 label=exp_name, color=color, linewidth=1.2)
    axes[2].plot(t, record["control"],
                 label=exp_name, color=color, linewidth=1.2)

axes[0].set_ylabel("角度 (°)", fontsize=12)
axes[0].set_title("单摆仿真 — 三种控制策略对比", fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel("角速度 (°/s)", fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].set_ylabel("控制力矩 (N·m)", fontsize=12)
axes[2].set_xlabel("时间 (s)", fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("simulation_result.png", dpi=150)
plt.close()
print("→ 图片已保存: simulation_result.png")

# ============================================================
# 5. 数据分析
# ============================================================
print("\n" + "=" * 60)
print("5. 数据分析")
print("=" * 60)

for exp_name, record in results.items():
    angle = record["angle"]
    vel = record["velocity"]
    print(f"\n【{exp_name}】")
    print(f"  角度范围: [{np.rad2deg(angle.min()):.1f}°, {np.rad2deg(angle.max()):.1f}°]")
    print(f"  最大角速度: {np.rad2deg(np.max(np.abs(vel))):.1f} °/s")
    print(f"  角度标准差: {np.rad2deg(angle.std()):.1f}°")

    # 计算振荡频率（通过零点交叉）
    zero_crossings = np.where(np.diff(np.sign(angle)))[0]
    if len(zero_crossings) >= 2:
        periods = np.diff(record["time"][zero_crossings])
        avg_half_period = periods.mean()
        freq = 1.0 / (2 * avg_half_period)
        print(f"  估计振荡频率: {freq:.2f} Hz")

print("\n✅ 第 03 节完成！")
print("观察: 无控制时单摆因阻尼逐渐静止；恒定力矩使其偏转；正弦控制可能产生共振。")
