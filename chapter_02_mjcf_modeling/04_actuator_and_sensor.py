"""
第 2 章 · 04 - 执行器与传感器

目标: 理解 MuJoCo 中所有执行器类型的工作原理和差异，
     掌握传感器的配置与读取，绘制执行器响应对比图。

运行: python 04_actuator_and_sensor.py
输出: actuator_comparison.png
"""

import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt

DIVIDER = "=" * 65
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. 执行器类型概览
# ============================================================
print(DIVIDER)
print("⚡ 1. MuJoCo 执行器类型概览")
print(DIVIDER)

actuator_info = {
    "motor": {
        "说明": "力矩电机 — 直接施加力/力矩",
        "公式": "force = gain * ctrl",
        "用途": "底层力控制、强化学习",
    },
    "position": {
        "说明": "位置伺服 — 内置 PD 控制器追踪目标位置",
        "公式": "force = kp * (ctrl - qpos) - kv * qvel",
        "用途": "位置控制、关节空间运动",
    },
    "velocity": {
        "说明": "速度伺服 — 追踪目标速度",
        "公式": "force = kv * (ctrl - qvel)",
        "用途": "速度控制、传送带",
    },
    "general": {
        "说明": "通用执行器 — 自定义 gain 和 bias",
        "公式": "force = gain * ctrl + bias[0] + bias[1]*qpos + bias[2]*qvel",
        "用途": "自定义控制律",
    },
}

for atype, info in actuator_info.items():
    print(f"\n  🔧 {atype}")
    print(f"     {info['说明']}")
    print(f"     公式: {info['公式']}")
    print(f"     用途: {info['用途']}")

# ============================================================
# 2. 构建对比模型：同一单摆 + 不同执行器
# ============================================================
print(f"\n{DIVIDER}")
print("🏗️  2. 构建执行器对比模型")
print(DIVIDER)

base_xml = """
<mujoco model="actuator_compare_{atype}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <body name="mount" pos="0 0 1.5">
      <geom type="cylinder" size="0.04 0.04" mass="0" rgba="0.5 0.5 0.5 1"/>
      <body name="pendulum">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.2" range="-180 180"/>
        <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.5" mass="1"
              rgba="0.3 0.6 0.9 1"/>
        <site name="tip" pos="0 0 -0.5" size="0.015"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    {actuator_xml}
  </actuator>

  <sensor>
    <jointpos name="angle" joint="hinge"/>
    <jointvel name="velocity" joint="hinge"/>
    <actuatorfrc name="act_force" actuator="act"/>
  </sensor>
</mujoco>
"""

actuator_xmls = {
    "motor": '<motor name="act" joint="hinge" ctrlrange="-5 5"/>',
    "position": '<position name="act" joint="hinge" kp="20" ctrlrange="-180 180"/>',
    "velocity": '<velocity name="act" joint="hinge" kv="5" ctrlrange="-10 10"/>',
    "general": '<general name="act" joint="hinge" gainprm="20" biasprm="0 -20 -1" ctrlrange="-180 180"/>',
}

models = {}
for atype, act_xml in actuator_xmls.items():
    xml = base_xml.format(atype=atype, actuator_xml=act_xml)
    m = mujoco.MjModel.from_xml_string(xml)
    models[atype] = m
    print(f"  ✅ {atype:>10} 模型: nq={m.nq}, nu={m.nu}")

# ============================================================
# 3. 仿真对比：相同指令，不同响应
# ============================================================
print(f"\n{DIVIDER}")
print("📊 3. 仿真对比：目标 = 90°（或等效指令）")
print(DIVIDER)

SIM_DURATION = 3.0  # 秒
DT = 0.002
N_STEPS = int(SIM_DURATION / DT)

ctrl_values = {
    "motor": 3.0,
    "position": np.radians(90),
    "velocity": 2.0,
    "general": np.radians(90),
}

results = {}

for atype in ["motor", "position", "velocity", "general"]:
    m = models[atype]
    d = mujoco.MjData(m)

    times = np.zeros(N_STEPS)
    angles = np.zeros(N_STEPS)
    velocities = np.zeros(N_STEPS)
    forces = np.zeros(N_STEPS)

    d.ctrl[0] = ctrl_values[atype]

    for step in range(N_STEPS):
        mujoco.mj_step(m, d)

        times[step] = d.time
        angles[step] = np.degrees(d.sensordata[0])
        velocities[step] = d.sensordata[1]
        forces[step] = d.sensordata[2]

    results[atype] = {
        "time": times,
        "angle": angles,
        "velocity": velocities,
        "force": forces,
    }

    final_angle = angles[-1]
    max_vel = np.max(np.abs(velocities))
    max_force = np.max(np.abs(forces))
    print(f"  {atype:>10}: 最终角度={final_angle:+7.2f}°  最大角速度={max_vel:.2f} rad/s  最大力矩={max_force:.2f} N·m")

# ============================================================
# 4. 绘制对比图
# ============================================================
print(f"\n{DIVIDER}")
print("📈 4. 绘制执行器响应对比图")
print(DIVIDER)

plt.rcParams['font.size'] = 10
colors = {
    "motor": "#e74c3c",
    "position": "#3498db",
    "velocity": "#2ecc71",
    "general": "#9b59b6",
}
labels = {
    "motor": "motor (力矩电机)",
    "position": "position (位置伺服)",
    "velocity": "velocity (速度伺服)",
    "general": "general (通用)",
}

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("MuJoCo 执行器类型对比", fontsize=14, fontweight="bold")

# 角度
ax = axes[0]
for atype in ["motor", "position", "velocity", "general"]:
    r = results[atype]
    ax.plot(r["time"], r["angle"], color=colors[atype], label=labels[atype], linewidth=1.5)
ax.axhline(y=90, color="gray", linestyle="--", alpha=0.5, label="target (90°)")
ax.set_ylabel("Joint Angle (deg)")
ax.set_title("Joint Angle Response")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# 角速度
ax = axes[1]
for atype in ["motor", "position", "velocity", "general"]:
    r = results[atype]
    ax.plot(r["time"], r["velocity"], color=colors[atype], label=labels[atype], linewidth=1.5)
ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
ax.set_ylabel("Joint Velocity (rad/s)")
ax.set_title("Joint Velocity")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# 力矩
ax = axes[2]
for atype in ["motor", "position", "velocity", "general"]:
    r = results[atype]
    ax.plot(r["time"], r["force"], color=colors[atype], label=labels[atype], linewidth=1.5)
ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
ax.set_ylabel("Actuator Force (N·m)")
ax.set_xlabel("Time (s)")
ax.set_title("Actuator Force")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, "actuator_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"  ✅ 图表已保存: {output_path}")

# ============================================================
# 5. 传感器类型详解
# ============================================================
print(f"\n{DIVIDER}")
print("📡 5. 传感器类型详解")
print(DIVIDER)

sensor_xml = """
<mujoco model="sensor_demo">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom type="plane" size="5 5 0.1"/>

    <body name="base" pos="0 0 0.5">
      <geom type="cylinder" size="0.05 0.05" mass="0"/>

      <body name="arm">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-90 90" damping="0.5"/>
        <geom name="link1" type="capsule" size="0.025" fromto="0 0 0 0 0 -0.3" mass="1"/>

        <!-- 加速度计和陀螺仪的挂载点 -->
        <site name="imu_site" pos="0 0 -0.15"/>
        <!-- 力传感器 site -->
        <site name="force_site" pos="0 0 -0.3"/>

        <body name="end" pos="0 0 -0.3">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-90 90" damping="0.3"/>
          <geom name="link2" type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.5"/>
          <site name="tip" pos="0 0 -0.2"/>
          <site name="touch_site" pos="0 0 -0.2" type="sphere" size="0.025"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="act1" joint="joint1" kp="20" ctrlrange="-90 90"/>
    <position name="act2" joint="joint2" kp="10" ctrlrange="-90 90"/>
  </actuator>

  <sensor>
    <!-- 关节传感器 -->
    <jointpos name="joint1_pos" joint="joint1"/>
    <jointpos name="joint2_pos" joint="joint2"/>
    <jointvel name="joint1_vel" joint="joint1"/>
    <jointvel name="joint2_vel" joint="joint2"/>

    <!-- 执行器传感器 -->
    <actuatorpos name="act1_pos" actuator="act1"/>
    <actuatorvel name="act1_vel" actuator="act1"/>
    <actuatorfrc name="act1_frc" actuator="act1"/>

    <!-- IMU 传感器 -->
    <accelerometer name="imu_acc" site="imu_site"/>
    <gyro name="imu_gyro" site="imu_site"/>

    <!-- 坐标系传感器 -->
    <framepos name="tip_pos" objtype="site" objname="tip"/>
    <framequat name="tip_quat" objtype="site" objname="tip"/>
    <framelinvel name="tip_linvel" objtype="site" objname="tip"/>
    <frameangvel name="tip_angvel" objtype="site" objname="tip"/>

    <!-- 触摸传感器 -->
    <touch name="tip_touch" site="touch_site"/>

    <!-- 关节限位传感器 -->
    <jointlimitfrc name="j1_limit_frc" joint="joint1"/>
  </sensor>
</mujoco>
"""

s_model = mujoco.MjModel.from_xml_string(sensor_xml)
s_data = mujoco.MjData(s_model)

# 设置控制输入并仿真
s_data.ctrl[0] = np.radians(45)
s_data.ctrl[1] = np.radians(-30)

for _ in range(500):
    mujoco.mj_step(s_model, s_data)

print(f"\n  仿真 {500 * s_model.opt.timestep:.1f}s 后的传感器读数:")
print()

sensor_type_map = {
    0: "touch", 1: "accelerometer", 2: "velocimeter", 3: "gyro",
    4: "force", 5: "torque", 6: "magnetometer", 7: "rangefinder",
    8: "jointpos", 9: "jointvel", 10: "tendonpos", 11: "tendonvel",
    12: "actuatorpos", 13: "actuatorvel", 14: "actuatorfrc",
    15: "ballquat", 16: "ballangvel",
    17: "jointlimitpos", 18: "jointlimitvel", 19: "jointlimitfrc",
    21: "framepos", 22: "framequat", 23: "framexaxis",
    24: "frameyaxis", 25: "framezaxis", 26: "framelinvel",
    27: "frameangvel", 28: "framelinacc", 29: "frameangacc",
}

sensor_descriptions = {
    "jointpos": "关节角度 (rad)",
    "jointvel": "关节角速度 (rad/s)",
    "actuatorpos": "执行器位置",
    "actuatorvel": "执行器速度",
    "actuatorfrc": "执行器力矩 (N·m)",
    "accelerometer": "线加速度 (m/s²)",
    "gyro": "角速度 (rad/s)",
    "framepos": "世界坐标位置 (m)",
    "framequat": "四元数姿态",
    "framelinvel": "线速度 (m/s)",
    "frameangvel": "角速度 (rad/s)",
    "touch": "接触力 (N)",
    "jointlimitfrc": "关节限位力 (N·m)",
}

print(f"  {'传感器名称':<18} {'类型':<16} {'维度':<6} {'描述':<20} {'读数'}")
print(f"  {'-'*18} {'-'*16} {'-'*6} {'-'*20} {'-'*30}")

for s in range(s_model.nsensor):
    name = mujoco.mj_id2name(s_model, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"sensor_{s}"
    stype_str = sensor_type_map.get(s_model.sensor_type[s], f"type_{s_model.sensor_type[s]}")
    dim = s_model.sensor_dim[s]
    adr = s_model.sensor_adr[s]
    values = s_data.sensordata[adr:adr + dim]
    desc = sensor_descriptions.get(stype_str, "")

    val_str = np.array2string(values, precision=4, separator=", ", suppress_small=True)
    print(f"  {name:<18} {stype_str:<16} {dim:<6} {desc:<20} {val_str}")

# ============================================================
# 6. 位置伺服阶跃响应分析
# ============================================================
print(f"\n{DIVIDER}")
print("📉 6. 位置伺服阶跃响应 — kp 参数影响")
print(DIVIDER)

step_xml_template = """
<mujoco>
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="mount" pos="0 0 1.5">
      <geom type="cylinder" size="0.04 0.04" mass="0"/>
      <body name="arm">
        <joint name="j" type="hinge" axis="0 1 0" damping="{damping}" range="-180 180"/>
        <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.5" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act" joint="j" kp="{kp}" ctrlrange="-180 180"/>
  </actuator>
  <sensor>
    <jointpos name="angle" joint="j"/>
  </sensor>
</mujoco>
"""

target_deg = 90
kp_values = [5, 20, 50, 100]
damping_values = [0.2, 0.2, 0.2, 0.2]

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Position Actuator Step Response Analysis", fontsize=14, fontweight="bold")

# kp 对比
ax = axes2[0]
for kp in kp_values:
    xml = step_xml_template.format(kp=kp, damping=0.2)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    d.ctrl[0] = np.radians(target_deg)

    n_steps = int(3.0 / m.opt.timestep)
    times = np.zeros(n_steps)
    angles = np.zeros(n_steps)

    for step in range(n_steps):
        mujoco.mj_step(m, d)
        times[step] = d.time
        angles[step] = np.degrees(d.sensordata[0])

    ax.plot(times, angles, label=f"kp={kp}", linewidth=1.5)

ax.axhline(y=target_deg, color="gray", linestyle="--", alpha=0.5, label=f"target ({target_deg}°)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Joint Angle (deg)")
ax.set_title("Effect of kp (damping=0.2)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 阻尼对比
ax = axes2[1]
for damp in [0.05, 0.5, 2.0, 5.0]:
    xml = step_xml_template.format(kp=30, damping=damp)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    d.ctrl[0] = np.radians(target_deg)

    n_steps = int(3.0 / m.opt.timestep)
    times = np.zeros(n_steps)
    angles = np.zeros(n_steps)

    for step in range(n_steps):
        mujoco.mj_step(m, d)
        times[step] = d.time
        angles[step] = np.degrees(d.sensordata[0])

    ax.plot(times, angles, label=f"damping={damp}", linewidth=1.5)

ax.axhline(y=target_deg, color="gray", linestyle="--", alpha=0.5, label=f"target ({target_deg}°)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Joint Angle (deg)")
ax.set_title("Effect of Joint Damping (kp=30)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = os.path.join(SCRIPT_DIR, "actuator_comparison.png")
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
plt.close()

print(f"  ✅ 阶跃响应图已保存: {output_path2}")

print(f"\n  kp 越大 → 响应越快，但可能振荡")
print(f"  阻尼越大 → 振荡越少，但响应变慢")
print(f"  最佳参数取决于具体应用场景")

# ============================================================
# 7. 总结
# ============================================================
print(f"\n{DIVIDER}")
print("📌 关键知识点总结")
print(DIVIDER)
print("""
  执行器:
  ┌─────────────┬───────────────────────────────────────┐
  │ motor       │ 直接力矩控制，最底层                   │
  │ position    │ 内置 PD，输入目标角度                  │
  │ velocity    │ 速度跟踪，输入目标角速度                │
  │ general     │ 自定义 gain + bias，最灵活              │
  └─────────────┴───────────────────────────────────────┘

  传感器:
  ┌─────────────────┬──────────────────────────────────┐
  │ jointpos/vel    │ 关节状态（最常用）                  │
  │ actuatorfrc     │ 执行器实际输出力矩                  │
  │ accelerometer   │ 线加速度（IMU）                    │
  │ gyro            │ 角速度（IMU）                      │
  │ framepos/quat   │ 刚体/site 的位姿                  │
  │ touch           │ 接触力传感器                       │
  └─────────────────┴──────────────────────────────────┘

  调参建议:
  - kp 决定响应速度，过高会导致振荡
  - joint damping 提供被动阻尼，帮助稳定
  - kv (如果有) 提供主动速度阻尼
  - ctrlrange 限制控制输入范围，防止过大指令
""")

print(f"{DIVIDER}")
print("✅ 执行器与传感器详解完成！")
print(DIVIDER)
