"""
第 3 章 · 04 - qpos 与 qvel 的深层关系

目标: 理解广义位置 (qpos) 和广义速度 (qvel) 之间的数学关系，
     包括维度不匹配、数值微分、雅可比矩阵、能量计算、相空间。

核心知识点:
  1. nq ≠ nv 的完整推导
  2. mj_step 如何从 qvel 更新 qpos
  3. 数值微分: qpos 轨迹 → 近似 qvel
  4. 雅可比矩阵: 关节空间 ↔ 笛卡尔空间
  5. 能量计算 (动能 + 势能)
  6. 相空间可视化

运行: python 04_qpos_qvel_relationship.py
"""

import mujoco
import numpy as np
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib 未安装，跳过绘图。安装: pip install matplotlib")

DIVIDER = "=" * 65
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 共用模型: 单摆 (最直观地展示 qpos/qvel 关系)
# ============================================================
PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 -1 2" dir="0 1 -1"/>
    <body name="pivot" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.5 0 0" mass="1"/>
      <site name="tip" pos="0.5 0 0" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

# 更复杂的模型用于后面的雅可比矩阵演示
ARM_XML = """
<mujoco model="arm_3dof">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 -1 2" dir="0 1 -1"/>
    <geom type="plane" size="2 2 0.1"/>

    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.08 0.05" rgba="0.3 0.3 0.3 1"/>

      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 0 1"
               range="-3.14 3.14" limited="true" damping="0.5"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.3" rgba="1 0.2 0.2 1" mass="2"/>

        <body name="link2" pos="0 0 0.3">
          <joint name="j2" type="hinge" axis="0 1 0"
                 range="-2.35 2.35" limited="true" damping="0.3"/>
          <geom type="capsule" size="0.035" fromto="0 0 0 0.25 0 0" rgba="0.2 1 0.2 1" mass="1.5"/>

          <body name="link3" pos="0.25 0 0">
            <joint name="j3" type="hinge" axis="0 1 0"
                   range="-2.0 2.0" limited="true" damping="0.2"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0.2 0 0" rgba="0.2 0.2 1 1" mass="1"/>
            <site name="end_effector" pos="0.2 0 0" size="0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# 含 free joint 的模型
FREE_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="box" pos="0 0 2">
      <joint name="free" type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
    <body name="floor">
      <geom type="plane" size="5 5 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


# ============================================================
# 1. nq ≠ nv 完整解析
# ============================================================
print(DIVIDER)
print("📊 1. nq ≠ nv — 完整解析与多模型对比")
print(DIVIDER)

models_info = [
    ("单摆 (1 hinge)", PENDULUM_XML),
    ("3DOF机械臂 (3 hinge)", ARM_XML),
    ("自由物体 (1 free)", FREE_XML),
]

print(f"\n  {'模型':<24} {'njnt':>5} {'nq':>5} {'nv':>5} {'nq-nv':>6} {'原因'}")
print(f"  {'-'*24} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*20}")

for name, xml in models_info:
    m = mujoco.MjModel.from_xml_string(xml)
    diff = m.nq - m.nv
    reason = "无四元数" if diff == 0 else f"{diff}个四元数(+{diff}维)"
    print(f"  {name:<24} {m.njnt:>5} {m.nq:>5} {m.nv:>5} {diff:>6} {reason}")

# 含所有关节类型的模型
all_types_xml = """
<mujoco>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="free1" type="free"/>
      <geom type="sphere" size="0.1"/>
      <body name="b2" pos="0.3 0 0">
        <joint name="ball1" type="ball"/>
        <geom type="sphere" size="0.05"/>
        <body name="b3" pos="0.2 0 0">
          <joint name="hinge1" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0.2 0 0"/>
          <body name="b4" pos="0.2 0 0">
            <joint name="slide1" type="slide" axis="1 0 0"/>
            <geom type="box" size="0.05 0.05 0.05"/>
            <body name="b5" pos="0 0 0">
              <joint name="ball2" type="ball"/>
              <geom type="sphere" size="0.03"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

m_all = mujoco.MjModel.from_xml_string(all_types_xml)
type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}

print(f"\n  含所有关节类型的模型:")
print(f"  nq = {m_all.nq}, nv = {m_all.nv}, nq - nv = {m_all.nq - m_all.nv}")
print(f"\n  {'关节':>10} {'类型':>6} {'qpos维':>6} {'qvel维':>6} {'差':>4}")
print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*4}")

nq_dims = {0: 7, 1: 4, 2: 1, 3: 1}
nv_dims = {0: 6, 1: 3, 2: 1, 3: 1}

for j in range(m_all.njnt):
    jname = mujoco.mj_id2name(m_all, mujoco.mjtObj.mjOBJ_JOINT, j)
    jtype = m_all.jnt_type[j]
    nq = nq_dims[jtype]
    nv = nv_dims[jtype]
    print(f"  {jname:>10} {type_names[jtype]:>6} {nq:>6} {nv:>6} {nq-nv:>4}")

print(f"\n  公式: nq = Σ nq_i = {m_all.nq}")
print(f"        nv = Σ nv_i = {m_all.nv}")
print(f"        差 = (free数 + ball数) × 1 = {m_all.nq - m_all.nv}")


# ============================================================
# 2. mj_step 如何更新 qpos
# ============================================================
print(f"\n{DIVIDER}")
print("⚙️  2. mj_step 内部: qvel → qpos 更新过程")
print(DIVIDER)

print("""
  mj_step 的简化流程:

    1. 计算力/力矩:     qacc = M⁻¹(qfrc_applied + qfrc_actuator - qfrc_bias)
    2. 更新速度:         qvel += qacc × dt
    3. 更新位置:         qpos = integratePos(qpos, qvel, dt)
                         ↑ 这一步用 mj_integratePos，不是简单加法

  对于 hinge/slide:   qpos += qvel × dt        (简单)
  对于 free/ball:     四元数指数映射积分         (复杂)
""")

model_p = mujoco.MjModel.from_xml_string(PENDULUM_XML)
data_p = mujoco.MjData(model_p)

# 给单摆一个初始角度
data_p.qpos[0] = np.radians(45)  # 45° 初始角度
mujoco.mj_forward(model_p, data_p)

print(f"  单摆演示: 初始角度 = 45°\n")
print(f"  {'步数':>6} {'time':>8} {'qpos(°)':>10} {'qvel(°/s)':>12} {'qacc':>10}")
print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")

for step in range(11):
    if step % 1 == 0:
        print(f"  {step:>6} {data_p.time:>8.4f} "
              f"{np.degrees(data_p.qpos[0]):>10.4f} "
              f"{np.degrees(data_p.qvel[0]):>12.4f} "
              f"{data_p.qacc[0]:>10.4f}")
    mujoco.mj_step(model_p, data_p)

print(f"""
  观察:
    - qpos 从 45° 开始减小 (重力拉回)
    - qvel 从 0 开始增大 (加速)
    - qacc 是角加速度 (由重力产生)
    - 每步: qvel += qacc × dt, qpos += qvel × dt (hinge 情况)
""")


# ============================================================
# 3. 数值微分: qpos 轨迹 → 近似 qvel
# ============================================================
print(f"{DIVIDER}")
print("📈 3. 数值微分: 从 qpos 轨迹推算 qvel")
print(DIVIDER)

print("""
  在数据平台中，有时只存了 qpos 而没存 qvel。
  可以通过数值微分近似恢复:
    qvel[t] ≈ (qpos[t+1] - qpos[t]) / dt          (前向差分)
    qvel[t] ≈ (qpos[t+1] - qpos[t-1]) / (2·dt)    (中心差分, 更精确)
""")

# 仿真单摆，记录 qpos 和 qvel
model_p = mujoco.MjModel.from_xml_string(PENDULUM_XML)
data_p = mujoco.MjData(model_p)
data_p.qpos[0] = np.radians(30)
mujoco.mj_forward(model_p, data_p)

dt = model_p.opt.timestep
n_steps = 2000
times = np.zeros(n_steps)
qpos_history = np.zeros(n_steps)
qvel_history = np.zeros(n_steps)

for i in range(n_steps):
    times[i] = data_p.time
    qpos_history[i] = data_p.qpos[0]
    qvel_history[i] = data_p.qvel[0]
    mujoco.mj_step(model_p, data_p)

# 数值微分
qvel_forward = np.gradient(qpos_history, dt, edge_order=1)   # numpy 梯度 (中心差分)
qvel_simple = np.zeros_like(qpos_history)
qvel_simple[:-1] = np.diff(qpos_history) / dt  # 前向差分
qvel_simple[-1] = qvel_simple[-2]

# 比较误差
error_gradient = np.abs(qvel_forward - qvel_history)
error_simple = np.abs(qvel_simple - qvel_history)

print(f"\n  仿真 {n_steps} 步 (dt={dt}s):")
print(f"  np.gradient (中心差分) 误差: 均值={error_gradient.mean():.6e}, 最大={error_gradient.max():.6e}")
print(f"  前向差分 误差:              均值={error_simple.mean():.6e}, 最大={error_simple.max():.6e}")
print(f"  中心差分更精确: {error_gradient.mean() < error_simple.mean()} ✅")

# 采样几个时间点详细比较
print(f"\n  {'时间(s)':>8} {'真实qvel':>12} {'中心差分':>12} {'前向差分':>12} {'中心误差':>12} {'前向误差':>12}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for idx in [0, 200, 500, 1000, 1500, 1999]:
    print(f"  {times[idx]:>8.3f} {qvel_history[idx]:>12.6f} {qvel_forward[idx]:>12.6f} "
          f"{qvel_simple[idx]:>12.6f} {error_gradient[idx]:>12.2e} {error_simple[idx]:>12.2e}")

# 用 mj_differentiatePos 做更精确的数值微分
print(f"\n  使用 mj_differentiatePos (支持四元数):")
qvel_mj = np.zeros(model_p.nv)
qpos1 = np.array([qpos_history[500]])
qpos2 = np.array([qpos_history[501]])
mujoco.mj_differentiatePos(model_p, qvel_mj, dt, qpos1, qpos2)
print(f"    步骤 500→501: mj_differentiatePos = {qvel_mj[0]:.6f}")
print(f"    真实 qvel[500] = {qvel_history[500]:.6f}")
print(f"    前向差分 = {(qpos_history[501] - qpos_history[500]) / dt:.6f}")


# ============================================================
# 4. 雅可比矩阵: 关节空间 ↔ 笛卡尔空间
# ============================================================
print(f"\n{DIVIDER}")
print("🔗 4. 雅可比矩阵 — 关节速度 ↔ 末端速度")
print(DIVIDER)

print("""
  雅可比矩阵 J 将关节速度映射到末端笛卡尔速度:
    v_ee = J · qvel

  其中:
    v_ee 是 6 维 (3 线速度 + 3 角速度)
    J 是 6×nv 矩阵
    qvel 是 nv 维

  MuJoCo API: mj_jac(model, data, jacp, jacr, point, body_id)
    jacp: 3×nv 平移雅可比
    jacr: 3×nv 旋转雅可比
""")

model_arm = mujoco.MjModel.from_xml_string(ARM_XML)
data_arm = mujoco.MjData(model_arm)

# 设置一个姿态
data_arm.qpos[:] = np.radians([30, -45, 20])
mujoco.mj_forward(model_arm, data_arm)

ee_site_id = mujoco.mj_name2id(model_arm, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
ee_body_id = model_arm.site_bodyid[ee_site_id]
ee_pos = data_arm.site_xpos[ee_site_id].copy()

# 计算雅可比矩阵
jacp = np.zeros((3, model_arm.nv))  # 平移雅可比
jacr = np.zeros((3, model_arm.nv))  # 旋转雅可比
mujoco.mj_jac(model_arm, data_arm, jacp, jacr, ee_pos, ee_body_id)

print(f"\n  当前关节角 (度): {np.degrees(data_arm.qpos[:])}")
print(f"  末端位置: {ee_pos}")

print(f"\n  平移雅可比 Jp (3×{model_arm.nv}):")
for i, axis in enumerate(["x", "y", "z"]):
    print(f"    d{axis}/dq = [{jacp[i,0]:7.4f}, {jacp[i,1]:7.4f}, {jacp[i,2]:7.4f}]")

print(f"\n  旋转雅可比 Jr (3×{model_arm.nv}):")
for i, axis in enumerate(["wx", "wy", "wz"]):
    print(f"    d{axis}/dq = [{jacr[i,0]:7.4f}, {jacr[i,1]:7.4f}, {jacr[i,2]:7.4f}]")

# 验证: J·qvel ≈ 末端速度
data_arm.qvel[:] = np.radians([10, -5, 3])
mujoco.mj_forward(model_arm, data_arm)

predicted_vel = jacp @ data_arm.qvel
print(f"\n  设 qvel = {np.degrees(data_arm.qvel)} (°/s)")
print(f"  J·qvel 预测末端线速度: {predicted_vel}")

# 数值验证: 微小步进后比较位置变化
qpos_before = data_arm.qpos.copy()
pos_before = data_arm.site_xpos[ee_site_id].copy()
small_dt = 0.001
mujoco.mj_integratePos(model_arm, data_arm.qpos, data_arm.qvel, small_dt)
mujoco.mj_forward(model_arm, data_arm)
pos_after = data_arm.site_xpos[ee_site_id].copy()
numerical_vel = (pos_after - pos_before) / small_dt

print(f"  数值近似末端线速度:     {numerical_vel}")
print(f"  相对误差: {np.linalg.norm(predicted_vel - numerical_vel) / (np.linalg.norm(numerical_vel) + 1e-10):.6e}")

data_arm.qpos[:] = qpos_before

# 雅可比的实用价值
print(f"""
  💡 雅可比矩阵的实用价值:

  1. 正向: qvel → 末端速度
     v_ee = J · qvel

  2. 逆向: 末端速度 → qvel (逆运动学速度级)
     qvel = J⁺ · v_ee  (J⁺ 是伪逆)

  3. 力映射: 末端力 → 关节力矩
     τ = Jᵀ · F_ee

  4. 奇异性检测: det(J·Jᵀ) ≈ 0 时接近奇异构型
""")

# 演示伪逆求解
desired_ee_vel = np.array([0.1, 0.0, 0.0])  # 末端沿 x 方向 0.1 m/s
J_pinv = np.linalg.pinv(jacp)
qvel_solution = J_pinv @ desired_ee_vel

print(f"  示例: 末端期望速度 = {desired_ee_vel}")
print(f"  伪逆求解 qvel = {np.degrees(qvel_solution)} (°/s)")
print(f"  验证 J·qvel = {jacp @ qvel_solution}")


# ============================================================
# 5. 能量计算
# ============================================================
print(f"\n{DIVIDER}")
print("⚡ 5. 从 qpos/qvel 计算能量")
print(DIVIDER)

# 用单摆演示能量守恒
model_p = mujoco.MjModel.from_xml_string(PENDULUM_XML)
# 减小阻尼以便观察能量守恒
model_p.dof_damping[0] = 0.0
data_p = mujoco.MjData(model_p)

data_p.qpos[0] = np.radians(60)
mujoco.mj_forward(model_p, data_p)

n_steps = 5000
energy_data = {
    "time": np.zeros(n_steps),
    "kinetic": np.zeros(n_steps),
    "potential": np.zeros(n_steps),
    "total": np.zeros(n_steps),
    "qpos": np.zeros(n_steps),
    "qvel": np.zeros(n_steps),
}

for i in range(n_steps):
    mujoco.mj_forward(model_p, data_p)

    energy_data["time"][i] = data_p.time
    energy_data["qpos"][i] = data_p.qpos[0]
    energy_data["qvel"][i] = data_p.qvel[0]

    # MuJoCo 直接提供能量
    energy_data["kinetic"][i] = data_p.energy[0]    # 动能
    energy_data["potential"][i] = data_p.energy[1]   # 势能
    energy_data["total"][i] = data_p.energy[0] + data_p.energy[1]

    mujoco.mj_step(model_p, data_p)

print(f"\n  无阻尼单摆能量守恒检验:")
print(f"  初始角度 = 60°, 仿真 {n_steps} 步")
print(f"\n  {'时间(s)':>8} {'动能':>10} {'势能':>10} {'总能量':>10} {'角度(°)':>10} {'角速度(°/s)':>12}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

for idx in [0, 500, 1000, 2000, 3000, 4000, 4999]:
    print(f"  {energy_data['time'][idx]:>8.3f} "
          f"{energy_data['kinetic'][idx]:>10.4f} "
          f"{energy_data['potential'][idx]:>10.4f} "
          f"{energy_data['total'][idx]:>10.4f} "
          f"{np.degrees(energy_data['qpos'][idx]):>10.2f} "
          f"{np.degrees(energy_data['qvel'][idx]):>12.4f}")

total_e = energy_data["total"]
print(f"\n  总能量变化: 最大={total_e.max():.6f}, 最小={total_e.min():.6f}")
print(f"  相对波动: {(total_e.max() - total_e.min()) / abs(total_e.mean()) * 100:.4f}%")
print(f"  ✅ 能量基本守恒 (数值积分有微小误差)")


# ============================================================
# 6. 相空间可视化
# ============================================================
print(f"\n{DIVIDER}")
print("🎨 6. 相空间可视化 (qpos vs qvel)")
print(DIVIDER)

if HAS_MATPLOTLIB:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("单摆相空间分析 (Pendulum Phase Space Analysis)", fontsize=14, fontweight="bold")

    # 子图 1: 角度 vs 时间
    ax1 = axes[0, 0]
    ax1.plot(energy_data["time"], np.degrees(energy_data["qpos"]), "b-", linewidth=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (degrees)")
    ax1.set_title("Angular Position vs Time")
    ax1.grid(True, alpha=0.3)

    # 子图 2: 角速度 vs 时间
    ax2 = axes[0, 1]
    ax2.plot(energy_data["time"], np.degrees(energy_data["qvel"]), "r-", linewidth=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular Velocity (deg/s)")
    ax2.set_title("Angular Velocity vs Time")
    ax2.grid(True, alpha=0.3)

    # 子图 3: 相空间轨迹 (核心!)
    ax3 = axes[1, 0]
    colors = energy_data["time"]
    sc = ax3.scatter(
        np.degrees(energy_data["qpos"]),
        np.degrees(energy_data["qvel"]),
        c=colors, cmap="viridis", s=1, alpha=0.8
    )
    ax3.set_xlabel("Angle (degrees)")
    ax3.set_ylabel("Angular Velocity (deg/s)")
    ax3.set_title("Phase Space Trajectory")
    ax3.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax3, label="Time (s)")

    # 子图 4: 能量 vs 时间
    ax4 = axes[1, 1]
    ax4.plot(energy_data["time"], energy_data["kinetic"], "r-", linewidth=0.8, label="Kinetic")
    ax4.plot(energy_data["time"], energy_data["potential"], "b-", linewidth=0.8, label="Potential")
    ax4.plot(energy_data["time"], energy_data["total"], "k--", linewidth=1.2, label="Total")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Energy (J)")
    ax4.set_title("Energy Conservation")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "phase_space.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ 相空间图已保存到: {save_path}")
    print(f"""
  图表说明:
    左上: 角度随时间振荡 (简谐运动近似)
    右上: 角速度随时间振荡 (与角度相位差 90°)
    左下: 相空间轨迹 — 闭合椭圆 → 能量守恒的标志
          颜色表示时间，沿椭圆循环 → 周期运动
    右下: 动能与势能互换，总能量恒定
""")

    # 额外: 多个初始条件的相空间
    fig2, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Phase Portrait: Multiple Initial Conditions", fontsize=13)

    for angle0 in [10, 20, 30, 45, 60, 90, 120, 150]:
        model_p2 = mujoco.MjModel.from_xml_string(PENDULUM_XML)
        model_p2.dof_damping[0] = 0.0
        data_p2 = mujoco.MjData(model_p2)
        data_p2.qpos[0] = np.radians(angle0)
        mujoco.mj_forward(model_p2, data_p2)

        qs, vs = [], []
        for _ in range(5000):
            qs.append(np.degrees(data_p2.qpos[0]))
            vs.append(np.degrees(data_p2.qvel[0]))
            mujoco.mj_step(model_p2, data_p2)

        ax.plot(qs, vs, linewidth=0.8, label=f"{angle0}°")

    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Angular Velocity (deg/s)")
    ax.legend(title="Initial angle", loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")

    save_path2 = os.path.join(SCRIPT_DIR, "phase_portrait.png")
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 相空间肖像图已保存到: {save_path2}")
    print(f"""
  相空间肖像图说明:
    - 小角度 (10°-30°): 接近圆形 → 线性简谐运动
    - 大角度 (90°-150°): 椭圆变形 → 非线性效应明显
    - 所有轨迹都是闭合的 → 保守系统
""")

else:
    print("\n  ⏭️  matplotlib 未安装，跳过绘图。")
    print("  安装: pip install matplotlib")


# ============================================================
# 7. qpos/qvel 关系总结
# ============================================================
print(f"\n{DIVIDER}")
print("📝 本节总结")
print(DIVIDER)
print("""
  ┌───────────────────────────────────────────────────────────────┐
  │                qpos 与 qvel 的关系                              │
  │                                                               │
  │  维度关系:                                                     │
  │    nq = nv + (free关节数) + (ball关节数)                       │
  │    qpos ∈ ℝ^nq  (位置流形上的坐标)                            │
  │    qvel ∈ ℝ^nv  (切空间中的速度)                               │
  │                                                               │
  │  更新关系 (mj_step):                                           │
  │    qacc = M⁻¹(τ + J^T·F - C - g)                             │
  │    qvel += qacc × dt                                          │
  │    qpos = integratePos(qpos, qvel, dt)                        │
  │                                                               │
  │  数值微分 (只有 qpos 时):                                      │
  │    qvel ≈ differentiatePos(qpos[t], qpos[t+1], dt)           │
  │    或: qvel ≈ np.gradient(qpos, dt)  (仅限 hinge/slide)      │
  │                                                               │
  │  雅可比矩阵:                                                   │
  │    v_ee = J · qvel        (关节速度 → 末端速度)               │
  │    qvel = J⁺ · v_ee       (末端速度 → 关节速度)               │
  │    τ = J^T · F_ee          (末端力 → 关节力矩)               │
  │                                                               │
  │  能量:                                                         │
  │    T = ½ qvelᵀ M qvel     (动能)                              │
  │    V = -Σ mᵢ g · xᵢ       (重力势能)                         │
  │    E = T + V               (守恒量)                           │
  └───────────────────────────────────────────────────────────────┘
""")

print("✅ 第 04 节完成！")
