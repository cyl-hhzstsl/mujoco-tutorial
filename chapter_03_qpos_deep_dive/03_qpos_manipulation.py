"""
第 3 章 · 03 - qpos 实战操作

目标: 掌握 qpos 的所有实用操作 — 读取、修改、积分、差值、
     正运动学、快照保存、插值、关节限位检查。

核心技能:
  1. 安全地读写特定关节的 qpos
  2. 用 mj_integratePos 正确积分 (不能简单 qpos += qvel*dt)
  3. 用 mj_differentiatePos 计算 qpos 差值
  4. qpos → 笛卡尔空间 (正运动学)
  5. 保存/加载 qpos 快照
  6. 在两个 qpos 之间插值
  7. 关节限位检查与裁剪

运行: python 03_qpos_manipulation.py
"""

import mujoco
import numpy as np
import json
import os
import tempfile

DIVIDER = "=" * 65

# ============================================================
# 共用的机械臂模型
# ============================================================
ARM_XML = """
<mujoco model="robot_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <light pos="0 -1 2" dir="0 1 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- 基座 (固定在世界) -->
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.08 0.05" rgba="0.3 0.3 0.3 1"/>

      <!-- 肩关节 (绕 Z 轴旋转) -->
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder_yaw" type="hinge" axis="0 0 1"
               range="-3.14 3.14" limited="true" damping="0.5"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.3" rgba="1 0.2 0.2 1"/>

        <!-- 肘关节 (绕 Y 轴旋转) -->
        <body name="link2" pos="0 0 0.3">
          <joint name="elbow_pitch" type="hinge" axis="0 1 0"
                 range="-2.35 2.35" limited="true" damping="0.3"/>
          <geom type="capsule" size="0.035" fromto="0 0 0 0.25 0 0" rgba="0.2 1 0.2 1"/>

          <!-- 腕关节 (绕 Y 轴旋转) -->
          <body name="link3" pos="0.25 0 0">
            <joint name="wrist_pitch" type="hinge" axis="0 1 0"
                   range="-2.0 2.0" limited="true" damping="0.2"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0.2 0 0" rgba="0.2 0.2 1 1"/>

            <!-- 末端 -->
            <site name="end_effector" pos="0.2 0 0" size="0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="act_shoulder" joint="shoulder_yaw" kp="50"/>
    <position name="act_elbow" joint="elbow_pitch" kp="40"/>
    <position name="act_wrist" joint="wrist_pitch" kp="30"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(ARM_XML)
data = mujoco.MjData(model)

print(f"模型: {model.nq} 维 qpos, {model.nv} 维 qvel, {model.njnt} 个关节")


# ============================================================
# 1. 设置初始姿态
# ============================================================
print(f"\n{DIVIDER}")
print("🎯 1. 通过 qpos 设置初始姿态")
print(DIVIDER)

mujoco.mj_resetData(model, data)
print(f"\n  默认 qpos:  {data.qpos}")

# 设置特定姿态: 肩转 45°, 肘弯 -60°, 腕弯 30°
target_angles_deg = [45, -60, 30]
target_angles_rad = np.radians(target_angles_deg)

data.qpos[:] = target_angles_rad
mujoco.mj_forward(model, data)

print(f"  设定姿态:   {data.qpos} (rad)")
print(f"  角度 (度):  {np.degrees(data.qpos)}")

# 读取末端执行器位置
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
ee_pos = data.site_xpos[ee_id].copy()
print(f"  末端位置:   [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")


# ============================================================
# 2. 按名称读取/修改关节
# ============================================================
print(f"\n{DIVIDER}")
print("🔧 2. 按名称读取和修改单个关节")
print(DIVIDER)


def get_joint_qpos(model, data, joint_name):
    """按名称获取关节 qpos 值"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert jid >= 0, f"关节 '{joint_name}' 不存在"
    jtype = model.jnt_type[jid]
    adr = model.jnt_qposadr[jid]
    nq = {0: 7, 1: 4, 2: 1, 3: 1}[jtype]  # free, ball, slide, hinge
    return data.qpos[adr:adr + nq].copy()


def set_joint_qpos(model, data, joint_name, value):
    """按名称设置关节 qpos 值"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert jid >= 0, f"关节 '{joint_name}' 不存在"
    jtype = model.jnt_type[jid]
    adr = model.jnt_qposadr[jid]
    nq = {0: 7, 1: 4, 2: 1, 3: 1}[jtype]
    value = np.atleast_1d(np.asarray(value, dtype=float))
    assert len(value) == nq, f"需要 {nq} 维, 得到 {len(value)} 维"
    data.qpos[adr:adr + nq] = value


joint_names = ["shoulder_yaw", "elbow_pitch", "wrist_pitch"]
print(f"\n  当前各关节角度:")
for name in joint_names:
    val = get_joint_qpos(model, data, name)
    print(f"    {name:<16} = {val[0]:8.4f} rad ({np.degrees(val[0]):7.2f}°)")

# 只修改肘关节
print(f"\n  修改 elbow_pitch = -90°:")
set_joint_qpos(model, data, "elbow_pitch", np.radians(-90))
mujoco.mj_forward(model, data)

for name in joint_names:
    val = get_joint_qpos(model, data, name)
    print(f"    {name:<16} = {val[0]:8.4f} rad ({np.degrees(val[0]):7.2f}°)")

new_ee_pos = data.site_xpos[ee_id]
print(f"  末端位置变化: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
print(f"             → [{new_ee_pos[0]:.4f}, {new_ee_pos[1]:.4f}, {new_ee_pos[2]:.4f}]")


# ============================================================
# 3. mj_integratePos — 正确的 qpos 积分
# ============================================================
print(f"\n{DIVIDER}")
print("📐 3. mj_integratePos — 为什么不能用 qpos += qvel * dt")
print(DIVIDER)

# 用含 free joint 的模型来演示差异
integrate_xml = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <joint name="free" type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

m_int = mujoco.MjModel.from_xml_string(integrate_xml)
d_int = mujoco.MjData(m_int)

print(f"\n  模型: nq={m_int.nq}, nv={m_int.nv}")
print(f"  问题: qpos 有 7 维, qvel 有 6 维, 不能直接相加！")

# 设置初始状态: 有位移和旋转速度
d_int.qpos[:] = [0, 0, 1, 1, 0, 0, 0]  # 位置 + 单位四元数
d_int.qvel[:] = [0.1, 0, 0, 0, 0, 0.5]  # 线速度 + 角速度 (绕 z)
dt = 0.01

# 方法 1: MuJoCo 正确积分
qpos_correct = d_int.qpos.copy()
mujoco.mj_integratePos(m_int, qpos_correct, d_int.qvel, dt)

# 方法 2: 天真的加法 (错误!)
qpos_naive = d_int.qpos.copy()
# 这会失败或产生错误结果，因为维度不匹配
# qpos_naive += d_int.qvel * dt  # 维度不对！
# 即使对位置部分做了，四元数部分也不能简单加
qpos_naive[:3] += d_int.qvel[:3] * dt  # 位置部分还行
# 四元数部分: 简单加角速度是错的
omega = d_int.qvel[3:6]
q_old = qpos_naive[3:7].copy()
# 错误方法: 直接往四元数上加
q_naive = q_old + dt * 0.5 * np.array([
    -omega[0]*q_old[1] - omega[1]*q_old[2] - omega[2]*q_old[3],
    omega[0]*q_old[0] + omega[2]*q_old[2] - omega[1]*q_old[3],
    omega[1]*q_old[0] - omega[2]*q_old[1] + omega[0]*q_old[3],
    omega[2]*q_old[0] + omega[1]*q_old[1] - omega[0]*q_old[2],
])
qpos_naive[3:7] = q_naive / np.linalg.norm(q_naive)

print(f"\n  初始 qpos: {d_int.qpos}")
print(f"  qvel:      {d_int.qvel}")
print(f"  dt = {dt}")
print(f"\n  mj_integratePos 结果: {qpos_correct}")
print(f"  手动积分结果:         {qpos_naive}")
print(f"  位置差异: {np.linalg.norm(qpos_correct[:3] - qpos_naive[:3]):.6e}")
print(f"  四元数差异: {np.linalg.norm(qpos_correct[3:7] - qpos_naive[3:7]):.6e}")

print(f"""
  💡 关键点:
    - mj_integratePos 正确处理了四元数的指数映射积分
    - 它保证积分后四元数仍然归一化
    - 对于纯 hinge/slide 关节，两种方法结果相同
    - 但只要有 free 或 ball 关节，就必须用 mj_integratePos
""")


# ============================================================
# 4. mj_differentiatePos — qpos 差值
# ============================================================
print(f"{DIVIDER}")
print("📏 4. mj_differentiatePos — 正确计算 qpos 差值")
print(DIVIDER)

print(f"""
  问题: 给定两个 qpos，如何计算它们的"差"?

  对于 hinge/slide: Δ = qpos2 - qpos1 (简单减法)
  对于 free/ball:   不能简单减! 四元数相减没有物理意义。

  mj_differentiatePos 返回的是 qvel 空间 (nv 维) 的差:
    qvel_diff = differentiatePos(qpos2, qpos1) / dt
""")

# 用回机械臂模型
mujoco.mj_resetData(model, data)

# 两个关节构型
qpos1 = np.radians([0, 0, 0])
qpos2 = np.radians([45, -30, 15])

data.qpos[:] = qpos1
mujoco.mj_forward(model, data)
ee_pos1 = data.site_xpos[ee_id].copy()

data.qpos[:] = qpos2
mujoco.mj_forward(model, data)
ee_pos2 = data.site_xpos[ee_id].copy()

# 计算 qpos 差值
qvel_diff = np.zeros(model.nv)
mujoco.mj_differentiatePos(model, qvel_diff, 1.0, qpos1, qpos2)

print(f"\n  qpos1 (度): {np.degrees(qpos1)}")
print(f"  qpos2 (度): {np.degrees(qpos2)}")
print(f"  简单减法:   {np.degrees(qpos2 - qpos1)} (度)")
print(f"  mj_differentiatePos: {np.degrees(qvel_diff)} (度/s, dt=1)")
print(f"  对于纯 hinge，两者相同: {np.allclose(qpos2 - qpos1, qvel_diff)} ✅")

print(f"\n  末端位置变化: {ee_pos1} → {ee_pos2}")
print(f"  笛卡尔距离: {np.linalg.norm(ee_pos2 - ee_pos1):.4f} m")


# ============================================================
# 5. 正运动学: qpos → 笛卡尔空间
# ============================================================
print(f"\n{DIVIDER}")
print("🗺️  5. 正运动学: qpos → 笛卡尔空间坐标")
print(DIVIDER)

print("""
  正运动学 (Forward Kinematics):
    给定关节角度 qpos，计算末端执行器在世界坐标系中的位置和朝向。
    在 MuJoCo 中，只需: 设置 qpos → mj_forward → 读取 site_xpos
""")

# 扫描肩关节，观察末端轨迹
print(f"\n  肩关节从 -180° 到 180° 扫描，末端执行器轨迹:")
print(f"  {'角度(°)':>8} {'x':>8} {'y':>8} {'z':>8} {'到原点距离':>10}")
print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

mujoco.mj_resetData(model, data)
trajectory = []

for angle_deg in range(-180, 181, 30):
    data.qpos[:] = [np.radians(angle_deg), np.radians(-45), np.radians(20)]
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[ee_id].copy()
    dist = np.linalg.norm(pos[:2])  # XY 平面上到原点距离
    trajectory.append(pos)
    print(f"  {angle_deg:>8} {pos[0]:>8.4f} {pos[1]:>8.4f} {pos[2]:>8.4f} {dist:>10.4f}")

print(f"\n  观察: 肩关节旋转时，末端在 XY 平面上画圆弧，Z 不变 ✅")


# ============================================================
# 6. 保存/加载 qpos 快照
# ============================================================
print(f"\n{DIVIDER}")
print("💾 6. 保存和加载 qpos 快照")
print(DIVIDER)


def save_qpos_snapshot(model, data, filepath, metadata=None):
    """保存 qpos 快照到 JSON 文件。

    包含足够的元信息用于验证兼容性。
    """
    snapshot = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "njnt": int(model.njnt),
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "time": float(data.time),
        "joints": {},
    }

    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jname:
            jtype = model.jnt_type[j]
            adr = model.jnt_qposadr[j]
            nq = {0: 7, 1: 4, 2: 1, 3: 1}[jtype]
            snapshot["joints"][jname] = {
                "type": ["free", "ball", "slide", "hinge"][jtype],
                "qpos_adr": int(adr),
                "nq": int(nq),
                "value": data.qpos[adr:adr + nq].tolist(),
            }

    if metadata:
        snapshot["metadata"] = metadata

    with open(filepath, "w") as f:
        json.dump(snapshot, f, indent=2)

    return filepath


def load_qpos_snapshot(model, data, filepath, strict=True):
    """从 JSON 文件加载 qpos 快照。

    strict=True 时检查 nq/nv 兼容性。
    """
    with open(filepath, "r") as f:
        snapshot = json.load(f)

    if strict:
        assert snapshot["nq"] == model.nq, \
            f"nq 不匹配: 文件={snapshot['nq']}, 模型={model.nq}"
        assert snapshot["nv"] == model.nv, \
            f"nv 不匹配: 文件={snapshot['nv']}, 模型={model.nv}"

    data.qpos[:] = snapshot["qpos"]
    data.qvel[:] = snapshot["qvel"]
    data.time = snapshot["time"]

    mujoco.mj_forward(model, data)
    return snapshot


# 演示
data.qpos[:] = np.radians([45, -60, 30])
data.qvel[:] = [0.1, -0.2, 0.3]
data.time = 1.5
mujoco.mj_forward(model, data)

tmpdir = tempfile.mkdtemp()
filepath = os.path.join(tmpdir, "arm_snapshot.json")

save_qpos_snapshot(model, data, filepath, metadata={"task": "pick_and_place", "step": 42})
print(f"\n  快照已保存到: {filepath}")

# 读取快照内容展示
with open(filepath) as f:
    content = json.load(f)
print(f"  快照内容预览:")
print(f"    nq={content['nq']}, nv={content['nv']}, time={content['time']}")
print(f"    qpos={content['qpos']}")
print(f"    关节详情:")
for jname, jinfo in content["joints"].items():
    print(f"      {jname}: {jinfo['type']}, value={jinfo['value']}")

# 重置后加载
mujoco.mj_resetData(model, data)
print(f"\n  重置后 qpos: {data.qpos}")

snapshot = load_qpos_snapshot(model, data, filepath)
print(f"  加载后 qpos: {data.qpos}")
print(f"  加载后 time: {data.time}")
print(f"  恢复成功 ✅")

# 清理
os.remove(filepath)
os.rmdir(tmpdir)


# ============================================================
# 7. 在两个 qpos 之间插值
# ============================================================
print(f"\n{DIVIDER}")
print("🔀 7. qpos 插值 — 生成平滑轨迹")
print(DIVIDER)

print("""
  对于纯 hinge/slide 关节: 可以直接线性插值
  对于含 free/ball 的模型: 必须对四元数部分做 SLERP

  这里的机械臂全是 hinge，线性插值即可。
  通用情况下应该用 mj_integratePos 来做。
""")

qpos_start = np.radians([0, 0, 0])
qpos_end = np.radians([90, -90, 45])

n_steps = 10

print(f"\n  起点 (度): {np.degrees(qpos_start)}")
print(f"  终点 (度): {np.degrees(qpos_end)}")
print(f"\n  {'步数':>4} {'shoulder':>10} {'elbow':>10} {'wrist':>10} {'末端 x':>8} {'末端 y':>8} {'末端 z':>8}")
print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

interpolated_trajectory = []

for i in range(n_steps + 1):
    t = i / n_steps
    qpos_t = (1 - t) * qpos_start + t * qpos_end
    data.qpos[:] = qpos_t
    mujoco.mj_forward(model, data)
    ee = data.site_xpos[ee_id].copy()
    interpolated_trajectory.append({"qpos": qpos_t.copy(), "ee_pos": ee.copy()})

    angles = np.degrees(qpos_t)
    print(f"  {i:>4} {angles[0]:>10.2f} {angles[1]:>10.2f} {angles[2]:>10.2f} "
          f"{ee[0]:>8.4f} {ee[1]:>8.4f} {ee[2]:>8.4f}")

print(f"\n  ⚠️  注意: 关节空间的线性插值 ≠ 笛卡尔空间的线性路径")
print(f"  末端走的是曲线而非直线！")


# ============================================================
# 7b. 通用 qpos 插值 (支持四元数)
# ============================================================
print(f"\n--- 通用 qpos 插值 (支持 free/ball 关节) ---")


def interpolate_qpos(model, qpos1, qpos2, t):
    """通用 qpos 插值: 自动处理四元数部分。

    利用 mj_differentiatePos 和 mj_integratePos 实现。
    """
    qvel_diff = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, qvel_diff, 1.0, qpos1, qpos2)

    result = qpos1.copy()
    mujoco.mj_integratePos(model, result, qvel_diff, t)
    return result


# 在机械臂上验证 (纯 hinge，应该和线性插值一样)
qpos_mid = interpolate_qpos(model, qpos_start, qpos_end, 0.5)
qpos_mid_linear = 0.5 * qpos_start + 0.5 * qpos_end
print(f"  通用插值 t=0.5: {np.degrees(qpos_mid)} (度)")
print(f"  线性插值 t=0.5: {np.degrees(qpos_mid_linear)} (度)")
print(f"  结果一致: {np.allclose(qpos_mid, qpos_mid_linear)} ✅")


# ============================================================
# 8. 关节限位检查
# ============================================================
print(f"\n{DIVIDER}")
print("🚧 8. 关节限位检查与裁剪")
print(DIVIDER)


def check_joint_limits(model, qpos, verbose=True):
    """检查 qpos 是否在关节限位范围内。

    返回: (all_ok: bool, violations: list[dict])
    """
    violations = []

    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue

        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        jtype = model.jnt_type[j]
        adr = model.jnt_qposadr[j]
        low, high = model.jnt_range[j]

        if jtype in (2, 3):  # slide or hinge
            val = qpos[adr]
            if val < low or val > high:
                violations.append({
                    "joint": jname,
                    "value": float(val),
                    "range": [float(low), float(high)],
                    "excess": float(max(low - val, val - high)),
                })
                if verbose:
                    print(f"  ❌ {jname}: {np.degrees(val):.2f}° 超出 "
                          f"[{np.degrees(low):.2f}°, {np.degrees(high):.2f}°]")
            elif verbose:
                print(f"  ✅ {jname}: {np.degrees(val):.2f}° 在 "
                      f"[{np.degrees(low):.2f}°, {np.degrees(high):.2f}°] 范围内")

        elif jtype == 1:  # ball — 限位是最大偏转角
            q = qpos[adr:adr + 4]
            qw = np.clip(np.abs(q[0]), 0.0, 1.0)
            angle = 2.0 * np.arccos(qw)
            if angle > high:
                violations.append({
                    "joint": jname,
                    "value": float(angle),
                    "range": [float(low), float(high)],
                    "excess": float(angle - high),
                })
                if verbose:
                    print(f"  ❌ {jname}: 偏转角 {np.degrees(angle):.2f}° 超出 "
                          f"最大 {np.degrees(high):.2f}°")
            elif verbose:
                print(f"  ✅ {jname}: 偏转角 {np.degrees(angle):.2f}° 在 "
                      f"最大 {np.degrees(high):.2f}° 范围内")

    return len(violations) == 0, violations


def clip_to_joint_limits(model, qpos):
    """将 qpos 裁剪到关节限位范围内 (in-place 修改并返回)"""
    qpos = qpos.copy()
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        jtype = model.jnt_type[j]
        adr = model.jnt_qposadr[j]
        low, high = model.jnt_range[j]
        if jtype in (2, 3):  # slide or hinge
            qpos[adr] = np.clip(qpos[adr], low, high)
        elif jtype == 1:  # ball — 将四元数裁剪到最大偏转角
            q = qpos[adr:adr + 4].copy()
            qw = np.abs(q[0])
            if qw < 1.0:
                angle = 2.0 * np.arccos(np.clip(qw, 0.0, 1.0))
                if angle > high:
                    axis = q[1:4]
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 1e-10:
                        axis /= axis_norm
                        half = high / 2.0
                        q[0] = np.cos(half) * np.sign(q[0])
                        q[1:4] = axis * np.sin(half)
                    qpos[adr:adr + 4] = q
    return qpos


# 正常姿态
print(f"\n  检查正常姿态 [45°, -60°, 30°]:")
qpos_ok = np.radians([45, -60, 30])
ok, _ = check_joint_limits(model, qpos_ok)

# 超限姿态
print(f"\n  检查超限姿态 [200°, -150°, 30°]:")
qpos_bad = np.radians([200, -150, 30])
ok, violations = check_joint_limits(model, qpos_bad)
print(f"  违规数量: {len(violations)}")

# 裁剪
qpos_clipped = clip_to_joint_limits(model, qpos_bad)
print(f"\n  裁剪后:")
print(f"    原始 (度): {np.degrees(qpos_bad)}")
print(f"    裁剪 (度): {np.degrees(qpos_clipped)}")
print(f"  裁剪后检查:")
check_joint_limits(model, qpos_clipped)


# ============================================================
# 9. 数据平台实用工具集
# ============================================================
print(f"\n{DIVIDER}")
print("🏗️  9. 数据平台实用工具函数")
print(DIVIDER)


def qpos_to_ee_pos(model, data, qpos, site_name="end_effector"):
    """qpos → 末端执行器笛卡尔位置 (正运动学)"""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def compute_workspace_bounds(model, data, n_samples=5000, site_name="end_effector"):
    """通过随机采样估算工作空间边界"""
    positions = []
    for _ in range(n_samples):
        # 在关节限位范围内随机采样
        for j in range(model.njnt):
            adr = model.jnt_qposadr[j]
            if model.jnt_limited[j]:
                low, high = model.jnt_range[j]
                data.qpos[adr] = np.random.uniform(low, high)
            else:
                data.qpos[adr] = np.random.uniform(-np.pi, np.pi)

        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        positions.append(data.site_xpos[site_id].copy())

    positions = np.array(positions)
    return {
        "min": positions.min(axis=0),
        "max": positions.max(axis=0),
        "mean": positions.mean(axis=0),
        "std": positions.std(axis=0),
    }


def validate_qpos_trajectory(model, trajectory):
    """验证一条 qpos 轨迹的数据质量。

    检查项: 维度、关节限位、四元数归一化、速度跳变。
    返回: dict 包含检查结果和统计信息。
    """
    trajectory = np.array(trajectory)
    n_frames, nq = trajectory.shape

    report = {
        "n_frames": n_frames,
        "nq": nq,
        "nq_match": nq == model.nq,
        "limit_violations": 0,
        "quat_norm_errors": 0,
        "velocity_jumps": 0,
    }

    for frame_idx in range(n_frames):
        qpos = trajectory[frame_idx]

        # 关节限位检查
        _, violations = check_joint_limits(model, qpos, verbose=False)
        report["limit_violations"] += len(violations)

        # 四元数归一化检查 (free 和 ball 关节)
        for j in range(model.njnt):
            jtype = model.jnt_type[j]
            if jtype == 0:  # free
                adr = model.jnt_qposadr[j]
                q = qpos[adr + 3:adr + 7]
                if abs(np.linalg.norm(q) - 1.0) > 1e-3:
                    report["quat_norm_errors"] += 1
            elif jtype == 1:  # ball
                adr = model.jnt_qposadr[j]
                q = qpos[adr:adr + 4]
                if abs(np.linalg.norm(q) - 1.0) > 1e-3:
                    report["quat_norm_errors"] += 1

    # 速度跳变检测
    if n_frames >= 2:
        diffs = np.diff(trajectory, axis=0)
        velocities = np.linalg.norm(diffs, axis=1)
        if n_frames >= 3:
            median_vel = np.median(velocities)
            if median_vel > 0:
                jump_threshold = 5 * median_vel
                report["velocity_jumps"] = int(np.sum(velocities > jump_threshold))

    return report


# 演示工作空间估算
print(f"\n  估算机械臂工作空间 (采样 5000 点)...")
ws = compute_workspace_bounds(model, data, n_samples=5000)
print(f"    X 范围: [{ws['min'][0]:.3f}, {ws['max'][0]:.3f}] m")
print(f"    Y 范围: [{ws['min'][1]:.3f}, {ws['max'][1]:.3f}] m")
print(f"    Z 范围: [{ws['min'][2]:.3f}, {ws['max'][2]:.3f}] m")

# 演示轨迹验证
print(f"\n  验证插值轨迹...")
test_traj = np.array([step["qpos"] for step in interpolated_trajectory])
report = validate_qpos_trajectory(model, test_traj)
print(f"    帧数: {report['n_frames']}")
print(f"    nq 匹配: {report['nq_match']}")
print(f"    限位违规: {report['limit_violations']}")
print(f"    四元数异常: {report['quat_norm_errors']}")
print(f"    速度跳变: {report['velocity_jumps']}")


# ============================================================
# 10. 总结
# ============================================================
print(f"\n{DIVIDER}")
print("📝 本节总结")
print(DIVIDER)
print("""
  ┌──────────────────────────────────────────────────────────────┐
  │                    qpos 操作速查                               │
  │                                                              │
  │  读取关节:  data.qpos[model.jnt_qposadr[jid]:adr+nq]       │
  │  修改关节:  设置 qpos 后调用 mj_forward 更新派生量           │
  │                                                              │
  │  正确积分:  mj_integratePos(model, qpos, qvel, dt)          │
  │  计算差值:  mj_differentiatePos(model, diff, dt, q1, q2)    │
  │                                                              │
  │  正运动学:  设 qpos → mj_forward → 读 site_xpos             │
  │  快照保存:  qpos + qvel + time + 关节元信息 → JSON          │
  │                                                              │
  │  插值:      hinge/slide 线性插值即可                          │
  │             free/ball 用 differentiatePos + integratePos     │
  │                                                              │
  │  限位检查:  model.jnt_limited + model.jnt_range              │
  └──────────────────────────────────────────────────────────────┘
""")

print("✅ 第 03 节完成！")
