"""
第 3 章 · 02 - 四元数深度教程

目标: 从零理解四元数，掌握在 MuJoCo 中处理旋转的所有技巧。

为什么需要四元数？
  - 欧拉角有万向锁 (gimbal lock) 问题
  - 旋转矩阵有 9 个元素但只有 3 个自由度，冗余且难以约束正交性
  - 四元数: 4 个数、无万向锁、容易插值、MuJoCo 默认使用

MuJoCo 四元数顺序: [w, x, y, z]
  - w 是实部 (标量部分)
  - (x, y, z) 是虚部 (向量部分)
  - ⚠️ scipy 用 [x, y, z, w]，顺序相反！

运行: python 02_quaternion_deep_dive.py
"""

import mujoco
import numpy as np

# scipy 是可选的——如果没装就跳过相关部分
try:
    from scipy.spatial.transform import Rotation, Slerp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy 未安装，部分高级功能将跳过。安装: pip install scipy")

DIVIDER = "=" * 65


# ============================================================
# 1. 四元数基础
# ============================================================
print(DIVIDER)
print("📐 1. 四元数基础 — 什么是四元数？")
print(DIVIDER)

print("""
  四元数 q = w + xi + yj + zk

  其中:
    w     — 实部 (标量)
    x,y,z — 虚部 (向量)
    i,j,k — 虚数单位，满足 i²=j²=k²=ijk=-1

  单位四元数 (|q|=1) 表示三维旋转:
    q = [cos(θ/2), sin(θ/2)·ax, sin(θ/2)·ay, sin(θ/2)·az]

  其中 (ax, ay, az) 是旋转轴 (单位向量)，θ 是旋转角度。

  MuJoCo 格式: [w, x, y, z]
  scipy 格式:  [x, y, z, w]
""")


# ============================================================
# 2. 常见旋转的四元数表示
# ============================================================
print(f"\n{DIVIDER}")
print("🎯 2. 常见旋转的四元数表示")
print(DIVIDER)


def angle_axis_to_quat(angle_deg, axis):
    """角度-轴 → MuJoCo 格式四元数 [w, x, y, z]"""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = np.radians(angle_deg) / 2.0
    w = np.cos(half)
    xyz = np.sin(half) * axis
    return np.array([w, xyz[0], xyz[1], xyz[2]])


common_rotations = [
    ("单位旋转 (无旋转)",          0,   [0, 0, 1]),
    ("绕 X 轴旋转 90°",          90,   [1, 0, 0]),
    ("绕 Y 轴旋转 90°",          90,   [0, 1, 0]),
    ("绕 Z 轴旋转 90°",          90,   [0, 0, 1]),
    ("绕 X 轴旋转 180°",        180,   [1, 0, 0]),
    ("绕 Y 轴旋转 180°",        180,   [0, 1, 0]),
    ("绕 Z 轴旋转 180°",        180,   [0, 0, 1]),
    ("绕 X 轴旋转 45°",          45,   [1, 0, 0]),
    ("绕 (1,1,0) 旋转 120°",   120,   [1, 1, 0]),
    ("绕 (1,1,1) 旋转 90°",     90,   [1, 1, 1]),
]

print(f"\n  {'旋转描述':<28} {'四元数 [w, x, y, z]':<36} {'|q|'}")
print(f"  {'-'*28} {'-'*36} {'-'*6}")

for desc, angle, axis in common_rotations:
    q = angle_axis_to_quat(angle, axis)
    norm = np.linalg.norm(q)
    q_str = f"[{q[0]:7.4f}, {q[1]:7.4f}, {q[2]:7.4f}, {q[3]:7.4f}]"
    print(f"  {desc:<28} {q_str:<36} {norm:.4f}")

print(f"""
  💡 记忆技巧:
    - 单位旋转 = [1, 0, 0, 0]  (w=1, 无虚部)
    - 180° 旋转: w=0, 虚部就是旋转轴
    - 90° 旋转: w = cos(45°) ≈ 0.7071, 虚部 = sin(45°)·轴 ≈ 0.7071·轴
""")


# ============================================================
# 3. 四元数乘法 (组合旋转)
# ============================================================
print(f"{DIVIDER}")
print("✖️  3. 四元数乘法 — 组合旋转")
print(DIVIDER)


def quat_multiply(q1, q2):
    """四元数乘法 (Hamilton 积): q1 ⊗ q2，MuJoCo [w,x,y,z] 格式。

    物理含义: 先做 q2 旋转，再做 q1 旋转。
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# 先绕 X 转 90°，再绕 Z 转 90°
qx90 = angle_axis_to_quat(90, [1, 0, 0])
qz90 = angle_axis_to_quat(90, [0, 0, 1])

q_combined = quat_multiply(qz90, qx90)  # 先 X 再 Z

print(f"\n  q_x90 = {qx90}")
print(f"  q_z90 = {qz90}")
print(f"  q_z90 ⊗ q_x90 (先绕X转90, 再绕Z转90) = {q_combined}")

# 用 MuJoCo 内置函数验证
q_mj = np.zeros(4)
mujoco.mju_mulQuat(q_mj, qz90, qx90)
print(f"  mju_mulQuat 验证 = {q_mj}")
print(f"  两者一致: {np.allclose(q_combined, q_mj)} ✅")

# 旋转顺序很重要！
q_reversed = quat_multiply(qx90, qz90)  # 先 Z 再 X
print(f"\n  q_x90 ⊗ q_z90 (先绕Z转90, 再绕X转90) = {q_reversed}")
print(f"  顺序不同结果不同: {not np.allclose(q_combined, q_reversed)} ✅ (旋转不可交换)")


# ============================================================
# 4. 四元数共轭与逆
# ============================================================
print(f"\n{DIVIDER}")
print("🔄 4. 四元数共轭与逆 — 反转旋转")
print(DIVIDER)


def quat_conjugate(q):
    """四元数共轭: 对于单位四元数，共轭 = 逆"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q):
    """四元数逆: q⁻¹ = q* / |q|²"""
    conj = quat_conjugate(q)
    return conj / np.dot(q, q)


q = angle_axis_to_quat(90, [1, 0, 0])
q_conj = quat_conjugate(q)
q_inv = quat_inverse(q)

print(f"\n  原始四元数 (绕X轴90°):  q     = {q}")
print(f"  共轭:                   q*    = {q_conj}")
print(f"  逆:                     q⁻¹   = {q_inv}")
print(f"  共轭 ≈ 逆 (单位四元数): {np.allclose(q_conj, q_inv)} ✅")

# q ⊗ q⁻¹ = 单位四元数
identity = quat_multiply(q, q_inv)
print(f"\n  q ⊗ q⁻¹ = {identity}")
print(f"  ≈ 单位四元数 [1,0,0,0]: {np.allclose(identity, [1, 0, 0, 0])} ✅")

# MuJoCo 内置: mju_negQuat 实际是共轭 (conjugate)，对应反向旋转
q_neg = np.zeros(4)
mujoco.mju_negQuat(q_neg, q)
print(f"\n  mju_negQuat(q)  = {q_neg}")
print(f"  注意: 名字虽叫 neg，实际是共轭 q* = [w, -x, -y, -z]（只有虚部取反）")
print(f"  与手动共轭一致: {np.allclose(q_neg, q_conj)} ✅")


# ============================================================
# 5. 四元数旋转向量
# ============================================================
print(f"\n{DIVIDER}")
print("🔀 5. 用四元数旋转一个向量")
print(DIVIDER)

v = np.array([1.0, 0.0, 0.0])
q_90z = angle_axis_to_quat(90, [0, 0, 1])

# 方法 1: v' = q ⊗ [0,v] ⊗ q*
v_quat = np.array([0, v[0], v[1], v[2]])
v_rotated_quat = quat_multiply(quat_multiply(q_90z, v_quat), quat_conjugate(q_90z))
v_rotated = v_rotated_quat[1:]

print(f"\n  原始向量: v = {v}")
print(f"  旋转四元数: q = {q_90z} (绕 Z 轴 90°)")
print(f"  旋转后: v' = {v_rotated}")
print(f"  期望: [0, 1, 0] (X 轴绕 Z 轴转 90° 变成 Y 轴)")
print(f"  正确: {np.allclose(v_rotated, [0, 1, 0])} ✅")

# 方法 2: MuJoCo 内置函数
v_mj = np.zeros(3)
mujoco.mju_rotVecQuat(v_mj, v, q_90z)
print(f"\n  mju_rotVecQuat 结果: {v_mj}")
print(f"  一致: {np.allclose(v_rotated, v_mj)} ✅")


# ============================================================
# 6. 四元数 ↔ 旋转矩阵
# ============================================================
print(f"\n{DIVIDER}")
print("📊 6. 四元数 ↔ 旋转矩阵 互转")
print(DIVIDER)


def quat_to_rotation_matrix(q):
    """MuJoCo 四元数 [w,x,y,z] → 3×3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


q = angle_axis_to_quat(90, [0, 0, 1])
R = quat_to_rotation_matrix(q)

print(f"\n  四元数 (绕Z轴90°): {q}")
print(f"  对应旋转矩阵:")
for row in R:
    print(f"    [{row[0]:7.4f}  {row[1]:7.4f}  {row[2]:7.4f}]")

# MuJoCo 内置
R_mj = np.zeros(9)
mujoco.mju_quat2Mat(R_mj, q)
R_mj = R_mj.reshape(3, 3)
print(f"\n  mju_quat2Mat 结果:")
for row in R_mj:
    print(f"    [{row[0]:7.4f}  {row[1]:7.4f}  {row[2]:7.4f}]")
print(f"  一致: {np.allclose(R, R_mj)} ✅")

# 反向: 旋转矩阵 → 四元数
q_back = np.zeros(4)
mujoco.mju_mat2Quat(q_back, R_mj.flatten())
print(f"\n  mju_mat2Quat 反转: {q_back}")
print(f"  与原始四元数一致: {np.allclose(q, q_back)} ✅")


# ============================================================
# 7. 四元数 ↔ 欧拉角 (需要 scipy)
# ============================================================
print(f"\n{DIVIDER}")
print("🔄 7. 四元数 ↔ 欧拉角转换")
print(DIVIDER)

if HAS_SCIPY:
    def mujoco_quat_to_euler(q_mj, seq="ZYX", degrees=True):
        """MuJoCo 四元数 [w,x,y,z] → 欧拉角 (默认 ZYX 内旋)"""
        q_scipy = np.array([q_mj[1], q_mj[2], q_mj[3], q_mj[0]])  # → [x,y,z,w]
        r = Rotation.from_quat(q_scipy)
        return r.as_euler(seq, degrees=degrees)

    def euler_to_mujoco_quat(euler_deg, seq="ZYX"):
        """欧拉角 → MuJoCo 四元数 [w,x,y,z]"""
        r = Rotation.from_euler(seq, euler_deg, degrees=True)
        q_scipy = r.as_quat()  # [x,y,z,w]
        return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

    test_cases = [
        ("单位旋转",              [1, 0, 0, 0]),
        ("绕 Z 轴 90°",         angle_axis_to_quat(90,  [0, 0, 1])),
        ("绕 X 轴 45°",         angle_axis_to_quat(45,  [1, 0, 0])),
        ("绕 Y 轴 -30°",        angle_axis_to_quat(-30, [0, 1, 0])),
        ("组合旋转 Z90+X45",     quat_multiply(
            angle_axis_to_quat(45, [1, 0, 0]),
            angle_axis_to_quat(90, [0, 0, 1]),
        )),
    ]

    print(f"\n  {'描述':<20} {'四元数 [w,x,y,z]':<36} {'欧拉角 ZYX (度)'}")
    print(f"  {'-'*20} {'-'*36} {'-'*20}")

    for desc, q in test_cases:
        q = np.array(q, dtype=float)
        euler = mujoco_quat_to_euler(q)
        q_str = f"[{q[0]:6.3f}, {q[1]:6.3f}, {q[2]:6.3f}, {q[3]:6.3f}]"
        e_str = f"[{euler[0]:7.2f}, {euler[1]:7.2f}, {euler[2]:7.2f}]"
        print(f"  {desc:<20} {q_str:<36} {e_str}")

    # 往返验证
    print(f"\n  往返验证 (euler → quat → euler):")
    original_euler = np.array([30.0, -15.0, 60.0])
    q_round = euler_to_mujoco_quat(original_euler)
    euler_back = mujoco_quat_to_euler(q_round)
    print(f"    原始欧拉角: {original_euler}")
    print(f"    → 四元数:   {q_round}")
    print(f"    → 欧拉角:   {euler_back}")
    print(f"    往返一致: {np.allclose(original_euler, euler_back)} ✅")

    print(f"""
  ⚠️  格式转换提醒:
    MuJoCo 四元数: [w, x, y, z]  ← w 在前
    scipy  四元数: [x, y, z, w]  ← w 在后

    转换代码:
      mj → scipy:  q_scipy = [q_mj[1], q_mj[2], q_mj[3], q_mj[0]]
      scipy → mj:  q_mj = [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]
""")
else:
    print("\n  ⏭️  scipy 未安装，跳过欧拉角转换。")
    print("  安装: pip install scipy")


# ============================================================
# 8. 四元数插值 (SLERP)
# ============================================================
print(f"{DIVIDER}")
print("🌀 8. 四元数球面线性插值 (SLERP)")
print(DIVIDER)

print("""
  SLERP = Spherical Linear intERPolation

  为什么不能对四元数做线性插值 (LERP)?
    - 线性插值后 |q| ≠ 1，不再是有效旋转
    - 即使归一化 (NLERP)，角速度也不均匀
    - SLERP 保证等角速度插值
""")


def slerp_quat(q1, q2, t):
    """球面线性插值: q1 和 q2 之间的 t ∈ [0, 1] 插值 (MuJoCo 格式)"""
    q1 = np.array(q1, dtype=float)
    q2 = np.array(q2, dtype=float)

    dot = np.dot(q1, q2)

    # q 和 -q 表示同一旋转，选近的那条路径
    if dot < 0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # 几乎平行，退化为线性插值 + 归一化
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2


q_start = angle_axis_to_quat(0, [0, 0, 1])    # 无旋转
q_end = angle_axis_to_quat(180, [0, 0, 1])     # 绕 Z 轴 180°

print(f"  起点: {q_start} (无旋转)")
print(f"  终点: {q_end} (绕 Z 轴 180°)")
print(f"\n  SLERP 插值过程:")
print(f"  {'t':>6} {'四元数 [w,x,y,z]':<36} {'|q|':>6} {'等效角度(°)':>10}")
print(f"  {'-'*6} {'-'*36} {'-'*6} {'-'*10}")

for t in np.linspace(0, 1, 11):
    q = slerp_quat(q_start, q_end, t)
    norm = np.linalg.norm(q)
    angle = 2 * np.degrees(np.arccos(np.clip(abs(q[0]), 0, 1)))
    q_str = f"[{q[0]:7.4f}, {q[1]:7.4f}, {q[2]:7.4f}, {q[3]:7.4f}]"
    print(f"  {t:6.2f} {q_str:<36} {norm:6.4f} {angle:10.1f}")

print(f"\n  ✅ 观察: |q| 始终 = 1，角度均匀增长 (等角速度)")

# 用 scipy 验证
if HAS_SCIPY:
    q_scipy_start = np.array([q_start[1], q_start[2], q_start[3], q_start[0]])
    q_scipy_end = np.array([q_end[1], q_end[2], q_end[3], q_end[0]])
    rots = Rotation.from_quat([q_scipy_start, q_scipy_end])
    slerp_fn = Slerp([0, 1], rots)
    q_scipy_mid = slerp_fn(0.5).as_quat()  # [x,y,z,w]
    q_mj_mid = np.array([q_scipy_mid[3], q_scipy_mid[0], q_scipy_mid[1], q_scipy_mid[2]])

    q_our_mid = slerp_quat(q_start, q_end, 0.5)
    print(f"\n  scipy Slerp(0.5) = {q_mj_mid}")
    print(f"  我们的 slerp(0.5)= {q_our_mid}")
    print(f"  一致: {np.allclose(q_mj_mid, q_our_mid) or np.allclose(q_mj_mid, -q_our_mid)} ✅")


# ============================================================
# 9. 归一化的重要性
# ============================================================
print(f"\n{DIVIDER}")
print("⚠️  9. 四元数归一化的重要性")
print(DIVIDER)

xml = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body name="box" pos="0 0 2">
      <joint name="free_jnt" type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
    <body name="floor">
      <geom type="plane" size="5 5 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)

# 实验 1: 正常的单位四元数
data_good = mujoco.MjData(model)
data_good.qpos[3:7] = [1, 0, 0, 0]

# 实验 2: 未归一化的四元数
data_bad = mujoco.MjData(model)
data_bad.qpos[3:7] = [2, 1, 1, 1]  # |q| = sqrt(7) ≈ 2.65

print(f"\n  正常四元数:   {data_good.qpos[3:7]}, |q| = {np.linalg.norm(data_good.qpos[3:7]):.4f}")
print(f"  异常四元数:   {data_bad.qpos[3:7]}, |q| = {np.linalg.norm(data_bad.qpos[3:7]):.4f}")

print(f"\n  仿真 100 步后比较位置:")

for i in range(100):
    mujoco.mj_step(model, data_good)
    mujoco.mj_step(model, data_bad)

print(f"    正常: pos = {data_good.qpos[:3]}, quat = {data_good.qpos[3:7]}")
print(f"    异常: pos = {data_bad.qpos[:3]}, quat = {data_bad.qpos[3:7]}")
print(f"    正常 |quat| = {np.linalg.norm(data_good.qpos[3:7]):.6f}")
print(f"    异常 |quat| = {np.linalg.norm(data_bad.qpos[3:7]):.6f}")

print(f"""
  💡 MuJoCo 会在积分时自动归一化四元数，但:
    1. 初始状态的非归一化四元数会导致第一帧计算异常
    2. 外部设置 qpos 时一定要自己归一化
    3. 数据管道中检测 |q| ≈ 1 是重要的质量校验

  归一化方法:
    q_normalized = q / np.linalg.norm(q)
    或 MuJoCo: mju_normalize4(q)
""")


def normalize_quat(q):
    """归一化四元数到单位长度"""
    q = np.array(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])  # 退化为单位四元数
    return q / norm


q_unnorm = np.array([2.0, 1.0, 1.0, 1.0])
q_norm = normalize_quat(q_unnorm)
print(f"  归一化示例: {q_unnorm} → {q_norm}, |q| = {np.linalg.norm(q_norm):.6f}")


# ============================================================
# 10. 四元数距离/差异
# ============================================================
print(f"\n{DIVIDER}")
print("📏 10. 两个旋转之间的距离")
print(DIVIDER)


def quat_distance(q1, q2):
    """计算两个四元数之间的角度距离 (弧度)。

    利用 |q1·q2| 越接近 1 表示越接近。
    返回 [0, π] 范围的角度。
    """
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, 0, 1)
    return 2.0 * np.arccos(dot)


def quat_difference(q1, q2):
    """计算从 q1 到 q2 的旋转差异: Δq = q2 ⊗ q1⁻¹"""
    return quat_multiply(q2, quat_inverse(q1))


q1 = angle_axis_to_quat(0, [0, 0, 1])
q2 = angle_axis_to_quat(30, [0, 0, 1])
q3 = angle_axis_to_quat(90, [0, 0, 1])
q4 = angle_axis_to_quat(180, [0, 0, 1])

pairs = [
    ("0° vs 30°",   q1, q2),
    ("0° vs 90°",   q1, q3),
    ("0° vs 180°",  q1, q4),
    ("30° vs 90°",  q2, q3),
    ("90° vs 180°", q3, q4),
]

print(f"\n  {'旋转对':<18} {'角度距离(°)':>12} {'差异四元数 [w,x,y,z]'}")
print(f"  {'-'*18} {'-'*12} {'-'*36}")

for desc, qa, qb in pairs:
    dist = np.degrees(quat_distance(qa, qb))
    diff = quat_difference(qa, qb)
    diff_str = f"[{diff[0]:7.4f}, {diff[1]:7.4f}, {diff[2]:7.4f}, {diff[3]:7.4f}]"
    print(f"  {desc:<18} {dist:12.2f} {diff_str}")


# ============================================================
# 11. MuJoCo 内置四元数工具函数一览
# ============================================================
print(f"\n{DIVIDER}")
print("🧰 11. MuJoCo 内置四元数工具函数一览")
print(DIVIDER)

print("""
  函数                       说明
  ────────────────────────  ────────────────────────────────
  mju_mulQuat(res, q1, q2)  四元数乘法 res = q1 ⊗ q2
  mju_negQuat(res, q)        四元数共轭 res = q* (反向旋转)
  mju_rotVecQuat(res, v, q)  用 q 旋转向量 v
  mju_quat2Mat(mat, q)       四元数 → 3×3 旋转矩阵 (行主序)
  mju_mat2Quat(q, mat)       3×3 旋转矩阵 → 四元数
  mju_axisAngle2Quat(q,a,θ)  轴角 → 四元数
  mju_quat2Vel(vel, q, dt)   四元数 → 角速度
  mju_subQuat(res, qa, qb)   四元数差 res = qa ⊖ qb (3维)
  mju_normalize4(q)           归一化四元数 (in-place)
""")

# 演示几个常用的
print("  演示 mju_axisAngle2Quat:")
q_test = np.zeros(4)
axis = np.array([0.0, 0.0, 1.0])
mujoco.mju_axisAngle2Quat(q_test, axis, np.radians(90))
print(f"    轴=[0,0,1], 角度=90° → q = {q_test}")
print(f"    与手动计算一致: {np.allclose(q_test, angle_axis_to_quat(90, [0,0,1]))} ✅")

print(f"\n  演示 mju_quat2Vel:")
vel = np.zeros(3)
q_small = angle_axis_to_quat(5, [1, 0, 0])  # 小角度旋转
mujoco.mju_quat2Vel(vel, q_small, 1.0)
print(f"    绕X轴5°旋转 → 角速度 ≈ {vel}")
print(f"    期望: ≈ [{np.radians(5):.4f}, 0, 0]")

print(f"\n  演示 mju_subQuat (三维旋转差):")
sub = np.zeros(3)
q_a = angle_axis_to_quat(90, [0, 0, 1])
q_b = angle_axis_to_quat(45, [0, 0, 1])
mujoco.mju_subQuat(sub, q_a, q_b)
print(f"    q_a(Z90°) ⊖ q_b(Z45°) = {sub}")
print(f"    这是一个 3 维向量，表示两个旋转之间的差异")


# ============================================================
# 12. 在仿真中使用四元数
# ============================================================
print(f"\n{DIVIDER}")
print("🎮 12. 在仿真中使用四元数")
print(DIVIDER)

sim_xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <body name="box" pos="0 0 0.5">
      <joint name="free" type="free"/>
      <geom type="box" size="0.2 0.1 0.05" rgba="0.2 0.6 1 1" mass="1"/>
    </body>
    <body name="floor">
      <geom type="plane" size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(sim_xml)
data = mujoco.MjData(model)

# 设置不同的初始旋转，观察效果
orientations = [
    ("无旋转",            [1, 0, 0, 0]),
    ("绕 Z 轴 45°",     angle_axis_to_quat(45, [0, 0, 1])),
    ("绕 X 轴 90°",     angle_axis_to_quat(90, [1, 0, 0])),
    ("任意旋转",          angle_axis_to_quat(60, [1, 1, 1])),
]

print(f"\n  设置不同初始朝向，执行 mj_forward 后检查几何体位置:\n")

for desc, quat in orientations:
    mujoco.mj_resetData(model, data)
    data.qpos[3:7] = quat
    mujoco.mj_forward(model, data)

    # 获取 box 体的世界朝向矩阵
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    xmat = data.xmat[body_id].reshape(3, 3)
    xpos = data.xpos[body_id]

    q_str = f"[{quat[0]:6.3f}, {quat[1]:6.3f}, {quat[2]:6.3f}, {quat[3]:6.3f}]"
    print(f"  {desc:<14} quat={q_str}")
    print(f"{'':>16} 世界位置: [{xpos[0]:.3f}, {xpos[1]:.3f}, {xpos[2]:.3f}]")
    print(f"{'':>16} X 轴朝向: [{xmat[0,0]:.3f}, {xmat[1,0]:.3f}, {xmat[2,0]:.3f}]")
    print(f"{'':>16} Z 轴朝向: [{xmat[0,2]:.3f}, {xmat[1,2]:.3f}, {xmat[2,2]:.3f}]")
    print()


# ============================================================
# 总结
# ============================================================
print(f"{DIVIDER}")
print("📝 本节总结")
print(DIVIDER)
print("""
  ┌──────────────────────────────────────────────────────────────┐
  │                    四元数速查                                  │
  │                                                              │
  │  MuJoCo 顺序: [w, x, y, z] ← 永远记住 w 在前                │
  │  scipy  顺序: [x, y, z, w] ← w 在后                         │
  │                                                              │
  │  单位旋转: [1, 0, 0, 0]                                     │
  │  绕轴 â 旋转 θ: [cos(θ/2), sin(θ/2)·â]                     │
  │                                                              │
  │  关键性质:                                                    │
  │    q 和 -q 表示同一旋转 (双覆盖)                              │
  │    必须保持 |q| = 1 (归一化)                                  │
  │    旋转不可交换: q1⊗q2 ≠ q2⊗q1                               │
  │                                                              │
  │  MuJoCo API:                                                 │
  │    mju_mulQuat      — 四元数乘法                              │
  │    mju_rotVecQuat   — 旋转向量                                │
  │    mju_quat2Mat     — 转旋转矩阵                              │
  │    mju_normalize4   — 归一化                                  │
  │    mju_subQuat      — 旋转差异                                │
  └──────────────────────────────────────────────────────────────┘
""")

print("✅ 第 02 节完成！")
