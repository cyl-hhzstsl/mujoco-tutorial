"""
第 2 章 · 01 - MJCF 结构全解析

目标: 加载一个 MJCF 模型，系统地打印出模型的每一个组成部分，
     深入理解 MJCF XML 各标签在编译后模型中的对应关系。

运行: python 01_mjcf_structure.py
"""

import os
import mujoco
import numpy as np

# ============================================================
# 加载模型
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "robot_arm.xml")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 前进一步以初始化所有量
mujoco.mj_forward(model, data)

DIVIDER = "=" * 65

# ============================================================
# 1. 基本信息
# ============================================================
print(DIVIDER)
print("📋 模型基本信息")
print(DIVIDER)
null_byte = b'\x00'
print(f"  模型名称:         {model.names[0:model.names.index(null_byte)].decode() if model.names[0:1] != null_byte else '(无名称)'}")
print(f"  自由度 (nv):      {model.nv}")
print(f"  广义坐标 (nq):    {model.nq}")
print(f"  刚体数 (nbody):   {model.nbody}")
print(f"  关节数 (njnt):    {model.njnt}")
print(f"  几何体数 (ngeom): {model.ngeom}")
print(f"  执行器数 (nu):    {model.nu}")
print(f"  传感器数 (nsensor):{model.nsensordata}")

# ============================================================
# 2. 编译器 / 选项设置
# ============================================================
print(f"\n{DIVIDER}")
print("⚙️  仿真选项 (option)")
print(DIVIDER)
print(f"  时间步长 (timestep):   {model.opt.timestep} s")
print(f"  重力 (gravity):        {model.opt.gravity}")

integrator_names = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
print(f"  积分器 (integrator):   {integrator_names.get(model.opt.integrator, model.opt.integrator)}")

cone_names = {0: "pyramidal", 1: "elliptic"}
print(f"  接触锥 (cone):         {cone_names.get(model.opt.cone, model.opt.cone)}")
print(f"  阻尼比 (impratio):     {model.opt.impratio}")

# ============================================================
# 3. 刚体层级树
# ============================================================
print(f"\n{DIVIDER}")
print("🌳 刚体层级树 (Body Hierarchy)")
print(DIVIDER)


def get_body_name(model, body_id):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name if name else f"body_{body_id}"


def print_body_tree(model, body_id, depth=0):
    """递归打印刚体树结构"""
    indent = "  │ " * depth
    connector = "  ├─" if depth > 0 else ""
    name = get_body_name(model, body_id)
    pos = model.body_pos[body_id]
    mass = model.body_mass[body_id]

    if depth == 0:
        print(f"  🌐 {name} (世界体)")
    else:
        print(f"{indent}{connector} 📦 {name}  pos={pos}  mass={mass:.3f}")

    # 收集该 body 的关节
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == body_id:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
            jtype_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
            jtype = jtype_names.get(model.jnt_type[j], str(model.jnt_type[j]))
            axis = model.jnt_axis[j]
            print(f"{indent}  │   🔗 关节: {jname}  type={jtype}  axis={axis}")

    # 收集该 body 的几何体
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id:
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
            gtype_names = {
                0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
                4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"
            }
            gtype = gtype_names.get(model.geom_type[g], str(model.geom_type[g]))
            gsize = model.geom_size[g]
            print(f"{indent}  │   🔵 几何: {gname}  type={gtype}  size={gsize}")

    # 收集该 body 的 site
    for s in range(model.nsite):
        if model.site_bodyid[s] == body_id:
            sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or f"site_{s}"
            spos = model.site_pos[s]
            print(f"{indent}  │   📍 站点: {sname}  pos={spos}")

    # 递归子体
    for child_id in range(model.nbody):
        if model.body_parentid[child_id] == body_id and child_id != body_id:
            print_body_tree(model, child_id, depth + 1)


print_body_tree(model, 0)

# ============================================================
# 4. 关节详情
# ============================================================
print(f"\n{DIVIDER}")
print("🔗 关节详情 (Joints)")
print(DIVIDER)
print(f"  {'名称':<16} {'类型':<8} {'轴':<18} {'范围':<20} {'阻尼':<8}")
print(f"  {'-'*16} {'-'*8} {'-'*18} {'-'*20} {'-'*8}")

jtype_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}

for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
    jtype = jtype_names.get(model.jnt_type[j], str(model.jnt_type[j]))
    axis = model.jnt_axis[j]
    limited = model.jnt_limited[j]
    jnt_range = model.jnt_range[j] if limited else [float('-inf'), float('inf')]
    damping = model.dof_damping[j] if j < model.nv else 0

    axis_str = f"[{axis[0]:5.2f} {axis[1]:5.2f} {axis[2]:5.2f}]"
    if limited:
        range_str = f"[{np.degrees(jnt_range[0]):7.1f}° ~ {np.degrees(jnt_range[1]):7.1f}°]"
    else:
        range_str = "[无限制]"

    print(f"  {name:<16} {jtype:<8} {axis_str:<18} {range_str:<20} {damping:<8.2f}")

# ============================================================
# 5. 几何体详情
# ============================================================
print(f"\n{DIVIDER}")
print("🔵 几何体详情 (Geoms)")
print(DIVIDER)

gtype_names = {
    0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
    4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"
}

print(f"  {'名称':<20} {'类型':<10} {'尺寸':<28} {'所属体':<12} {'碰撞'}")
print(f"  {'-'*20} {'-'*10} {'-'*28} {'-'*12} {'-'*10}")

for g in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
    gtype = gtype_names.get(model.geom_type[g], str(model.geom_type[g]))
    size = model.geom_size[g]
    body_id = model.geom_bodyid[g]
    body_name = get_body_name(model, body_id)
    contype = model.geom_contype[g]
    conaffinity = model.geom_conaffinity[g]

    size_str = f"[{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]"
    collision_str = f"type={contype} aff={conaffinity}"

    print(f"  {name:<20} {gtype:<10} {size_str:<28} {body_name:<12} {collision_str}")

# ============================================================
# 6. 执行器详情
# ============================================================
print(f"\n{DIVIDER}")
print("⚡ 执行器详情 (Actuators)")
print(DIVIDER)

for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
    trntype_names = {0: "joint", 1: "tendon", 2: "site", 3: "body"}
    trntype = trntype_names.get(model.actuator_trntype[i], str(model.actuator_trntype[i]))

    gain = model.actuator_gainprm[i]
    bias = model.actuator_biasprm[i]
    ctrlrange = model.actuator_ctrlrange[i]

    # 判断执行器类型
    if gain[0] != 0 and bias[1] != 0:
        act_type = "位置伺服 (position)"
    elif gain[0] != 0 and bias[2] != 0:
        act_type = "速度伺服 (velocity)"
    elif gain[0] != 0:
        act_type = "力矩电机 (motor)"
    else:
        act_type = "通用 (general)"

    print(f"\n  执行器 #{i}: {name}")
    print(f"    类型推断:     {act_type}")
    print(f"    传动类型:     {trntype}")
    print(f"    控制范围:     [{ctrlrange[0]:.2f}, {ctrlrange[1]:.2f}]")
    print(f"    增益参数 kp:  {gain[0]:.2f}")
    print(f"    偏置参数:     [{bias[0]:.2f}, {bias[1]:.2f}, {bias[2]:.2f}]")

# ============================================================
# 7. 传感器详情
# ============================================================
print(f"\n{DIVIDER}")
print("📡 传感器详情 (Sensors)")
print(DIVIDER)

sensor_type_names = {}
for attr in dir(mujoco):
    if attr.startswith("mjtSensor_"):
        continue
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

print(f"  {'名称':<22} {'类型':<18} {'维度':<6} {'当前值'}")
print(f"  {'-'*22} {'-'*18} {'-'*6} {'-'*30}")

for s in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"sensor_{s}"
    stype = sensor_type_map.get(model.sensor_type[s], f"type_{model.sensor_type[s]}")
    dim = model.sensor_dim[s]
    adr = model.sensor_adr[s]
    values = data.sensordata[adr:adr + dim]

    val_str = np.array2string(values, precision=4, separator=", ")
    print(f"  {name:<22} {stype:<18} {dim:<6} {val_str}")

# ============================================================
# 8. 当前状态
# ============================================================
print(f"\n{DIVIDER}")
print("📊 当前状态快照")
print(DIVIDER)
print(f"  仿真时间:  {data.time:.4f} s")
print(f"  qpos:      {data.qpos}")
print(f"  qvel:      {data.qvel}")
print(f"  ctrl:      {data.ctrl}")

# 末端执行器位置
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
if ee_site_id >= 0:
    ee_pos = data.site_xpos[ee_site_id]
    print(f"  末端位置:  [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")

print(f"\n{DIVIDER}")
print("✅ MJCF 结构解析完成！")
print(DIVIDER)
