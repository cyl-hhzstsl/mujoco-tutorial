"""
第 2 章 · 03 - 几何体与碰撞详解

目标: 深入理解 MuJoCo 中所有几何体类型、碰撞检测机制、
     摩擦参数以及接触力的读取方式。

运行: python 03_geom_and_collision.py
"""

import mujoco
import numpy as np

DIVIDER = "=" * 65

# ============================================================
# 1. 所有几何体类型概览
# ============================================================
print(DIVIDER)
print("🔷 1. MuJoCo 几何体类型")
print(DIVIDER)

geom_types = {
    "plane": {
        "xml": '<geom name="ground" type="plane" size="5 5 0.1"/>',
        "说明": "无限平面，size 前两个分量定义可视范围，第三个是网格间距",
        "参数": "size='半长 半宽 网格间距'",
    },
    "sphere": {
        "xml": '<geom name="ball" type="sphere" size="0.1"/>',
        "说明": "球体，最简单的碰撞体",
        "参数": "size='半径'",
    },
    "capsule": {
        "xml": '<geom name="rod" type="capsule" size="0.03" fromto="0 0 0 0 0 0.5"/>',
        "说明": "胶囊体 = 圆柱 + 两端半球，机器人连杆首选",
        "参数": "size='半径' fromto='起点xyz 终点xyz' 或 size='半径 半长'",
    },
    "box": {
        "xml": '<geom name="cube" type="box" size="0.1 0.1 0.1"/>',
        "说明": "长方体，size 是半尺寸",
        "参数": "size='半长x 半长y 半长z'",
    },
    "cylinder": {
        "xml": '<geom name="can" type="cylinder" size="0.05 0.1"/>',
        "说明": "圆柱体",
        "参数": "size='半径 半高'",
    },
    "ellipsoid": {
        "xml": '<geom name="egg" type="ellipsoid" size="0.1 0.07 0.05"/>',
        "说明": "椭球体，三个半轴",
        "参数": "size='半轴x 半轴y 半轴z'",
    },
}

for gtype, info in geom_types.items():
    print(f"\n  📐 {gtype}")
    print(f"     {info['说明']}")
    print(f"     参数: {info['参数']}")
    print(f"     XML:  {info['xml']}")

# ============================================================
# 2. 创建含所有几何体的场景
# ============================================================
print(f"\n{DIVIDER}")
print("🎨 2. 多几何体场景")
print(DIVIDER)

scene_xml = """
<mujoco model="geom_showcase">
  <compiler angle="degree"/>
  <option timestep="0.001" gravity="0 0 -9.81"/>

  <default>
    <geom contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>
  </default>

  <asset>
    <material name="mat_red" rgba="0.9 0.2 0.2 1"/>
    <material name="mat_green" rgba="0.2 0.9 0.2 1"/>
    <material name="mat_blue" rgba="0.2 0.2 0.9 1"/>
    <material name="mat_yellow" rgba="0.9 0.9 0.2 1"/>
    <material name="mat_purple" rgba="0.7 0.2 0.9 1"/>
    <material name="mat_cyan" rgba="0.2 0.9 0.9 1"/>
  </asset>

  <worldbody>
    <light pos="0 -2 3" dir="0 1 -1"/>

    <!-- 地面 (plane) -->
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>

    <!-- 球体 (sphere) -->
    <body name="sphere_body" pos="-0.4 0 0.5">
      <joint type="free"/>
      <geom name="sphere1" type="sphere" size="0.08" material="mat_red" mass="1"/>
    </body>

    <!-- 盒子 (box) -->
    <body name="box_body" pos="-0.15 0 0.5">
      <joint type="free"/>
      <geom name="box1" type="box" size="0.06 0.06 0.06" material="mat_green" mass="1"/>
    </body>

    <!-- 胶囊体 (capsule) -->
    <body name="capsule_body" pos="0.1 0 0.5">
      <joint type="free"/>
      <geom name="capsule1" type="capsule" size="0.04 0.08" material="mat_blue" mass="1"/>
    </body>

    <!-- 圆柱体 (cylinder) -->
    <body name="cylinder_body" pos="0.35 0 0.5">
      <joint type="free"/>
      <geom name="cylinder1" type="cylinder" size="0.05 0.07" material="mat_yellow" mass="1"/>
    </body>

    <!-- 椭球体 (ellipsoid) -->
    <body name="ellipsoid_body" pos="0.6 0 0.5">
      <joint type="free"/>
      <geom name="ellipsoid1" type="ellipsoid" size="0.08 0.05 0.04" material="mat_purple" mass="1"/>
    </body>

    <!-- 第二层：高处落下 -->
    <body name="high_sphere" pos="0 0.2 1.5">
      <joint type="free"/>
      <geom name="sphere_high" type="sphere" size="0.1" material="mat_cyan" mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(scene_xml)
data = mujoco.MjData(model)

gtype_names = {
    0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"
}

print(f"\n  场景中的几何体:")
print(f"  {'名称':<16} {'类型':<12} {'尺寸':<24} {'质量体'}")
print(f"  {'-'*16} {'-'*12} {'-'*24} {'-'*12}")

for g in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
    gtype = gtype_names.get(model.geom_type[g], f"type_{model.geom_type[g]}")
    size = model.geom_size[g]
    body_id = model.geom_bodyid[g]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"

    size_str = f"[{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]"
    print(f"  {name:<16} {gtype:<12} {size_str:<24} {body_name}")

# ============================================================
# 3. 碰撞几何体 vs 视觉几何体
# ============================================================
print(f"\n{DIVIDER}")
print("👁️  3. 碰撞几何体 vs 视觉几何体 (contype / conaffinity)")
print(DIVIDER)

collision_xml = """
<mujoco model="collision_demo">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"
          contype="1" conaffinity="1"/>

    <!-- 碰撞体：会与地面发生碰撞 -->
    <body name="collider" pos="0 0 1">
      <joint type="free"/>
      <geom name="col_geom" type="sphere" size="0.1"
            contype="1" conaffinity="1"
            rgba="1 0 0 1" mass="1"/>
    </body>

    <!-- 幽灵体：穿过地面（视觉用） -->
    <body name="ghost" pos="0.5 0 1">
      <joint type="free"/>
      <geom name="ghost_geom" type="sphere" size="0.1"
            contype="0" conaffinity="0"
            rgba="0 0 1 0.3" mass="1"/>
    </body>

    <!-- 选择性碰撞：只与特定物体碰撞 -->
    <body name="selective" pos="-0.5 0 1">
      <joint type="free"/>
      <!-- contype=2 不匹配地面的 conaffinity=1 -->
      <geom name="sel_geom" type="sphere" size="0.1"
            contype="2" conaffinity="2"
            rgba="0 1 0 0.7" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

col_model = mujoco.MjModel.from_xml_string(collision_xml)
col_data = mujoco.MjData(col_model)

print("\n  碰撞规则: 两个几何体碰撞当且仅当")
print("    (geom1.contype & geom2.conaffinity) != 0  或")
print("    (geom2.contype & geom1.conaffinity) != 0")
print()
print(f"  {'几何体':<14} {'contype':<10} {'conaffinity':<14} {'与地面碰撞?'}")
print(f"  {'-'*14} {'-'*10} {'-'*14} {'-'*12}")

for g in range(col_model.ngeom):
    name = mujoco.mj_id2name(col_model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
    ct = col_model.geom_contype[g]
    ca = col_model.geom_conaffinity[g]
    floor_ct = col_model.geom_contype[0]
    floor_ca = col_model.geom_conaffinity[0]
    collides = bool((ct & floor_ca) or (floor_ct & ca))
    mark = "✅ 是" if collides else "❌ 否"
    print(f"  {name:<14} {ct:<10} {ca:<14} {mark}")

# 仿真验证
print("\n  仿真 1 秒后的高度:")
for _ in range(1000):
    mujoco.mj_step(col_model, col_data)

for body_name in ["collider", "ghost", "selective"]:
    bid = mujoco.mj_name2id(col_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    z = col_data.xpos[bid][2]
    status = "停在地面" if z > -0.5 else "穿过地面"
    print(f"    {body_name:<12}: z = {z:+.4f} m  ({status})")

# ============================================================
# 4. 摩擦参数
# ============================================================
print(f"\n{DIVIDER}")
print("🧊 4. 摩擦参数实验")
print(DIVIDER)

print("\n  MuJoCo 摩擦模型: friction='滑动 扭转 滚动'")
print("  - 滑动摩擦 (tangential): 阻止物体滑动")
print("  - 扭转摩擦 (torsional):  阻止物体绕法线旋转")
print("  - 滚动摩擦 (rolling):    阻止物体滚动")
print()

friction_xml_template = """
<mujoco model="friction_test">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <!-- 倾斜平面 (30度) -->
    <geom name="ramp" type="box" size="2 0.5 0.01" pos="0 0 0.5"
          euler="0 -30 0" friction="{floor_friction}" contype="1" conaffinity="1"
          rgba="0.4 0.4 0.4 1"/>

    <body name="slider" pos="-0.5 0 1.2">
      <joint type="free"/>
      <geom name="block" type="box" size="0.05 0.05 0.05"
            friction="{obj_friction}" mass="1"
            contype="1" conaffinity="1" rgba="1 0.3 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""

friction_configs = [
    ("冰面 (低摩擦)", "0.05 0.001 0.0001", "0.05 0.001 0.0001"),
    ("木材 (中摩擦)", "0.5 0.005 0.001", "0.5 0.005 0.001"),
    ("橡胶 (高摩擦)", "1.5 0.01 0.005", "1.5 0.01 0.005"),
    ("极高摩擦",      "3.0 0.1 0.01", "3.0 0.1 0.01"),
]

print(f"  {'配置':<18} {'1秒后X位移':>12} {'X速度':>10} {'状态'}")
print(f"  {'-'*18} {'-'*12} {'-'*10} {'-'*10}")

for name, floor_f, obj_f in friction_configs:
    xml = friction_xml_template.format(floor_friction=floor_f, obj_friction=obj_f)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    for _ in range(1000):
        mujoco.mj_step(m, d)

    dx = d.qpos[0] - (-0.5)
    vx = d.qvel[0]
    state = "滑动" if abs(vx) > 0.01 else "静止"
    print(f"  {name:<18} {dx:>+12.4f} m {vx:>+10.4f}  {state}")

# ============================================================
# 5. 接触检测与接触力
# ============================================================
print(f"\n{DIVIDER}")
print("💥 5. 接触检测与接触力")
print(DIVIDER)

# 用之前的多几何体场景
model = mujoco.MjModel.from_xml_string(scene_xml)
data = mujoco.MjData(model)

# 仿真到物体落地
for _ in range(2000):
    mujoco.mj_step(model, data)

print(f"\n  仿真 {2000 * model.opt.timestep:.1f}s 后的接触信息:")
print(f"  活跃接触数 (ncon): {data.ncon}")
print()

if data.ncon > 0:
    print(f"  {'#':<4} {'几何体1':<16} {'几何体2':<16} {'接触位置':<30} {'法向力':<10}")
    print(f"  {'-'*4} {'-'*16} {'-'*16} {'-'*30} {'-'*10}")

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
        pos = contact.pos
        dist = contact.dist

        # 获取接触力
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        normal_force = np.linalg.norm(force[:3])

        pos_str = f"[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
        print(f"  {i:<4} {geom1_name:<16} {geom2_name:<16} {pos_str:<30} {normal_force:>8.3f} N")

    # 总接触力统计
    total_force = 0
    for i in range(data.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        total_force += np.linalg.norm(force[:3])

    print(f"\n  总法向接触力: {total_force:.3f} N")

    # 计算总质量验证
    total_mass = sum(model.body_mass[b] for b in range(model.nbody))
    expected_force = total_mass * 9.81
    print(f"  总质量 × g:   {expected_force:.3f} N (理论值)")
else:
    print("  没有检测到接触")

# ============================================================
# 6. 接触参数调节
# ============================================================
print(f"\n{DIVIDER}")
print("🔧 6. 接触参数说明")
print(DIVIDER)
print("""
  MuJoCo 接触模型参数:
  ┌─────────────────────────────────────────────────────────┐
  │ solref = [timeconst, dampratio]                         │
  │   timeconst: 约束恢复时间常数 (越小越硬)                 │
  │   dampratio: 阻尼比 (1.0 = 临界阻尼)                    │
  │                                                         │
  │ solimp = [dmin, dmax, width, midpoint, power]           │
  │   定义阻抗函数的形状                                     │
  │                                                         │
  │ friction = [slide, spin, roll]                           │
  │   slide: 滑动摩擦系数                                    │
  │   spin:  扭转摩擦系数                                    │
  │   roll:  滚动摩擦系数                                    │
  │                                                         │
  │ contype / conaffinity: 碰撞过滤位掩码                    │
  │   碰撞条件: (A.contype & B.conaffinity) ||               │
  │             (B.contype & A.conaffinity)                  │
  │                                                         │
  │ condim: 接触维度                                         │
  │   1 = 仅法向 (无摩擦)                                    │
  │   3 = 法向 + 2个切向 (普通摩擦)                          │
  │   4 = 法向 + 2切向 + 扭转                                │
  │   6 = 法向 + 2切向 + 扭转 + 2滚动                       │
  └─────────────────────────────────────────────────────────┘
""")

# ============================================================
# 7. condim 对比实验
# ============================================================
print(f"{DIVIDER}")
print("🎯 7. condim 接触维度对比")
print(DIVIDER)

condim_xml_template = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" condim="{condim}" friction="0.5 0.01 0.005"/>
    <body pos="0 0 0.1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1" condim="{condim}" friction="0.5 0.01 0.005"/>
    </body>
  </worldbody>
</mujoco>
"""

condim_descs = {
    1: "仅法向 (无摩擦)",
    3: "法向 + 切向摩擦",
    4: "法向 + 切向 + 扭转",
    6: "完整 (含滚动)",
}

print(f"\n  初始条件: 球体在地面上方 0.1m，给予水平初速 2 m/s")
print()
print(f"  {'condim':<8} {'说明':<24} {'1秒后X位移':>12} {'X速度':>10}")
print(f"  {'-'*8} {'-'*24} {'-'*12} {'-'*10}")

for cd in [1, 3, 4, 6]:
    xml = condim_xml_template.format(condim=cd)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    d.qvel[0] = 2.0

    for _ in range(1000):
        mujoco.mj_step(m, d)

    dx = d.qpos[0]
    vx = d.qvel[0]
    print(f"  {cd:<8} {condim_descs[cd]:<24} {dx:>+12.4f} m {vx:>+10.4f}")

print(f"\n{DIVIDER}")
print("✅ 几何体与碰撞详解完成！")
print(DIVIDER)
