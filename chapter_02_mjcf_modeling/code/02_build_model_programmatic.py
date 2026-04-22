"""
第 2 章 · 02 - 程序化构建模型

目标: 学会用 Python 动态创建和修改 MJCF 模型，而不是手写 XML。
     包括 XML 字符串方式和 MjSpec API（MuJoCo 3.x）。

运行: python 02_build_model_programmatic.py
"""

import mujoco
import numpy as np

DIVIDER = "=" * 65

# ============================================================
# 1. 从 XML 字符串创建模型
# ============================================================
print(DIVIDER)
print("📝 1. 从 XML 字符串创建模型")
print(DIVIDER)

xml_string = """
<mujoco model="from_string">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 2">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

print(f"  模型名称:    from_string")
print(f"  nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
print(f"  重力:        {model.opt.gravity}")
print(f"  时间步长:    {model.opt.timestep}")

# ============================================================
# 2. 修改仿真参数
# ============================================================
print(f"\n{DIVIDER}")
print("🔧 2. 运行时修改参数")
print(DIVIDER)

print("\n  --- 修改重力 ---")
gravities = [
    ("地球", [0, 0, -9.81]),
    ("月球", [0, 0, -1.62]),
    ("火星", [0, 0, -3.72]),
    ("零重力", [0, 0, 0]),
]

for name, g in gravities:
    model_copy = mujoco.MjModel.from_xml_string(xml_string)
    data_copy = mujoco.MjData(model_copy)

    model_copy.opt.gravity[:] = g
    data_copy.qpos[2] = 2.0  # 初始高度 2m

    for _ in range(500):
        mujoco.mj_step(model_copy, data_copy)

    t = 500 * model_copy.opt.timestep
    print(f"  {name:>6}: g={g}, {t:.1f}s 后高度 = {data_copy.qpos[2]:+.4f} m")

print("\n  --- 修改时间步长 ---")
timesteps = [0.001, 0.002, 0.005, 0.01]

for dt in timesteps:
    model_copy = mujoco.MjModel.from_xml_string(xml_string)
    data_copy = mujoco.MjData(model_copy)
    model_copy.opt.timestep = dt
    data_copy.qpos[2] = 2.0

    steps = int(1.0 / dt)
    for _ in range(steps):
        mujoco.mj_step(model_copy, data_copy)

    print(f"  dt={dt:.3f}s, {steps} 步后 (1.0s): 高度 = {data_copy.qpos[2]:+.6f} m")

print(f"\n  理论值 (1s 自由落体): h = 2.0 - 0.5*9.81*1² = {2.0 - 0.5*9.81:.4f} m")

# ============================================================
# 3. MjSpec API (MuJoCo 3.x) 或 XML 字符串操作
# ============================================================
print(f"\n{DIVIDER}")
print("🏗️  3. 程序化构建模型")
print(DIVIDER)

HAS_MJSPEC = hasattr(mujoco, "MjSpec")

if HAS_MJSPEC:
    print("  ✅ 检测到 MjSpec API (MuJoCo 3.x)")
    print()

    # 用 MjSpec 构建一个双摆
    spec = mujoco.MjSpec()
    spec.modelname = "spec_pendulum"

    spec.compiler.angle = "degree"
    spec.option.timestep = 0.002
    spec.option.gravity = (0, 0, -9.81)

    # 添加地面
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5, 5, 0.1]
    floor.rgba = [0.3, 0.3, 0.3, 1]

    # 添加支撑体
    mount = spec.worldbody.add_body()
    mount.name = "mount"
    mount.pos = [0, 0, 1.5]

    # 连杆1
    link1 = mount.add_body()
    link1.name = "link1"

    joint1 = link1.add_joint()
    joint1.name = "joint1"
    joint1.type = mujoco.mjtJoint.mjJNT_HINGE
    joint1.axis = [0, 1, 0]
    joint1.damping = 0.1

    geom1 = link1.add_geom()
    geom1.name = "link1_geom"
    geom1.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    geom1.size = [0.03, 0.25, 0]
    geom1.fromto = [0, 0, 0, 0, 0, -0.5]
    geom1.rgba = [0.2, 0.6, 1.0, 1]

    # 连杆2
    link2 = link1.add_body()
    link2.name = "link2"
    link2.pos = [0, 0, -0.5]

    joint2 = link2.add_joint()
    joint2.name = "joint2"
    joint2.type = mujoco.mjtJoint.mjJNT_HINGE
    joint2.axis = [0, 1, 0]
    joint2.damping = 0.1

    geom2 = link2.add_geom()
    geom2.name = "link2_geom"
    geom2.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    geom2.size = [0.025, 0.2, 0]
    geom2.fromto = [0, 0, 0, 0, 0, -0.4]
    geom2.rgba = [1.0, 0.4, 0.2, 1]

    # 编译
    spec_model = spec.compile()
    spec_data = mujoco.MjData(spec_model)

    print(f"  MjSpec 构建的模型:")
    print(f"    名称:   {spec.modelname}")
    print(f"    nbody:  {spec_model.nbody}")
    print(f"    njnt:   {spec_model.njnt}")
    print(f"    nq:     {spec_model.nq}")

    # 导出为 XML 验证
    xml_out = spec.to_xml()
    print(f"\n  导出的 XML (前 500 字符):")
    print("  " + xml_out[:500].replace("\n", "\n  "))

else:
    print("  ⚠️  MjSpec 不可用 (需要 MuJoCo >= 3.0)")
    print("  使用 XML 字符串拼接方式构建模型\n")

    def build_chain_xml(n_links, link_length=0.4, link_radius=0.03):
        """通过拼接 XML 字符串构建 N 连杆链"""
        xml = f'''<mujoco model="chain_{n_links}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint type="hinge" axis="0 1 0" damping="0.5"/>
    <geom type="capsule" size="{link_radius}" rgba="0.3 0.6 0.9 1"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="mount" pos="0 0 1.5">
      <geom type="cylinder" size="0.05 0.05" rgba="0.5 0.5 0.5 1" mass="0"/>
'''
        indent = "      "
        for i in range(n_links):
            pos_z = 0 if i == 0 else -link_length
            xml += f'{indent}<body name="link{i+1}" pos="0 0 {pos_z}">\n'
            xml += f'{indent}  <joint name="joint{i+1}" range="-180 180"/>\n'
            xml += f'{indent}  <geom name="link{i+1}_geom" fromto="0 0 0 0 0 {-link_length}" mass="{1.0/(i+1):.2f}"/>\n'
            indent += "  "

        # 添加末端 site
        xml += f'{indent}<site name="tip" pos="0 0 {-link_length}" size="0.02" rgba="1 0 0 1"/>\n'

        for i in range(n_links):
            indent = indent[:-2]
            xml += f'{indent}</body>\n'

        xml += '''    </body>
  </worldbody>
</mujoco>'''
        return xml

    print("  构建函数已定义: build_chain_xml(n_links)")
    print()

    # 展示生成的 XML 示例
    sample_xml = build_chain_xml(2)
    print("  === 2 连杆 XML 示例 ===")
    for line in sample_xml.split("\n")[:20]:
        print(f"  {line}")
    print("  ...")

# ============================================================
# 4. 不同连杆数的模型对比
# ============================================================
print(f"\n{DIVIDER}")
print("🔗 4. N 连杆链模型对比")
print(DIVIDER)


def make_chain_xml(n_links, link_length=0.4):
    """构建 N 连杆链的 XML"""
    xml = f'''<mujoco model="chain_{n_links}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="mount" pos="0 0 2">
      <geom type="cylinder" size="0.05 0.05" mass="0"/>
'''
    indent = "      "
    for i in range(n_links):
        pos_z = 0 if i == 0 else -link_length
        xml += f'{indent}<body name="link{i+1}" pos="0 0 {pos_z}">\n'
        xml += f'{indent}  <joint name="j{i+1}" type="hinge" axis="0 1 0" damping="0.3"/>\n'
        xml += f'{indent}  <geom type="capsule" size="0.025" fromto="0 0 0 0 0 {-link_length}" mass="{0.5:.1f}"/>\n'
        indent += "  "

    for i in range(n_links):
        indent = indent[:-2]
        xml += f'{indent}</body>\n'

    xml += '''    </body>
  </worldbody>
  <actuator>
'''
    for i in range(n_links):
        xml += f'    <motor name="motor{i+1}" joint="j{i+1}" ctrlrange="-5 5"/>\n'
    xml += '''  </actuator>
</mujoco>'''
    return xml


print(f"\n  {'连杆数':<8} {'nq':<6} {'nv':<6} {'nbody':<8} {'nu':<6} {'说明'}")
print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*20}")

for n in [1, 2, 3, 5, 7, 10]:
    xml = make_chain_xml(n)
    m = mujoco.MjModel.from_xml_string(xml)
    note = ""
    if n == 1:
        note = "单摆"
    elif n == 2:
        note = "双摆"
    elif n == 3:
        note = "类似3DOF机械臂"
    elif n >= 5:
        note = f"柔性链/蛇形"
    print(f"  {n:<8} {m.nq:<6} {m.nv:<6} {m.nbody:<8} {m.nu:<6} {note}")

# ============================================================
# 5. 仿真对比：不同链长的动力学行为
# ============================================================
print(f"\n{DIVIDER}")
print("📈 5. 自由摆动对比（初始水平释放）")
print(DIVIDER)

for n in [1, 2, 3, 5]:
    xml = make_chain_xml(n)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    d.qpos[:] = np.radians(90)

    for _ in range(1000):
        mujoco.mj_step(m, d)

    t = 1000 * m.opt.timestep
    angles_deg = np.degrees(d.qpos)
    max_vel = np.max(np.abs(d.qvel))
    energy = d.energy[0] + d.energy[1] if hasattr(d, 'energy') and len(d.energy) >= 2 else 0

    angle_str = ", ".join([f"{a:+.1f}°" for a in angles_deg[:min(3, n)]])
    if n > 3:
        angle_str += ", ..."

    print(f"\n  {n} 连杆链 ({t:.1f}s 后):")
    print(f"    关节角度: [{angle_str}]")
    print(f"    最大角速度: {max_vel:.2f} rad/s")

# ============================================================
# 6. 总结
# ============================================================
print(f"\n{DIVIDER}")
print("📌 关键知识点总结")
print(DIVIDER)
print("""
  1. MjModel.from_xml_string() — 从字符串创建模型
  2. MjModel.from_xml_path()   — 从文件创建模型
  3. model.opt.gravity/timestep — 运行时可修改仿真参数
  4. MjSpec API (3.x)          — 结构化构建和修改模型
  5. XML 字符串拼接             — 兼容所有版本的动态构建方式
  6. nq = 铰链关节数            — 每个 hinge 贡献 1 个 qpos
""")

print(f"{DIVIDER}")
print("✅ 程序化模型构建完成！")
print(DIVIDER)
