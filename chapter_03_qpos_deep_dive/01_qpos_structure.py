"""
第 3 章 · 01 - qpos 内存布局全解析

目标: 彻底搞懂 qpos 的每一个元素是什么意思，
     掌握从 qpos 索引反查关节名称的方法。

核心知识点:
  1. 每种关节类型对 qpos 的维度贡献
  2. jnt_qposadr / jnt_dofadr 索引映射
  3. nq != nv 的本质原因
  4. 反向查找: qpos[i] → 哪个关节的哪个分量

Java 类比:
  qpos 就像一个扁平的 byte[] 缓冲区，
  jnt_qposadr 就是各字段的 offset，
  你需要知道每个 offset 对应的字段名和长度。

运行: python 01_qpos_structure.py
"""

import mujoco
import numpy as np

DIVIDER = "=" * 65

# ============================================================
# 关节类型常量与元信息
# ============================================================
# MuJoCo 关节类型编号
JNT_FREE = 0
JNT_BALL = 1
JNT_SLIDE = 2
JNT_HINGE = 3

# 每种关节在 qpos 和 qvel 中占用的维度
JOINT_DIMS = {
    JNT_FREE:  {"name": "free",  "nq": 7, "nv": 6},
    JNT_BALL:  {"name": "ball",  "nq": 4, "nv": 3},
    JNT_SLIDE: {"name": "slide", "nq": 1, "nv": 1},
    JNT_HINGE: {"name": "hinge", "nq": 1, "nv": 1},
}

# qpos 各分量的语义标签
QPOS_LABELS = {
    JNT_FREE:  ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"],
    JNT_BALL:  ["quat_w", "quat_x", "quat_y", "quat_z"],
    JNT_SLIDE: ["displacement"],
    JNT_HINGE: ["angle_rad"],
}

QVEL_LABELS = {
    JNT_FREE:  ["vel_x", "vel_y", "vel_z", "angvel_x", "angvel_y", "angvel_z"],
    JNT_BALL:  ["angvel_x", "angvel_y", "angvel_z"],
    JNT_SLIDE: ["velocity"],
    JNT_HINGE: ["angular_vel"],
}


# ============================================================
# 1. 逐个关节类型分析
# ============================================================
print(DIVIDER)
print("🔍 1. 逐个分析每种关节类型对 qpos 的贡献")
print(DIVIDER)

# --- 1a. Free joint ---
print("\n📦 1a. Free Joint (自由关节)")
print("-" * 50)

free_xml = """
<mujoco>
  <worldbody>
    <body name="floating_box" pos="0 0 1">
      <joint name="free_jnt" type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(free_xml)
d = mujoco.MjData(m)

print(f"  nq = {m.nq}  (qpos 长度)")
print(f"  nv = {m.nv}  (qvel 长度)")
print(f"  qpos 默认值: {d.qpos}")
print(f"  各分量含义:")
for i, label in enumerate(QPOS_LABELS[JNT_FREE]):
    print(f"    qpos[{i}] = {d.qpos[i]:8.4f}  → {label}")

print(f"\n  解读:")
print(f"    位置部分 qpos[0:3] = [{d.qpos[0]}, {d.qpos[1]}, {d.qpos[2]}]")
print(f"      → 物体在世界坐标系中的 (x, y, z) 位置")
print(f"    旋转部分 qpos[3:7] = [{d.qpos[3]}, {d.qpos[4]}, {d.qpos[5]}, {d.qpos[6]}]")
print(f"      → 四元数 (w, x, y, z)，[1,0,0,0] 是单位旋转")

# --- 1b. Ball joint ---
print(f"\n📦 1b. Ball Joint (球铰关节)")
print("-" * 50)

ball_xml = """
<mujoco>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="ball_jnt" type="ball"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(ball_xml)
d = mujoco.MjData(m)

print(f"  nq = {m.nq}  (qpos 长度)")
print(f"  nv = {m.nv}  (qvel 长度)")
print(f"  qpos 默认值: {d.qpos}")
print(f"  各分量含义:")
for i, label in enumerate(QPOS_LABELS[JNT_BALL]):
    print(f"    qpos[{i}] = {d.qpos[i]:8.4f}  → {label}")
print(f"  ⚠️  注意: nq=4 但 nv=3，四元数 4 维 → 角速度 3 维")

# --- 1c. Hinge joint ---
print(f"\n📦 1c. Hinge Joint (铰链关节)")
print("-" * 50)

hinge_xml = """
<mujoco>
  <worldbody>
    <body name="door" pos="0 0 1">
      <joint name="hinge_jnt" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.3 0.02 0.4" mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(hinge_xml)
d = mujoco.MjData(m)

print(f"  nq = {m.nq}  (qpos 长度)")
print(f"  nv = {m.nv}  (qvel 长度)")
print(f"  qpos 默认值: {d.qpos}")
print(f"  各分量含义:")
for i, label in enumerate(QPOS_LABELS[JNT_HINGE]):
    print(f"    qpos[{i}] = {d.qpos[i]:8.4f}  → {label} (弧度)")
print(f"  ✅ nq=nv=1，铰链关节最简单")

# --- 1d. Slide joint ---
print(f"\n📦 1d. Slide Joint (滑动关节)")
print("-" * 50)

slide_xml = """
<mujoco>
  <worldbody>
    <body name="slider" pos="0 0 1">
      <joint name="slide_jnt" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(slide_xml)
d = mujoco.MjData(m)

print(f"  nq = {m.nq}  (qpos 长度)")
print(f"  nv = {m.nv}  (qvel 长度)")
print(f"  qpos 默认值: {d.qpos}")
print(f"  各分量含义:")
for i, label in enumerate(QPOS_LABELS[JNT_SLIDE]):
    print(f"    qpos[{i}] = {d.qpos[i]:8.4f}  → {label} (米)")
print(f"  ✅ nq=nv=1，滑动关节也很简单")


# ============================================================
# 2. 组合模型: 完整 qpos 布局
# ============================================================
print(f"\n{DIVIDER}")
print("🏗️  2. 组合模型 — 完整 qpos 内存布局")
print(DIVIDER)

combined_xml = """
<mujoco>
  <worldbody>
    <!-- 自由浮动基座 -->
    <body name="base" pos="0 0 1">
      <joint name="base_free" type="free"/>
      <geom type="box" size="0.2 0.2 0.05" mass="5"/>

      <!-- 肩部球铰 -->
      <body name="shoulder" pos="0.2 0 0">
        <joint name="shoulder_ball" type="ball"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0" mass="1"/>

        <!-- 肘部铰链 -->
        <body name="elbow" pos="0.3 0 0">
          <joint name="elbow_hinge" type="hinge" axis="0 1 0" range="-2.0 2.0" limited="true"/>
          <geom type="capsule" size="0.025" fromto="0 0 0 0.25 0 0" mass="0.8"/>

          <!-- 腕部铰链 -->
          <body name="wrist" pos="0.25 0 0">
            <joint name="wrist_hinge" type="hinge" axis="0 0 1" range="-1.57 1.57" limited="true"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.3"/>

            <!-- 手指滑动 -->
            <body name="finger" pos="0.1 0 0">
              <joint name="finger_slide" type="slide" axis="0 1 0" range="0 0.05" limited="true"/>
              <geom type="box" size="0.01 0.02 0.03" mass="0.1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(combined_xml)
data = mujoco.MjData(model)

print(f"\n  总维度: nq = {model.nq}, nv = {model.nv}")
print(f"  关节数: njnt = {model.njnt}")
print(f"  nq - nv = {model.nq - model.nv} (因为 free 和 ball 的四元数)")

print(f"\n  {'关节名':<18} {'类型':<8} {'qpos起始':>8} {'qpos长度':>8} {'qvel起始':>8} {'qvel长度':>8}")
print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for j in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
    jtype = model.jnt_type[j]
    info = JOINT_DIMS[jtype]
    qpos_adr = model.jnt_qposadr[j]
    dof_adr = model.jnt_dofadr[j]

    print(f"  {jname:<18} {info['name']:<8} {qpos_adr:>8} {info['nq']:>8} {dof_adr:>8} {info['nv']:>8}")

# 详细的 qpos 索引表
print(f"\n  📋 完整 qpos 索引映射表:")
print(f"  {'qpos[i]':>8} {'值':>10} {'关节名':<18} {'含义'}")
print(f"  {'-'*8} {'-'*10} {'-'*18} {'-'*20}")

for j in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
    jtype = model.jnt_type[j]
    qpos_adr = model.jnt_qposadr[j]
    labels = QPOS_LABELS[jtype]

    for k, label in enumerate(labels):
        idx = qpos_adr + k
        print(f"  qpos[{idx:>2}] {data.qpos[idx]:>10.4f} {jname:<18} {label}")

# qvel 索引表
print(f"\n  📋 完整 qvel 索引映射表:")
print(f"  {'qvel[i]':>8} {'值':>10} {'关节名':<18} {'含义'}")
print(f"  {'-'*8} {'-'*10} {'-'*18} {'-'*20}")

for j in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
    jtype = model.jnt_type[j]
    dof_adr = model.jnt_dofadr[j]
    labels = QVEL_LABELS[jtype]

    for k, label in enumerate(labels):
        idx = dof_adr + k
        print(f"  qvel[{idx:>2}] {data.qvel[idx]:>10.4f} {jname:<18} {label}")


# ============================================================
# 3. jnt_qposadr 和 jnt_dofadr 详解
# ============================================================
print(f"\n{DIVIDER}")
print("📍 3. jnt_qposadr 与 jnt_dofadr 详解")
print(DIVIDER)

print(f"\n  model.jnt_qposadr = {model.jnt_qposadr}")
print(f"  model.jnt_dofadr  = {model.jnt_dofadr}")

print(f"""
  jnt_qposadr[i] 告诉你: 关节 i 的数据从 qpos 的第几个元素开始
  jnt_dofadr[i]  告诉你: 关节 i 的数据从 qvel 的第几个元素开始

  举例 (当前模型):
    base_free (free):     qpos 从索引 {model.jnt_qposadr[0]} 开始, 占 7 个
    shoulder_ball (ball): qpos 从索引 {model.jnt_qposadr[1]} 开始, 占 4 个
    elbow_hinge (hinge):  qpos 从索引 {model.jnt_qposadr[2]} 开始, 占 1 个
    wrist_hinge (hinge):  qpos 从索引 {model.jnt_qposadr[3]} 开始, 占 1 个
    finger_slide (slide): qpos 从索引 {model.jnt_qposadr[4]} 开始, 占 1 个

  验证: {model.jnt_qposadr[0]} + 7 = {model.jnt_qposadr[1]} (base_free 结束 = shoulder_ball 开始) ✅
        {model.jnt_qposadr[1]} + 4 = {model.jnt_qposadr[2]} (shoulder_ball 结束 = elbow_hinge 开始) ✅
""")


# ============================================================
# 4. nq != nv — 为什么维度不同?
# ============================================================
print(f"{DIVIDER}")
print("⚠️  4. nq ≠ nv — 维度不匹配的根本原因")
print(DIVIDER)

print(f"""
  当前模型维度汇总:

  关节            qpos贡献  qvel贡献  差异
  ─────────────  ────────  ────────  ────
  base_free       7 (pos3+quat4)  6 (vel3+angvel3)  +1
  shoulder_ball   4 (quat4)       3 (angvel3)        +1
  elbow_hinge     1               1                   0
  wrist_hinge     1               1                   0
  finger_slide    1               1                   0
  ─────────────  ────────  ────────  ────
  总计            nq={model.nq:<4}       nv={model.nv:<4}        +{model.nq - model.nv}

  根本原因:
    - 旋转的"位置"用四元数表示: 4 个数 (w, x, y, z)
    - 旋转的"速度"用角速度表示: 3 个数 (ωx, ωy, ωz)
    - 四元数有归一化约束 (w² + x² + y² + z² = 1)，实际只有 3 个自由度
    - 所以每个四元数多占 1 维: nq 比 nv 多出来的就是四元数的个数

  公式: nq - nv = (free关节数 × 1) + (ball关节数 × 1)
         {model.nq} - {model.nv} = 1 (free) + 1 (ball) = {model.nq - model.nv} ✅
""")


# ============================================================
# 5. 反向查找: 给定 qpos 索引，找到关节和分量
# ============================================================
print(f"{DIVIDER}")
print("🔄 5. 反向查找函数: qpos 索引 → 关节信息")
print(DIVIDER)


def build_qpos_index_map(model):
    """构建 qpos 索引 → (关节名, 关节类型, 分量标签) 的映射表。

    返回值:
        dict[int, dict]: key 是 qpos 索引，value 包含:
            - joint_name: 关节名称
            - joint_id: 关节编号
            - joint_type: 关节类型名称
            - component: 分量语义标签 (如 "pos_x", "quat_w", "angle_rad")
            - offset: 该分量在关节内的偏移
    """
    index_map = {}
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        jtype = model.jnt_type[j]
        qpos_adr = model.jnt_qposadr[j]
        labels = QPOS_LABELS[jtype]

        for k, label in enumerate(labels):
            index_map[qpos_adr + k] = {
                "joint_name": jname,
                "joint_id": j,
                "joint_type": JOINT_DIMS[jtype]["name"],
                "component": label,
                "offset": k,
            }
    return index_map


def build_qvel_index_map(model):
    """构建 qvel 索引 → (关节名, 关节类型, 分量标签) 的映射表。"""
    index_map = {}
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        jtype = model.jnt_type[j]
        dof_adr = model.jnt_dofadr[j]
        labels = QVEL_LABELS[jtype]

        for k, label in enumerate(labels):
            index_map[dof_adr + k] = {
                "joint_name": jname,
                "joint_id": j,
                "joint_type": JOINT_DIMS[jtype]["name"],
                "component": label,
                "offset": k,
            }
    return index_map


def lookup_qpos(model, index):
    """查询 qpos[index] 属于哪个关节的哪个分量"""
    qmap = build_qpos_index_map(model)
    if index in qmap:
        info = qmap[index]
        return f"qpos[{index}] → 关节 '{info['joint_name']}' ({info['joint_type']}) 的 {info['component']}"
    return f"qpos[{index}] → 无效索引 (nq={model.nq})"


# 演示反向查找
print("\n  演示: 逐个查询每个 qpos 索引的含义\n")
for i in range(model.nq):
    print(f"    {lookup_qpos(model, i)}")

# 交互式查询示例
print(f"\n  完整索引映射字典 (可用于数据平台):")
qmap = build_qpos_index_map(model)
for idx in sorted(qmap.keys()):
    info = qmap[idx]
    print(f"    {idx:>2}: {{'joint': '{info['joint_name']:<16}', 'type': '{info['joint_type']:<6}', 'component': '{info['component']}'}}")


# ============================================================
# 6. 实用工具: 获取特定关节的 qpos 切片
# ============================================================
print(f"\n{DIVIDER}")
print("🛠️  6. 实用工具: 按关节名获取 qpos 切片")
print(DIVIDER)


def get_joint_qpos(model, data, joint_name):
    """获取指定关节在 qpos 中的值 (返回数组切片)"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"关节 '{joint_name}' 不存在")
    jtype = model.jnt_type[jid]
    adr = model.jnt_qposadr[jid]
    length = JOINT_DIMS[jtype]["nq"]
    return data.qpos[adr:adr + length]


def set_joint_qpos(model, data, joint_name, value):
    """设置指定关节在 qpos 中的值"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"关节 '{joint_name}' 不存在")
    jtype = model.jnt_type[jid]
    adr = model.jnt_qposadr[jid]
    length = JOINT_DIMS[jtype]["nq"]
    value = np.asarray(value)
    assert value.shape == (length,), \
        f"关节 '{joint_name}' ({JOINT_DIMS[jtype]['name']}) 需要 {length} 维, 得到 {value.shape}"
    data.qpos[adr:adr + length] = value


def get_joint_qvel(model, data, joint_name):
    """获取指定关节在 qvel 中的值"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"关节 '{joint_name}' 不存在")
    jtype = model.jnt_type[jid]
    adr = model.jnt_dofadr[jid]
    length = JOINT_DIMS[jtype]["nv"]
    return data.qvel[adr:adr + length]


# 演示
print("\n  按名称读取各关节的 qpos:")
joint_names = ["base_free", "shoulder_ball", "elbow_hinge", "wrist_hinge", "finger_slide"]
for name in joint_names:
    val = get_joint_qpos(model, data, name)
    print(f"    {name:<18} → {val}")

print("\n  修改 elbow_hinge 为 0.5 rad:")
set_joint_qpos(model, data, "elbow_hinge", np.array([0.5]))
print(f"    elbow_hinge qpos = {get_joint_qpos(model, data, 'elbow_hinge')}")

# 恢复默认
data.qpos[:] = model.qpos0


# ============================================================
# 7. 总结
# ============================================================
print(f"\n{DIVIDER}")
print("📝 7. 本节总结")
print(DIVIDER)
print("""
  ┌─────────────────────────────────────────────────────────┐
  │                    qpos 内存布局                         │
  │                                                         │
  │  关节类型    qpos维度   qvel维度   默认值                │
  │  ────────   ────────   ────────   ──────────            │
  │  free       7          6          [0,0,0, 1,0,0,0]     │
  │  ball       4          3          [1,0,0,0]             │
  │  hinge      1          1          0                     │
  │  slide      1          1          0                     │
  │                                                         │
  │  nq - nv = free关节数 + ball关节数                       │
  │                                                         │
  │  关键 API:                                              │
  │    model.jnt_qposadr[i] → 关节 i 在 qpos 中的起始位     │
  │    model.jnt_dofadr[i]  → 关节 i 在 qvel 中的起始位     │
  │    model.jnt_type[i]    → 关节 i 的类型 (0/1/2/3)      │
  └─────────────────────────────────────────────────────────┘
""")

print("✅ 第 01 节完成！")
