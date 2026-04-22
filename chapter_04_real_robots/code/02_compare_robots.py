"""
第 4 章 · 02 - 对比不同类型机器人

目标: 深入对比固定基座机械臂与移动机器人的 qpos 结构差异，
     理解 free joint 对数据维度的影响，生成可视化对比图。

核心知识点:
  1. 固定基座 vs 浮动基座的 qpos 布局
  2. 不同机器人类型的 DOF 对比
  3. 关节范围 (joint range) 的含义
  4. nq ≠ nv 的根本原因

运行: python 02_compare_robots.py
"""

import mujoco
import numpy as np
import os

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

JNT_TYPE_NAMES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
JNT_QPOS_DIM = {0: 7, 1: 4, 2: 1, 3: 1}
JNT_QVEL_DIM = {0: 6, 1: 3, 2: 1, 3: 1}


# ============================================================
# 内置简化模型
# ============================================================

# 固定基座 6DOF 机械臂（类似 UR5e）
UR5E_XML = """
<mujoco model="ur5e_simple">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.06 0.03" rgba="0.2 0.2 0.2 1"/>
      <body name="link1" pos="0 0 0.089">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
        <geom type="cylinder" size="0.06 0.06" rgba="0.3 0.5 0.8 1"/>
        <body name="link2" pos="0 0.138 0" euler="0 90 0">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.425 0 0" rgba="0.3 0.5 0.8 1"/>
          <body name="link3" pos="0.425 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom type="capsule" size="0.04" fromto="0 0 0 0.392 0 0" rgba="0.3 0.5 0.8 1"/>
            <body name="link4" pos="0.392 0 0.127">
              <joint name="wrist_1" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
              <geom type="cylinder" size="0.04 0.03" rgba="0.2 0.2 0.2 1"/>
              <body name="link5" pos="0 -0.1 0">
                <joint name="wrist_2" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
                <geom type="cylinder" size="0.04 0.03" rgba="0.2 0.2 0.2 1"/>
                <body name="link6" pos="0 0 -0.1">
                  <joint name="wrist_3" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
                  <geom type="cylinder" size="0.035 0.02" rgba="0.2 0.2 0.2 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act1" joint="shoulder_pan"  kp="1000"/>
    <position name="act2" joint="shoulder_lift" kp="1000"/>
    <position name="act3" joint="elbow"         kp="1000"/>
    <position name="act4" joint="wrist_1"       kp="500"/>
    <position name="act5" joint="wrist_2"       kp="500"/>
    <position name="act6" joint="wrist_3"       kp="500"/>
  </actuator>
</mujoco>
"""

# 固定基座 7DOF+2 手指（类似 Franka Panda）
FRANKA_XML = """
<mujoco model="franka_simple">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="link0" pos="0 0 0">
      <geom type="cylinder" size="0.06 0.03" rgba="0.9 0.9 0.9 1"/>
      <body name="link1" pos="0 0 0.333">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2.90 2.90"/>
        <geom type="cylinder" size="0.06 0.06" rgba="0.9 0.9 0.9 1"/>
        <body name="link2" pos="0 0 0" euler="0 0 -90">
          <joint name="j2" type="hinge" axis="0 0 1" range="-1.76 1.76"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0 -0.316 0" rgba="0.9 0.9 0.9 1"/>
          <body name="link3" pos="0 -0.316 0" euler="0 0 90">
            <joint name="j3" type="hinge" axis="0 0 1" range="-2.90 2.90"/>
            <geom type="cylinder" size="0.05 0.04" rgba="0.9 0.9 0.9 1"/>
            <body name="link4" pos="0.0825 0 0" euler="0 0 90">
              <joint name="j4" type="hinge" axis="0 0 1" range="-3.07 -0.07"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 -0.0825 0.384 0" rgba="0.9 0.9 0.9 1"/>
              <body name="link5" pos="-0.0825 0.384 0" euler="0 0 -90">
                <joint name="j5" type="hinge" axis="0 0 1" range="-2.90 2.90"/>
                <geom type="cylinder" size="0.05 0.03" rgba="0.9 0.9 0.9 1"/>
                <body name="link6" pos="0 0 0" euler="0 0 90">
                  <joint name="j6" type="hinge" axis="0 0 1" range="-0.02 3.75"/>
                  <geom type="cylinder" size="0.04 0.04" rgba="0.9 0.9 0.9 1"/>
                  <body name="link7" pos="0.088 0 0" euler="0 0 90">
                    <joint name="j7" type="hinge" axis="0 0 1" range="-2.90 2.90"/>
                    <geom type="cylinder" size="0.04 0.02" rgba="0.3 0.3 0.3 1"/>
                    <body name="hand" pos="0 0 0.107">
                      <body name="fl" pos="0 0.04 0.06">
                        <joint name="finger_l" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom type="box" size="0.01 0.01 0.03" rgba="0.8 0.8 0.8 1"/>
                      </body>
                      <body name="fr" pos="0 -0.04 0.06">
                        <joint name="finger_r" type="slide" axis="0 -1 0" range="0 0.04"/>
                        <geom type="box" size="0.01 0.01 0.03" rgba="0.8 0.8 0.8 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="1000"/>
    <position name="a2" joint="j2" kp="1000"/>
    <position name="a3" joint="j3" kp="1000"/>
    <position name="a4" joint="j4" kp="1000"/>
    <position name="a5" joint="j5" kp="500"/>
    <position name="a6" joint="j6" kp="500"/>
    <position name="a7" joint="j7" kp="500"/>
    <position name="af1" joint="finger_l" kp="100"/>
    <position name="af2" joint="finger_r" kp="100"/>
  </actuator>
</mujoco>
"""

# 四足机器人（类似 Go2）—— 浮动基座 + 12 关节
GO2_XML = """
<mujoco model="go2_simple">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="trunk" pos="0 0 0.35">
      <freejoint name="float_base"/>
      <geom type="box" size="0.2 0.08 0.05" rgba="0.1 0.1 0.1 1" mass="6"/>
      <body name="FL_hip" pos="0.19 0.05 0">
        <joint name="FL_hip_j" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0.08 0" rgba="0.8 0.2 0.2 1"/>
        <body name="FL_thigh" pos="0 0.08 0">
          <joint name="FL_thigh_j" type="hinge" axis="0 1 0" range="-0.69 4.50"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.2 0.8 1"/>
          <body name="FL_calf" pos="0 0 -0.213">
            <joint name="FL_calf_j" type="hinge" axis="0 1 0" range="-2.77 -0.61"/>
            <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.8 0.2 1"/>
          </body>
        </body>
      </body>
      <body name="FR_hip" pos="0.19 -0.05 0">
        <joint name="FR_hip_j" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 -0.08 0" rgba="0.8 0.2 0.2 1"/>
        <body name="FR_thigh" pos="0 -0.08 0">
          <joint name="FR_thigh_j" type="hinge" axis="0 1 0" range="-0.69 4.50"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.2 0.8 1"/>
          <body name="FR_calf" pos="0 0 -0.213">
            <joint name="FR_calf_j" type="hinge" axis="0 1 0" range="-2.77 -0.61"/>
            <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.8 0.2 1"/>
          </body>
        </body>
      </body>
      <body name="RL_hip" pos="-0.19 0.05 0">
        <joint name="RL_hip_j" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0.08 0" rgba="0.8 0.2 0.2 1"/>
        <body name="RL_thigh" pos="0 0.08 0">
          <joint name="RL_thigh_j" type="hinge" axis="0 1 0" range="-0.69 4.50"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.2 0.8 1"/>
          <body name="RL_calf" pos="0 0 -0.213">
            <joint name="RL_calf_j" type="hinge" axis="0 1 0" range="-2.77 -0.61"/>
            <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.8 0.2 1"/>
          </body>
        </body>
      </body>
      <body name="RR_hip" pos="-0.19 -0.05 0">
        <joint name="RR_hip_j" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 -0.08 0" rgba="0.8 0.2 0.2 1"/>
        <body name="RR_thigh" pos="0 -0.08 0">
          <joint name="RR_thigh_j" type="hinge" axis="0 1 0" range="-0.69 4.50"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.2 0.8 1"/>
          <body name="RR_calf" pos="0 0 -0.213">
            <joint name="RR_calf_j" type="hinge" axis="0 1 0" range="-2.77 -0.61"/>
            <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.8 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="FL_hip_a"   joint="FL_hip_j"   gear="25"/>
    <motor name="FL_thigh_a" joint="FL_thigh_j" gear="25"/>
    <motor name="FL_calf_a"  joint="FL_calf_j"  gear="25"/>
    <motor name="FR_hip_a"   joint="FR_hip_j"   gear="25"/>
    <motor name="FR_thigh_a" joint="FR_thigh_j" gear="25"/>
    <motor name="FR_calf_a"  joint="FR_calf_j"  gear="25"/>
    <motor name="RL_hip_a"   joint="RL_hip_j"   gear="25"/>
    <motor name="RL_thigh_a" joint="RL_thigh_j" gear="25"/>
    <motor name="RL_calf_a"  joint="RL_calf_j"  gear="25"/>
    <motor name="RR_hip_a"   joint="RR_hip_j"   gear="25"/>
    <motor name="RR_thigh_a" joint="RR_thigh_j" gear="25"/>
    <motor name="RR_calf_a"  joint="RR_calf_j"  gear="25"/>
  </actuator>
</mujoco>
"""

# 人形机器人（类似 H1）—— 浮动基座 + 19 关节
H1_XML = """
<mujoco model="h1_simple">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="pelvis" pos="0 0 1.0">
      <freejoint name="float_base"/>
      <geom type="box" size="0.1 0.12 0.1" rgba="0.2 0.2 0.2 1" mass="10"/>
      <body name="torso" pos="0 0 0.2">
        <joint name="torso_j" type="hinge" axis="0 0 1" range="-2.35 2.35"/>
        <geom type="box" size="0.08 0.1 0.15" rgba="0.3 0.3 0.3 1" mass="8"/>
      </body>
      <body name="l_hip_yaw" pos="0 0.1 0">
        <joint name="l_hip_yaw_j" type="hinge" axis="0 0 1" range="-0.43 0.43"/>
        <body name="l_hip_roll" pos="0 0 0">
          <joint name="l_hip_roll_j" type="hinge" axis="1 0 0" range="-0.43 0.43"/>
          <body name="l_hip_pitch" pos="0 0.05 -0.05">
            <joint name="l_hip_pitch_j" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
            <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" rgba="0.2 0.2 0.8 1"/>
            <body name="l_knee" pos="0 0 -0.4">
              <joint name="l_knee_j" type="hinge" axis="0 1 0" range="-0.26 2.05"/>
              <geom type="capsule" size="0.035" fromto="0 0 0 0 0 -0.4" rgba="0.2 0.8 0.2 1"/>
              <body name="l_ankle" pos="0 0 -0.4">
                <joint name="l_ankle_j" type="hinge" axis="0 1 0" range="-0.87 0.52"/>
                <geom type="box" size="0.1 0.04 0.02" pos="0.03 0 -0.02" rgba="0.5 0.5 0.5 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="r_hip_yaw" pos="0 -0.1 0">
        <joint name="r_hip_yaw_j" type="hinge" axis="0 0 1" range="-0.43 0.43"/>
        <body name="r_hip_roll" pos="0 0 0">
          <joint name="r_hip_roll_j" type="hinge" axis="1 0 0" range="-0.43 0.43"/>
          <body name="r_hip_pitch" pos="0 -0.05 -0.05">
            <joint name="r_hip_pitch_j" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
            <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" rgba="0.2 0.2 0.8 1"/>
            <body name="r_knee" pos="0 0 -0.4">
              <joint name="r_knee_j" type="hinge" axis="0 1 0" range="-0.26 2.05"/>
              <geom type="capsule" size="0.035" fromto="0 0 0 0 0 -0.4" rgba="0.2 0.8 0.2 1"/>
              <body name="r_ankle" pos="0 0 -0.4">
                <joint name="r_ankle_j" type="hinge" axis="0 1 0" range="-0.87 0.52"/>
                <geom type="box" size="0.1 0.04 0.02" pos="0.03 0 -0.02" rgba="0.5 0.5 0.5 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="l_shoulder" pos="0 0.15 0.3">
        <joint name="l_sh_pitch_j" type="hinge" axis="0 1 0" range="-2.87 2.87"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0.05 -0.25" rgba="0.6 0.6 0.6 1"/>
        <body name="l_elbow" pos="0 0.05 -0.25">
          <joint name="l_sh_roll_j" type="hinge" axis="1 0 0" range="-0.34 3.11"/>
          <body name="l_forearm" pos="0 0 0">
            <joint name="l_sh_yaw_j" type="hinge" axis="0 0 1" range="-1.3 4.45"/>
            <body name="l_wrist" pos="0 0 -0.25">
              <joint name="l_elbow_j" type="hinge" axis="0 1 0" range="-1.25 2.61"/>
              <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.2" rgba="0.6 0.6 0.6 1"/>
            </body>
          </body>
        </body>
      </body>
      <body name="r_shoulder" pos="0 -0.15 0.3">
        <joint name="r_sh_pitch_j" type="hinge" axis="0 1 0" range="-2.87 2.87"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0 -0.05 -0.25" rgba="0.6 0.6 0.6 1"/>
        <body name="r_elbow" pos="0 -0.05 -0.25">
          <joint name="r_sh_roll_j" type="hinge" axis="1 0 0" range="-3.11 0.34"/>
          <body name="r_forearm" pos="0 0 0">
            <joint name="r_sh_yaw_j" type="hinge" axis="0 0 1" range="-4.45 1.3"/>
            <body name="r_wrist" pos="0 0 -0.25">
              <joint name="r_elbow_j" type="hinge" axis="0 1 0" range="-1.25 2.61"/>
              <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.2" rgba="0.6 0.6 0.6 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="torso_a"       joint="torso_j"       gear="200"/>
    <motor name="l_hip_yaw_a"   joint="l_hip_yaw_j"   gear="100"/>
    <motor name="l_hip_roll_a"  joint="l_hip_roll_j"   gear="100"/>
    <motor name="l_hip_pitch_a" joint="l_hip_pitch_j"  gear="200"/>
    <motor name="l_knee_a"      joint="l_knee_j"       gear="200"/>
    <motor name="l_ankle_a"     joint="l_ankle_j"      gear="50"/>
    <motor name="r_hip_yaw_a"   joint="r_hip_yaw_j"   gear="100"/>
    <motor name="r_hip_roll_a"  joint="r_hip_roll_j"   gear="100"/>
    <motor name="r_hip_pitch_a" joint="r_hip_pitch_j"  gear="200"/>
    <motor name="r_knee_a"      joint="r_knee_j"       gear="200"/>
    <motor name="r_ankle_a"     joint="r_ankle_j"      gear="50"/>
    <motor name="l_sh_pitch_a"  joint="l_sh_pitch_j"   gear="50"/>
    <motor name="l_sh_roll_a"   joint="l_sh_roll_j"    gear="50"/>
    <motor name="l_sh_yaw_a"    joint="l_sh_yaw_j"     gear="50"/>
    <motor name="l_elbow_a"     joint="l_elbow_j"      gear="50"/>
    <motor name="r_sh_pitch_a"  joint="r_sh_pitch_j"   gear="50"/>
    <motor name="r_sh_roll_a"   joint="r_sh_roll_j"    gear="50"/>
    <motor name="r_sh_yaw_a"    joint="r_sh_yaw_j"     gear="50"/>
    <motor name="r_elbow_a"     joint="r_elbow_j"      gear="50"/>
  </actuator>
</mujoco>
"""


# ============================================================
# 工具函数
# ============================================================

def load_model(xml_string):
    return mujoco.MjModel.from_xml_string(xml_string)


def get_robot_info(model, name, category):
    """提取机器人关键信息。"""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    has_free = False
    joint_types = {}
    joint_ranges = []

    for j in range(model.njnt):
        jtype = int(model.jnt_type[j])
        tname = JNT_TYPE_NAMES.get(jtype, "?")
        joint_types[tname] = joint_types.get(tname, 0) + 1
        if jtype == 0:
            has_free = True
        if model.jnt_limited[j]:
            lo = model.jnt_range[j][0]
            hi = model.jnt_range[j][1]
            joint_ranges.append((model.joint(j).name, np.degrees(lo), np.degrees(hi)))

    return {
        "name": name,
        "category": category,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "njnt": model.njnt,
        "nbody": model.nbody,
        "has_free_joint": has_free,
        "joint_types": joint_types,
        "joint_ranges": joint_ranges,
        "qpos0": data.qpos.copy(),
    }


# ============================================================
# 1. qpos 结构对比
# ============================================================
def compare_qpos_structure(robots):
    """对比各机器人的 qpos 布局。"""
    print(f"\n{DIVIDER}")
    print("  【对比 1】qpos 结构差异")
    print(DIVIDER)

    for r in robots:
        print(f"\n  ◆ {r['name']} ({r['category']})")
        print(f"    基座类型: {'浮动 (free joint)' if r['has_free_joint'] else '固定 (welded)'}")
        print(f"    nq={r['nq']:3d}  nv={r['nv']:3d}  nu={r['nu']:3d}  (nq-nv={r['nq']-r['nv']})")
        if r['has_free_joint']:
            print(f"    qpos 布局: [x,y,z,qw,qx,qy,qz | joint_angles...]")
            print(f"               ← 7 (free joint) → ← {r['nq']-7} (关节角) →")
        else:
            print(f"    qpos 布局: [joint_angles...]")
            print(f"               ← {r['nq']} (全部关节角/位移) →")

    # nq vs nv 表
    print(f"\n  {SUB_DIVIDER}")
    print(f"  为什么 nq ≠ nv ?")
    print(f"  {SUB_DIVIDER}")
    print(f"  • hinge/slide 关节: nq=1, nv=1  → 相等")
    print(f"  • ball 关节:        nq=4, nv=3  → 四元数(4) vs 角速度(3)")
    print(f"  • free 关节:        nq=7, nv=6  → pos+quat(7) vs vel+angvel(6)")
    print(f"  • 所以有 free joint 的机器人: nq = nv + 1")


# ============================================================
# 2. 自由度 (DOF) 对比
# ============================================================
def compare_dof(robots):
    """对比各机器人的自由度。"""
    print(f"\n{DIVIDER}")
    print("  【对比 2】自由度 (DOF) 对比")
    print(DIVIDER)

    print(f"\n  {'机器人':18s} {'类型':10s} {'总DOF(nv)':>9} {'base DOF':>9} {'关节DOF':>8} {'执行器':>6}")
    print(f"  {'-'*18} {'-'*10} {'-'*9} {'-'*9} {'-'*8} {'-'*6}")

    for r in robots:
        base_dof = 6 if r["has_free_joint"] else 0
        joint_dof = r["nv"] - base_dof
        print(f"  {r['name']:18s} {r['category']:10s} {r['nv']:9d} {base_dof:9d} {joint_dof:8d} {r['nu']:6d}")

    print(f"\n  关键观察:")
    fixed = [r for r in robots if not r["has_free_joint"]]
    mobile = [r for r in robots if r["has_free_joint"]]
    if fixed:
        print(f"    固定基座: {', '.join(r['name'] for r in fixed)}")
        print(f"      → base DOF = 0, 所有 DOF 都是可控关节")
    if mobile:
        print(f"    浮动基座: {', '.join(r['name'] for r in mobile)}")
        print(f"      → base DOF = 6 (3平移 + 3旋转), 这 6 个 DOF 无直接执行器")


# ============================================================
# 3. 关节类型分布
# ============================================================
def compare_joint_types(robots):
    """对比各机器人的关节类型分布。"""
    print(f"\n{DIVIDER}")
    print("  【对比 3】关节类型分布")
    print(DIVIDER)

    all_types = set()
    for r in robots:
        all_types.update(r["joint_types"].keys())
    all_types = sorted(all_types)

    header = f"  {'机器人':18s}"
    for t in all_types:
        header += f"  {t:>7s}"
    header += f"  {'总计':>5s}"
    print(f"\n{header}")
    print(f"  {'-'*18}" + "".join(f"  {'-'*7}" for _ in all_types) + f"  {'-'*5}")

    for r in robots:
        line = f"  {r['name']:18s}"
        total = 0
        for t in all_types:
            cnt = r["joint_types"].get(t, 0)
            total += cnt
            line += f"  {cnt:7d}"
        line += f"  {total:5d}"
        print(line)


# ============================================================
# 4. 关节范围
# ============================================================
def compare_joint_ranges(robots):
    """展示典型关节范围。"""
    print(f"\n{DIVIDER}")
    print("  【对比 4】典型关节范围 (度)")
    print(DIVIDER)

    for r in robots:
        print(f"\n  ◆ {r['name']}")
        if not r["joint_ranges"]:
            print(f"    (无限位关节)")
            continue
        shown = r["joint_ranges"][:6]
        for jname, lo, hi in shown:
            bar_lo = max(-180, lo)
            bar_hi = min(180, hi)
            range_deg = hi - lo
            print(f"    {jname:22s}  [{lo:7.1f}° ~ {hi:7.1f}°]  范围={range_deg:.1f}°")
        if len(r["joint_ranges"]) > 6:
            print(f"    ... 还有 {len(r['joint_ranges']) - 6} 个关节省略")


# ============================================================
# 5. 可视化（DOF 柱状图）
# ============================================================
def create_visualization(robots, output_path):
    """生成 DOF 对比柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  ⚠ matplotlib 未安装，跳过可视化")
        print("    安装方法: pip install matplotlib")
        return

    # 支持中文
    plt.rcParams["font.sans-serif"] = ["SimHei", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    names = [r["name"] for r in robots]
    nv_vals = [r["nv"] for r in robots]
    nu_vals = [r["nu"] for r in robots]
    base_dof = [6 if r["has_free_joint"] else 0 for r in robots]
    joint_dof = [r["nv"] - b for r, b in zip(robots, base_dof)]

    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 图 1: DOF 堆叠柱状图 ---
    ax1 = axes[0]
    bars_base = ax1.bar(x, base_dof, width * 2, label="Base DOF (free joint)",
                        color="#ff9999", edgecolor="black", linewidth=0.5)
    bars_joint = ax1.bar(x, joint_dof, width * 2, bottom=base_dof,
                         label="Joint DOF", color="#66b3ff", edgecolor="black", linewidth=0.5)

    for i, (b, j) in enumerate(zip(base_dof, joint_dof)):
        total = b + j
        ax1.text(i, total + 0.3, str(total), ha="center", va="bottom", fontweight="bold")

    ax1.set_xlabel("Robot")
    ax1.set_ylabel("DOF (nv)")
    ax1.set_title("DOF Breakdown: Base vs Joint")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # --- 图 2: nq / nv / nu 对比 ---
    ax2 = axes[1]
    nq_vals = [r["nq"] for r in robots]
    b1 = ax2.bar(x - width, nq_vals, width, label="nq (qpos dim)", color="#ff9999",
                 edgecolor="black", linewidth=0.5)
    b2 = ax2.bar(x, nv_vals, width, label="nv (qvel dim / DOF)", color="#66b3ff",
                 edgecolor="black", linewidth=0.5)
    b3 = ax2.bar(x + width, nu_vals, width, label="nu (ctrl dim)", color="#99ff99",
                 edgecolor="black", linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.2, str(int(h)),
                     ha="center", va="bottom", fontsize=8)

    ax2.set_xlabel("Robot")
    ax2.set_ylabel("Dimension")
    ax2.set_title("nq / nv / nu Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  ✅ 对比图已保存: {output_path}")
    plt.close()


# ============================================================
# 主程序
# ============================================================
def main():
    print(DIVIDER)
    print("  第 4 章 · 02 - 对比不同类型机器人")
    print(DIVIDER)

    # 加载所有模型
    models = [
        (UR5E_XML,   "UR5e",   "固定机械臂"),
        (FRANKA_XML, "Franka", "固定机械臂"),
        (GO2_XML,    "Go2",    "四足机器人"),
        (H1_XML,     "H1",     "人形机器人"),
    ]

    robots = []
    for xml, name, category in models:
        try:
            model = load_model(xml)
            info = get_robot_info(model, name, category)
            robots.append(info)
            print(f"  ✓ {name} 加载成功")
        except Exception as e:
            print(f"  ✗ {name} 加载失败: {e}")

    if not robots:
        print("❌ 无模型可供对比")
        return

    # 执行各项对比
    compare_qpos_structure(robots)
    compare_dof(robots)
    compare_joint_types(robots)
    compare_joint_ranges(robots)

    # 生成可视化
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "robot_comparison.png")
    create_visualization(robots, output_path)

    # 总结
    print(f"\n{DIVIDER}")
    print("  【总结】")
    print(DIVIDER)
    print(f"""
  1. 固定基座机械臂（UR5e, Franka）:
     • 没有 free joint，qpos 直接从关节角开始
     • nq == nv（都是 hinge/slide 关节）
     • 所有 DOF 都有对应的执行器

  2. 四足机器人（Go2）:
     • 有 free joint，qpos 前 7 位是基座位姿
     • nq = nv + 1（因为四元数 4 维 vs 角速度 3 维）
     • 基座的 6 DOF 没有直接执行器（靠腿的反力驱动）

  3. 人形机器人（H1）:
     • 同样有 free joint
     • DOF 最多，包含躯干、双腿、双臂
     • 控制维度最高，但基座 6 DOF 仍然是欠驱动的
    """)

    print(f"  ✅ 对比完成！下一步: python 03_trajectory_recording.py")
    print(DIVIDER)


if __name__ == "__main__":
    main()
