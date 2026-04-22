"""
第 4 章 · 01 - 加载真实机器人模型

目标: 学会从 robot_descriptions 库加载工业级机器人模型，
     逐个分析其关节结构、执行器配置和 qpos 布局。

核心知识点:
  1. robot_descriptions 库的使用方法
  2. 不同机器人的 nq/nv/nu 差异
  3. 关节类型映射与 qpos 索引
  4. 执行器与关节的绑定关系

运行: python 01_load_robots.py
"""

import mujoco
import numpy as np

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# ============================================================
# 关节类型常量
# ============================================================
JNT_TYPE_NAMES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
JNT_QPOS_DIM = {0: 7, 1: 4, 2: 1, 3: 1}
JNT_QVEL_DIM = {0: 6, 1: 3, 2: 1, 3: 1}


# ============================================================
# 内置回退模型（当 robot_descriptions 不可用时使用）
# ============================================================

FALLBACK_MODELS = {
    "UR5e (fallback)": """
    <mujoco model="ur5e_fallback">
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <body name="base_link" pos="0 0 0">
          <geom type="cylinder" size="0.06 0.03" rgba="0.2 0.2 0.2 1"/>
          <body name="shoulder_link" pos="0 0 0.089">
            <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
            <geom type="cylinder" size="0.06 0.06" rgba="0.3 0.5 0.8 1"/>
            <body name="upper_arm_link" pos="0 0.138 0" euler="0 90 0">
              <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0.425 0 0" rgba="0.3 0.5 0.8 1"/>
              <body name="forearm_link" pos="0.425 0 0">
                <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                <geom type="capsule" size="0.04" fromto="0 0 0 0.392 0 0" rgba="0.3 0.5 0.8 1"/>
                <body name="wrist_1_link" pos="0.392 0 0.127">
                  <joint name="wrist_1" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
                  <geom type="cylinder" size="0.04 0.03" rgba="0.2 0.2 0.2 1"/>
                  <body name="wrist_2_link" pos="0 -0.1 0">
                    <joint name="wrist_2" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
                    <geom type="cylinder" size="0.04 0.03" rgba="0.2 0.2 0.2 1"/>
                    <body name="wrist_3_link" pos="0 0 -0.1">
                      <joint name="wrist_3" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
                      <geom type="cylinder" size="0.035 0.02" rgba="0.2 0.2 0.2 1"/>
                      <site name="ee_site" pos="0 0 -0.05"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="shoulder_pan_act"  joint="shoulder_pan"  kp="1000"/>
        <position name="shoulder_lift_act" joint="shoulder_lift" kp="1000"/>
        <position name="elbow_act"         joint="elbow"         kp="1000"/>
        <position name="wrist_1_act"       joint="wrist_1"       kp="500"/>
        <position name="wrist_2_act"       joint="wrist_2"       kp="500"/>
        <position name="wrist_3_act"       joint="wrist_3"       kp="500"/>
      </actuator>
    </mujoco>
    """,

    "Franka Panda (fallback)": """
    <mujoco model="franka_panda_fallback">
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <body name="link0" pos="0 0 0">
          <geom type="cylinder" size="0.06 0.03" rgba="0.9 0.9 0.9 1"/>
          <body name="link1" pos="0 0 0.333">
            <joint name="joint1" type="hinge" axis="0 0 1" range="-2.90 2.90"/>
            <geom type="cylinder" size="0.06 0.06" rgba="0.9 0.9 0.9 1"/>
            <body name="link2" pos="0 0 0" euler="0 0 -90">
              <joint name="joint2" type="hinge" axis="0 0 1" range="-1.76 1.76"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0 -0.316 0" rgba="0.9 0.9 0.9 1"/>
              <body name="link3" pos="0 -0.316 0" euler="0 0 90">
                <joint name="joint3" type="hinge" axis="0 0 1" range="-2.90 2.90"/>
                <geom type="cylinder" size="0.05 0.04" rgba="0.9 0.9 0.9 1"/>
                <body name="link4" pos="0.0825 0 0" euler="0 0 90">
                  <joint name="joint4" type="hinge" axis="0 0 1" range="-3.07 -0.07"/>
                  <geom type="capsule" size="0.05" fromto="0 0 0 -0.0825 0.384 0" rgba="0.9 0.9 0.9 1"/>
                  <body name="link5" pos="-0.0825 0.384 0" euler="0 0 -90">
                    <joint name="joint5" type="hinge" axis="0 0 1" range="-2.90 2.90"/>
                    <geom type="cylinder" size="0.05 0.03" rgba="0.9 0.9 0.9 1"/>
                    <body name="link6" pos="0 0 0" euler="0 0 90">
                      <joint name="joint6" type="hinge" axis="0 0 1" range="-0.02 3.75"/>
                      <geom type="cylinder" size="0.04 0.04" rgba="0.9 0.9 0.9 1"/>
                      <body name="link7" pos="0.088 0 0" euler="0 0 90">
                        <joint name="joint7" type="hinge" axis="0 0 1" range="-2.90 2.90"/>
                        <geom type="cylinder" size="0.04 0.02" rgba="0.3 0.3 0.3 1"/>
                        <body name="hand" pos="0 0 0.107">
                          <geom type="box" size="0.02 0.04 0.04" rgba="0.3 0.3 0.3 1"/>
                          <body name="finger_left" pos="0 0.04 0.06">
                            <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
                            <geom type="box" size="0.01 0.01 0.03" rgba="0.8 0.8 0.8 1"/>
                          </body>
                          <body name="finger_right" pos="0 -0.04 0.06">
                            <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.04"/>
                            <geom type="box" size="0.01 0.01 0.03" rgba="0.8 0.8 0.8 1"/>
                          </body>
                        </body>
                        <site name="ee_site" pos="0 0 0.107"/>
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
        <position name="joint1_act" joint="joint1" kp="1000"/>
        <position name="joint2_act" joint="joint2" kp="1000"/>
        <position name="joint3_act" joint="joint3" kp="1000"/>
        <position name="joint4_act" joint="joint4" kp="1000"/>
        <position name="joint5_act" joint="joint5" kp="500"/>
        <position name="joint6_act" joint="joint6" kp="500"/>
        <position name="joint7_act" joint="joint7" kp="500"/>
        <position name="finger1_act" joint="finger_joint1" kp="100"/>
        <position name="finger2_act" joint="finger_joint2" kp="100"/>
      </actuator>
    </mujoco>
    """,

    "Unitree Go2 (fallback)": """
    <mujoco model="go2_fallback">
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
        <body name="trunk" pos="0 0 0.35">
          <freejoint name="float_base"/>
          <geom type="box" size="0.2 0.08 0.05" rgba="0.1 0.1 0.1 1" mass="6"/>
          <!-- 左前腿 (FL) -->
          <body name="FL_hip" pos="0.19 0.05 0">
            <joint name="FL_hip_joint" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 0.08 0" rgba="0.8 0.2 0.2 1"/>
            <body name="FL_thigh" pos="0 0.08 0">
              <joint name="FL_thigh_joint" type="hinge" axis="0 1 0" range="-0.69 4.50"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.2 0.8 1"/>
              <body name="FL_calf" pos="0 0 -0.213">
                <joint name="FL_calf_joint" type="hinge" axis="0 1 0" range="-2.77 -0.61"/>
                <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.8 0.2 1"/>
              </body>
            </body>
          </body>
          <!-- 右前腿 (FR) -->
          <body name="FR_hip" pos="0.19 -0.05 0">
            <joint name="FR_hip_joint" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 -0.08 0" rgba="0.8 0.2 0.2 1"/>
            <body name="FR_thigh" pos="0 -0.08 0">
              <joint name="FR_thigh_joint" type="hinge" axis="0 1 0" range="-0.69 4.50"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.2 0.8 1"/>
              <body name="FR_calf" pos="0 0 -0.213">
                <joint name="FR_calf_joint" type="hinge" axis="0 1 0" range="-2.77 -0.61"/>
                <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.8 0.2 1"/>
              </body>
            </body>
          </body>
          <!-- 左后腿 (RL) -->
          <body name="RL_hip" pos="-0.19 0.05 0">
            <joint name="RL_hip_joint" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 0.08 0" rgba="0.8 0.2 0.2 1"/>
            <body name="RL_thigh" pos="0 0.08 0">
              <joint name="RL_thigh_joint" type="hinge" axis="0 1 0" range="-0.69 4.50"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.2 0.8 1"/>
              <body name="RL_calf" pos="0 0 -0.213">
                <joint name="RL_calf_joint" type="hinge" axis="0 1 0" range="-2.77 -0.61"/>
                <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.8 0.2 1"/>
              </body>
            </body>
          </body>
          <!-- 右后腿 (RR) -->
          <body name="RR_hip" pos="-0.19 -0.05 0">
            <joint name="RR_hip_joint" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 -0.08 0" rgba="0.8 0.2 0.2 1"/>
            <body name="RR_thigh" pos="0 -0.08 0">
              <joint name="RR_thigh_joint" type="hinge" axis="0 1 0" range="-0.69 4.50"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.2 0.8 1"/>
              <body name="RR_calf" pos="0 0 -0.213">
                <joint name="RR_calf_joint" type="hinge" axis="0 1 0" range="-2.77 -0.61"/>
                <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" rgba="0.2 0.8 0.2 1"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="FL_hip_act"   joint="FL_hip_joint"   gear="25"/>
        <motor name="FL_thigh_act" joint="FL_thigh_joint" gear="25"/>
        <motor name="FL_calf_act"  joint="FL_calf_joint"  gear="25"/>
        <motor name="FR_hip_act"   joint="FR_hip_joint"   gear="25"/>
        <motor name="FR_thigh_act" joint="FR_thigh_joint" gear="25"/>
        <motor name="FR_calf_act"  joint="FR_calf_joint"  gear="25"/>
        <motor name="RL_hip_act"   joint="RL_hip_joint"   gear="25"/>
        <motor name="RL_thigh_act" joint="RL_thigh_joint" gear="25"/>
        <motor name="RL_calf_act"  joint="RL_calf_joint"  gear="25"/>
        <motor name="RR_hip_act"   joint="RR_hip_joint"   gear="25"/>
        <motor name="RR_thigh_act" joint="RR_thigh_joint" gear="25"/>
        <motor name="RR_calf_act"  joint="RR_calf_joint"  gear="25"/>
      </actuator>
    </mujoco>
    """,

    "Unitree H1 (fallback)": """
    <mujoco model="h1_fallback">
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
        <body name="pelvis" pos="0 0 1.0">
          <freejoint name="float_base"/>
          <geom type="box" size="0.1 0.12 0.1" rgba="0.2 0.2 0.2 1" mass="10"/>
          <!-- 躯干 -->
          <body name="torso" pos="0 0 0.2">
            <joint name="torso_joint" type="hinge" axis="0 0 1" range="-2.35 2.35"/>
            <geom type="box" size="0.08 0.1 0.15" rgba="0.3 0.3 0.3 1" mass="8"/>
          </body>
          <!-- 左腿 -->
          <body name="left_hip_yaw_link" pos="0 0.1 0">
            <joint name="left_hip_yaw" type="hinge" axis="0 0 1" range="-0.43 0.43"/>
            <geom type="sphere" size="0.05" rgba="0.8 0.2 0.2 1"/>
            <body name="left_hip_roll_link" pos="0 0 0">
              <joint name="left_hip_roll" type="hinge" axis="1 0 0" range="-0.43 0.43"/>
              <body name="left_hip_pitch_link" pos="0 0.05 -0.05">
                <joint name="left_hip_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" rgba="0.2 0.2 0.8 1"/>
                <body name="left_knee_link" pos="0 0 -0.4">
                  <joint name="left_knee" type="hinge" axis="0 1 0" range="-0.26 2.05"/>
                  <geom type="capsule" size="0.035" fromto="0 0 0 0 0 -0.4" rgba="0.2 0.8 0.2 1"/>
                  <body name="left_ankle_link" pos="0 0 -0.4">
                    <joint name="left_ankle" type="hinge" axis="0 1 0" range="-0.87 0.52"/>
                    <geom type="box" size="0.1 0.04 0.02" pos="0.03 0 -0.02" rgba="0.5 0.5 0.5 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
          <!-- 右腿 -->
          <body name="right_hip_yaw_link" pos="0 -0.1 0">
            <joint name="right_hip_yaw" type="hinge" axis="0 0 1" range="-0.43 0.43"/>
            <geom type="sphere" size="0.05" rgba="0.8 0.2 0.2 1"/>
            <body name="right_hip_roll_link" pos="0 0 0">
              <joint name="right_hip_roll" type="hinge" axis="1 0 0" range="-0.43 0.43"/>
              <body name="right_hip_pitch_link" pos="0 -0.05 -0.05">
                <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" rgba="0.2 0.2 0.8 1"/>
                <body name="right_knee_link" pos="0 0 -0.4">
                  <joint name="right_knee" type="hinge" axis="0 1 0" range="-0.26 2.05"/>
                  <geom type="capsule" size="0.035" fromto="0 0 0 0 0 -0.4" rgba="0.2 0.8 0.2 1"/>
                  <body name="right_ankle_link" pos="0 0 -0.4">
                    <joint name="right_ankle" type="hinge" axis="0 1 0" range="-0.87 0.52"/>
                    <geom type="box" size="0.1 0.04 0.02" pos="0.03 0 -0.02" rgba="0.5 0.5 0.5 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
          <!-- 左臂 (简化) -->
          <body name="left_shoulder_link" pos="0 0.15 0.3">
            <joint name="left_shoulder_pitch" type="hinge" axis="0 1 0" range="-2.87 2.87"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0 0.05 -0.25" rgba="0.6 0.6 0.6 1"/>
            <body name="left_elbow_link" pos="0 0.05 -0.25">
              <joint name="left_shoulder_roll" type="hinge" axis="1 0 0" range="-0.34 3.11"/>
              <body name="left_forearm_link" pos="0 0 0">
                <joint name="left_shoulder_yaw" type="hinge" axis="0 0 1" range="-1.3 4.45"/>
                <body name="left_wrist_link" pos="0 0 -0.25">
                  <joint name="left_elbow" type="hinge" axis="0 1 0" range="-1.25 2.61"/>
                  <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.2" rgba="0.6 0.6 0.6 1"/>
                </body>
              </body>
            </body>
          </body>
          <!-- 右臂 (简化) -->
          <body name="right_shoulder_link" pos="0 -0.15 0.3">
            <joint name="right_shoulder_pitch" type="hinge" axis="0 1 0" range="-2.87 2.87"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0 -0.05 -0.25" rgba="0.6 0.6 0.6 1"/>
            <body name="right_elbow_link" pos="0 -0.05 -0.25">
              <joint name="right_shoulder_roll" type="hinge" axis="1 0 0" range="-3.11 0.34"/>
              <body name="right_forearm_link" pos="0 0 0">
                <joint name="right_shoulder_yaw" type="hinge" axis="0 0 1" range="-4.45 1.3"/>
                <body name="right_wrist_link" pos="0 0 -0.25">
                  <joint name="right_elbow" type="hinge" axis="0 1 0" range="-1.25 2.61"/>
                  <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.2" rgba="0.6 0.6 0.6 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="torso_act"             joint="torso_joint"          gear="200"/>
        <motor name="left_hip_yaw_act"      joint="left_hip_yaw"        gear="100"/>
        <motor name="left_hip_roll_act"     joint="left_hip_roll"       gear="100"/>
        <motor name="left_hip_pitch_act"    joint="left_hip_pitch"      gear="200"/>
        <motor name="left_knee_act"         joint="left_knee"           gear="200"/>
        <motor name="left_ankle_act"        joint="left_ankle"          gear="50"/>
        <motor name="right_hip_yaw_act"     joint="right_hip_yaw"      gear="100"/>
        <motor name="right_hip_roll_act"    joint="right_hip_roll"      gear="100"/>
        <motor name="right_hip_pitch_act"   joint="right_hip_pitch"     gear="200"/>
        <motor name="right_knee_act"        joint="right_knee"          gear="200"/>
        <motor name="right_ankle_act"       joint="right_ankle"         gear="50"/>
        <motor name="left_shoulder_pitch_act"  joint="left_shoulder_pitch"  gear="50"/>
        <motor name="left_shoulder_roll_act"   joint="left_shoulder_roll"   gear="50"/>
        <motor name="left_shoulder_yaw_act"    joint="left_shoulder_yaw"    gear="50"/>
        <motor name="left_elbow_act"           joint="left_elbow"           gear="50"/>
        <motor name="right_shoulder_pitch_act" joint="right_shoulder_pitch" gear="50"/>
        <motor name="right_shoulder_roll_act"  joint="right_shoulder_roll"  gear="50"/>
        <motor name="right_shoulder_yaw_act"   joint="right_shoulder_yaw"   gear="50"/>
        <motor name="right_elbow_act"          joint="right_elbow"          gear="50"/>
      </actuator>
    </mujoco>
    """,

    "ALOHA (fallback)": """
    <mujoco model="aloha_fallback">
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <!-- 桌面 -->
        <geom type="box" size="0.4 0.3 0.01" pos="0 0 0.5" rgba="0.6 0.4 0.2 1"/>
        <!-- 左臂 -->
        <body name="left_base" pos="-0.2 0 0.51">
          <geom type="cylinder" size="0.04 0.02" rgba="0.3 0.3 0.3 1"/>
          <body name="left_link1" pos="0 0 0.04">
            <joint name="left_joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.1" rgba="0.9 0.9 0.9 1"/>
            <body name="left_link2" pos="0 0 0.1">
              <joint name="left_joint2" type="hinge" axis="0 1 0" range="-1.76 1.76"/>
              <geom type="capsule" size="0.025" fromto="0 0 0 0.15 0 0" rgba="0.9 0.9 0.9 1"/>
              <body name="left_link3" pos="0.15 0 0">
                <joint name="left_joint3" type="hinge" axis="0 1 0" range="-1.76 1.76"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0.15 0 0" rgba="0.9 0.9 0.9 1"/>
                <body name="left_link4" pos="0.15 0 0">
                  <joint name="left_joint4" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                  <geom type="capsule" size="0.015" fromto="0 0 0 0 0 0.08" rgba="0.9 0.9 0.9 1"/>
                  <body name="left_link5" pos="0 0 0.08">
                    <joint name="left_joint5" type="hinge" axis="0 1 0" range="-1.76 1.76"/>
                    <geom type="capsule" size="0.015" fromto="0 0 0 0 0 0.07" rgba="0.9 0.9 0.9 1"/>
                    <body name="left_link6" pos="0 0 0.07">
                      <joint name="left_joint6" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                      <geom type="cylinder" size="0.02 0.01" rgba="0.3 0.3 0.3 1"/>
                      <body name="left_gripper" pos="0 0 0.02">
                        <joint name="left_gripper_joint" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom type="box" size="0.01 0.005 0.02" pos="0 0.015 0" rgba="0.2 0.2 0.2 1"/>
                        <geom type="box" size="0.01 0.005 0.02" pos="0 -0.015 0" rgba="0.2 0.2 0.2 1"/>
                      </body>
                      <site name="left_ee" pos="0 0 0.03"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
        <!-- 右臂（镜像结构） -->
        <body name="right_base" pos="0.2 0 0.51">
          <geom type="cylinder" size="0.04 0.02" rgba="0.3 0.3 0.3 1"/>
          <body name="right_link1" pos="0 0 0.04">
            <joint name="right_joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.1" rgba="0.9 0.9 0.9 1"/>
            <body name="right_link2" pos="0 0 0.1">
              <joint name="right_joint2" type="hinge" axis="0 1 0" range="-1.76 1.76"/>
              <geom type="capsule" size="0.025" fromto="0 0 0 -0.15 0 0" rgba="0.9 0.9 0.9 1"/>
              <body name="right_link3" pos="-0.15 0 0">
                <joint name="right_joint3" type="hinge" axis="0 1 0" range="-1.76 1.76"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 -0.15 0 0" rgba="0.9 0.9 0.9 1"/>
                <body name="right_link4" pos="-0.15 0 0">
                  <joint name="right_joint4" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                  <geom type="capsule" size="0.015" fromto="0 0 0 0 0 0.08" rgba="0.9 0.9 0.9 1"/>
                  <body name="right_link5" pos="0 0 0.08">
                    <joint name="right_joint5" type="hinge" axis="0 1 0" range="-1.76 1.76"/>
                    <geom type="capsule" size="0.015" fromto="0 0 0 0 0 0.07" rgba="0.9 0.9 0.9 1"/>
                    <body name="right_link6" pos="0 0 0.07">
                      <joint name="right_joint6" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                      <geom type="cylinder" size="0.02 0.01" rgba="0.3 0.3 0.3 1"/>
                      <body name="right_gripper" pos="0 0 0.02">
                        <joint name="right_gripper_joint" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom type="box" size="0.01 0.005 0.02" pos="0 0.015 0" rgba="0.2 0.2 0.2 1"/>
                        <geom type="box" size="0.01 0.005 0.02" pos="0 -0.015 0" rgba="0.2 0.2 0.2 1"/>
                      </body>
                      <site name="right_ee" pos="0 0 0.03"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="left_j1_act"  joint="left_joint1"  kp="500"/>
        <position name="left_j2_act"  joint="left_joint2"  kp="500"/>
        <position name="left_j3_act"  joint="left_joint3"  kp="500"/>
        <position name="left_j4_act"  joint="left_joint4"  kp="300"/>
        <position name="left_j5_act"  joint="left_joint5"  kp="300"/>
        <position name="left_j6_act"  joint="left_joint6"  kp="300"/>
        <position name="left_grip"    joint="left_gripper_joint" kp="100"/>
        <position name="right_j1_act" joint="right_joint1" kp="500"/>
        <position name="right_j2_act" joint="right_joint2" kp="500"/>
        <position name="right_j3_act" joint="right_joint3" kp="500"/>
        <position name="right_j4_act" joint="right_joint4" kp="300"/>
        <position name="right_j5_act" joint="right_joint5" kp="300"/>
        <position name="right_j6_act" joint="right_joint6" kp="300"/>
        <position name="right_grip"   joint="right_gripper_joint" kp="100"/>
      </actuator>
    </mujoco>
    """,
}


# ============================================================
# 尝试从 robot_descriptions 加载真实模型
# ============================================================
def try_load_robot_descriptions():
    """尝试加载 robot_descriptions 库中的模型路径。"""
    robots = {}
    try:
        import robot_descriptions
    except ImportError:
        print("⚠  robot_descriptions 未安装，将使用内置回退模型")
        print("   安装方法: pip install robot_descriptions\n")
        return robots

    # (名称, 模块属性路径) —— robot_descriptions 的 MJCF 路径
    candidates = [
        ("UR5e",         "robot_descriptions.ur5e_mj_description",    "MJCF_PATH"),
        ("Franka Panda", "robot_descriptions.panda_mj_description",   "MJCF_PATH"),
        ("Unitree Go2",  "robot_descriptions.go2_mj_description",     "MJCF_PATH"),
        ("Unitree H1",   "robot_descriptions.h1_mj_description",      "MJCF_PATH"),
        ("ALOHA",        "robot_descriptions.aloha_mj_description",    "MJCF_PATH"),
    ]
    for name, module_path, attr in candidates:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            path = getattr(mod, attr)
            robots[name] = ("file", path)
            print(f"  ✓ {name}: {path}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return robots


def load_model(name, source):
    """统一加载接口：从文件路径或 XML 字符串创建 MjModel。"""
    if source[0] == "file":
        return mujoco.MjModel.from_xml_path(source[1])
    else:
        return mujoco.MjModel.from_xml_string(source[1])


# ============================================================
# 分析函数
# ============================================================
def analyze_robot(model: mujoco.MjModel, name: str):
    """全面分析机器人模型的关节和执行器结构。"""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"\n{DIVIDER}")
    print(f"  机器人: {name}")
    print(f"  模型名: {model.opt.timestep:.4f}s 时间步长")
    print(DIVIDER)

    # --- 基本维度 ---
    print(f"\n  【基本维度】")
    print(f"    nq (广义坐标维度) = {model.nq}")
    print(f"    nv (广义速度维度) = {model.nv}")
    print(f"    nu (执行器数量)   = {model.nu}")
    print(f"    nbody (刚体数)    = {model.nbody}")
    print(f"    njnt  (关节数)    = {model.njnt}")
    print(f"    ngeom (几何体数)  = {model.ngeom}")

    # --- 关节详细信息 ---
    print(f"\n  【关节详细信息】  (共 {model.njnt} 个关节)")
    print(f"  {'序号':>4}  {'关节名':20s}  {'类型':6s}  {'qpos起始':>8}  {'qpos维度':>8}  {'dof起始':>7}  {'dof维度':>7}")
    print(f"  {'-'*4}  {'-'*20}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

    has_free_joint = False
    for j in range(model.njnt):
        jnt_name = model.joint(j).name or f"(unnamed_{j})"
        jnt_type = int(model.jnt_type[j])
        type_name = JNT_TYPE_NAMES.get(jnt_type, f"?{jnt_type}")
        qpos_adr = model.jnt_qposadr[j]
        dof_adr = model.jnt_dofadr[j]
        nq = JNT_QPOS_DIM.get(jnt_type, 0)
        nv = JNT_QVEL_DIM.get(jnt_type, 0)
        if jnt_type == 0:
            has_free_joint = True
        print(f"  {j:4d}  {jnt_name:20s}  {type_name:6s}  {qpos_adr:8d}  {nq:8d}  {dof_adr:7d}  {nv:7d}")

    if has_free_joint:
        print(f"\n  ⚡ 该模型包含 free joint（浮动基座）")
        print(f"     qpos 前 7 位 = [x, y, z, qw, qx, qy, qz]")
    else:
        print(f"\n  📌 该模型为固定基座，无 free joint")

    # --- 执行器 ---
    print(f"\n  【执行器信息】  (共 {model.nu} 个执行器)")
    if model.nu > 0:
        print(f"  {'序号':>4}  {'执行器名':25s}  {'关联关节':20s}")
        print(f"  {'-'*4}  {'-'*25}  {'-'*20}")
        for a in range(model.nu):
            act_name = model.actuator(a).name or f"(unnamed_{a})"
            # trntype 0 = joint, trnid[0] 是关联的 joint id
            trntype = model.actuator_trntype[a]
            if trntype == 0 and model.actuator_trnid[a][0] >= 0:
                jnt_id = model.actuator_trnid[a][0]
                jnt_name = model.joint(jnt_id).name or f"(joint_{jnt_id})"
            else:
                jnt_name = f"(trntype={trntype})"
            print(f"  {a:4d}  {act_name:25s}  {jnt_name:20s}")

    # --- 初始 qpos ---
    print(f"\n  【初始 qpos】  (长度 {model.nq})")
    qpos0 = data.qpos.copy()
    if len(qpos0) <= 20:
        for i in range(len(qpos0)):
            print(f"    qpos[{i:2d}] = {qpos0[i]:10.4f}")
    else:
        for i in range(10):
            print(f"    qpos[{i:2d}] = {qpos0[i]:10.4f}")
        print(f"    ... (省略 {len(qpos0) - 20} 个) ...")
        for i in range(len(qpos0) - 10, len(qpos0)):
            print(f"    qpos[{i:2d}] = {qpos0[i]:10.4f}")

    return {
        "name": name,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "njnt": model.njnt,
        "has_free_joint": has_free_joint,
    }


# ============================================================
# 汇总对比表
# ============================================================
def print_comparison_table(results):
    """打印所有机器人的对比汇总表。"""
    print(f"\n\n{'='*80}")
    print("  机器人模型对比汇总")
    print(f"{'='*80}")
    print(f"  {'机器人':20s}  {'nq':>4}  {'nv':>4}  {'nu':>4}  {'njnt':>5}  {'浮动基座':8s}")
    print(f"  {'-'*20}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*8}")
    for r in results:
        fb = "是 (free)" if r["has_free_joint"] else "否 (fixed)"
        print(f"  {r['name']:20s}  {r['nq']:4d}  {r['nv']:4d}  {r['nu']:4d}  {r['njnt']:5d}  {fb}")

    print(f"\n  说明:")
    print(f"    nq = 广义坐标维度（qpos 长度）")
    print(f"    nv = 广义速度维度（qvel 长度，也是 DOF 数）")
    print(f"    nu = 执行器数量（ctrl 长度）")
    print(f"    njnt = 关节数量")
    print(f"    浮动基座 = 是否有 free joint (nq 比 nv 多 1)")


# ============================================================
# 主程序
# ============================================================
def main():
    print(DIVIDER)
    print("  第 4 章 · 01 - 加载真实机器人模型")
    print(DIVIDER)

    # 1. 尝试加载 robot_descriptions
    print("\n【步骤 1】尝试从 robot_descriptions 加载模型...")
    rd_robots = try_load_robot_descriptions()

    # 2. 构建最终模型列表（真实模型 + 回退模型）
    robots_to_load = {}
    real_names = {"UR5e", "Franka Panda", "Unitree Go2", "Unitree H1", "ALOHA"}
    for name in real_names:
        if name in rd_robots:
            robots_to_load[name] = rd_robots[name]
        else:
            fallback_name = f"{name} (fallback)"
            if fallback_name in FALLBACK_MODELS:
                robots_to_load[fallback_name] = ("xml", FALLBACK_MODELS[fallback_name])

    if not robots_to_load:
        print("❌ 没有可用的机器人模型！")
        return

    print(f"\n【步骤 2】将加载 {len(robots_to_load)} 个机器人模型")
    for name in robots_to_load:
        src = robots_to_load[name]
        tag = "文件" if src[0] == "file" else "内置 XML"
        print(f"  • {name} ({tag})")

    # 3. 逐个加载并分析
    results = []
    for name, source in robots_to_load.items():
        try:
            model = load_model(name, source)
            info = analyze_robot(model, name)
            results.append(info)
        except Exception as e:
            print(f"\n❌ 加载 {name} 失败: {e}")

    # 4. 汇总对比
    if results:
        print_comparison_table(results)

    print(f"\n{DIVIDER}")
    print("  ✅ 加载完成！下一步: python 02_compare_robots.py")
    print(DIVIDER)


if __name__ == "__main__":
    main()
