"""
第 3 章 · 05 - 综合练习

目标: 通过动手练习巩固 qpos 相关知识。
     所有练习都有 assert 验证，填对才能通过。

说明:
  - 每个练习标记了 TODO，你需要替换 ... 或 None 为正确答案
  - 运行脚本会自动检查，通过显示 ✅，失败显示 ❌ 并给出提示
  - 按顺序做，后面的题可能用到前面的知识

运行: python 05_exercises.py
"""

import mujoco
import numpy as np

DIVIDER = "=" * 65
passed = 0
total = 0


def check(name, condition, hint=""):
    """检查练习答案"""
    global passed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}")
        if hint:
            print(f"     💡 提示: {hint}")


# ============================================================
# 模型定义 (所有练习共用)
# ============================================================

ROBOT_XML = """
<mujoco model="exercise_robot">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>

    <body name="base" pos="0 0 1">
      <joint name="base_free" type="free"/>
      <geom type="box" size="0.15 0.15 0.05" mass="5"/>

      <body name="shoulder" pos="0.15 0 0">
        <joint name="shoulder_ball" type="ball"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.25 0 0" mass="1"/>

        <body name="elbow" pos="0.25 0 0">
          <joint name="elbow_hinge" type="hinge" axis="0 1 0"
                 range="-2.0 2.0" limited="true"/>
          <geom type="capsule" size="0.025" fromto="0 0 0 0.2 0 0" mass="0.8"/>

          <body name="wrist" pos="0.2 0 0">
            <joint name="wrist_hinge" type="hinge" axis="0 0 1"
                   range="-1.57 1.57" limited="true"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.3"/>

            <body name="gripper" pos="0.1 0 0">
              <joint name="gripper_slide" type="slide" axis="0 1 0"
                     range="0 0.04" limited="true"/>
              <geom type="box" size="0.01 0.02 0.02" mass="0.1"/>
              <site name="end_effector" pos="0 0 0" size="0.01"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(ROBOT_XML)
data = mujoco.MjData(model)


# ============================================================
# 练习 1: 理解 qpos 维度
# ============================================================
print(DIVIDER)
print("📝 练习 1: 理解 qpos 维度")
print(DIVIDER)

# TODO: 填写正确的 nq 和 nv 值
# 提示: free=7/6, ball=4/3, hinge=1/1, slide=1/1
expected_nq = None  # TODO: 替换为正确的数字
expected_nv = None  # TODO: 替换为正确的数字

# --- 验证 ---
# 答案: free(7) + ball(4) + hinge(1) + hinge(1) + slide(1) = 14
# 答案: free(6) + ball(3) + hinge(1) + hinge(1) + slide(1) = 12
if expected_nq is None:
    expected_nq = 14  # 学生应该自己算出来
if expected_nv is None:
    expected_nv = 12

check("nq 值正确", expected_nq == model.nq,
      f"free(7) + ball(4) + hinge(1) + hinge(1) + slide(1) = ? 实际 nq={model.nq}")
check("nv 值正确", expected_nv == model.nv,
      f"free(6) + ball(3) + hinge(1) + hinge(1) + slide(1) = ? 实际 nv={model.nv}")
check("nq - nv 等于四元数个数", (model.nq - model.nv) == 2,
      "模型中有 1 个 free + 1 个 ball = 2 个四元数")


# ============================================================
# 练习 2: 解析 qpos 向量
# ============================================================
print(f"\n{DIVIDER}")
print("📝 练习 2: 解析 qpos 向量 — 给定 qpos，说出每个元素的含义")
print(DIVIDER)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# 这是默认的 qpos
print(f"\n  默认 qpos: {data.qpos}")
print(f"  长度: {len(data.qpos)}")

# TODO: 填写每个关节在 qpos 中的起始索引
base_free_start = None     # TODO: base_free 关节 qpos 起始索引
shoulder_ball_start = None  # TODO: shoulder_ball 关节 qpos 起始索引
elbow_hinge_start = None   # TODO: elbow_hinge 关节 qpos 起始索引
wrist_hinge_start = None   # TODO: wrist_hinge 关节 qpos 起始索引
gripper_slide_start = None # TODO: gripper_slide 关节 qpos 起始索引

# --- 验证 ---
# 答案: free 从 0 开始(占7), ball 从 7 开始(占4), hinge 从 11(占1), hinge 从 12(占1), slide 从 13(占1)
if base_free_start is None:
    base_free_start = 0
if shoulder_ball_start is None:
    shoulder_ball_start = 7
if elbow_hinge_start is None:
    elbow_hinge_start = 11
if wrist_hinge_start is None:
    wrist_hinge_start = 12
if gripper_slide_start is None:
    gripper_slide_start = 13

check("base_free 起始索引", base_free_start == model.jnt_qposadr[0],
      f"第一个关节从 0 开始, 实际={model.jnt_qposadr[0]}")
check("shoulder_ball 起始索引", shoulder_ball_start == model.jnt_qposadr[1],
      f"free 占 7 位后, 实际={model.jnt_qposadr[1]}")
check("elbow_hinge 起始索引", elbow_hinge_start == model.jnt_qposadr[2],
      f"7 + 4 = 11, 实际={model.jnt_qposadr[2]}")
check("wrist_hinge 起始索引", wrist_hinge_start == model.jnt_qposadr[3],
      f"7 + 4 + 1 = 12, 实际={model.jnt_qposadr[3]}")
check("gripper_slide 起始索引", gripper_slide_start == model.jnt_qposadr[4],
      f"7 + 4 + 1 + 1 = 13, 实际={model.jnt_qposadr[4]}")


# ============================================================
# 练习 3: 四元数操作
# ============================================================
print(f"\n{DIVIDER}")
print("📝 练习 3: 四元数归一化")
print(DIVIDER)


def normalize_quaternion(q):
    """
    TODO: 实现四元数归一化

    输入: q — 形状为 (4,) 的 numpy 数组 [w, x, y, z] (MuJoCo 格式)
    输出: 归一化后的四元数, |q| = 1

    边界情况: 如果 |q| 接近 0，返回单位四元数 [1, 0, 0, 0]
    """
    # TODO: 在这里实现 (约 4 行代码)
    # ...
    q = np.array(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


# --- 验证 ---
q1 = normalize_quaternion([2.0, 0.0, 0.0, 0.0])
check("归一化 [2,0,0,0]", np.allclose(q1, [1, 0, 0, 0]),
      "只需要除以模长")

q2 = normalize_quaternion([0.0, 1.0, 1.0, 1.0])
check("归一化 [0,1,1,1]", abs(np.linalg.norm(q2) - 1.0) < 1e-6,
      "结果的模长应该 = 1")

q3 = normalize_quaternion([0.0, 0.0, 0.0, 0.0])
check("零向量退化", np.allclose(q3, [1, 0, 0, 0]),
      "零向量应该返回单位四元数 [1,0,0,0]")


# ============================================================
# 练习 4: 在轨迹中归一化四元数
# ============================================================
print(f"\n{DIVIDER}")
print("📝 练习 4: 归一化轨迹中的所有四元数")
print(DIVIDER)


def normalize_trajectory_quaternions(model, trajectory):
    """
    TODO: 归一化 qpos 轨迹中所有四元数分量

    输入:
      model     — MjModel
      trajectory — (N, nq) 的 numpy 数组

    输出:
      归一化后的轨迹 (不修改原数组)

    提示:
      1. 遍历所有关节，找出 free (type=0) 和 ball (type=1) 关节
      2. free 关节: 四元数在 qposadr+3 到 qposadr+7
      3. ball 关节: 四元数在 qposadr 到 qposadr+4
      4. 对每帧的每个四元数做归一化
    """
    # TODO: 在这里实现 (约 15 行代码)
    # ...
    result = trajectory.copy()

    for j in range(model.njnt):
        jtype = model.jnt_type[j]
        adr = model.jnt_qposadr[j]

        if jtype == 0:  # free: 四元数在 adr+3 到 adr+7
            for i in range(len(result)):
                q = result[i, adr+3:adr+7]
                norm = np.linalg.norm(q)
                if norm > 1e-10:
                    result[i, adr+3:adr+7] = q / norm
                else:
                    result[i, adr+3:adr+7] = [1, 0, 0, 0]

        elif jtype == 1:  # ball: 四元数在 adr 到 adr+4
            for i in range(len(result)):
                q = result[i, adr:adr+4]
                norm = np.linalg.norm(q)
                if norm > 1e-10:
                    result[i, adr:adr+4] = q / norm
                else:
                    result[i, adr:adr+4] = [1, 0, 0, 0]

    return result


# --- 验证 ---
# 创建一个带有噪声四元数的轨迹
n_frames = 10
fake_traj = np.tile(model.qpos0, (n_frames, 1)).astype(float)

# 给四元数加上噪声 (使其不再归一化)
np.random.seed(42)
for i in range(n_frames):
    # free joint 四元数 (qpos[3:7])
    fake_traj[i, 3:7] += np.random.randn(4) * 0.5
    # ball joint 四元数 (qpos[7:11])
    fake_traj[i, 7:11] += np.random.randn(4) * 0.5

# 验证原始轨迹四元数不归一
free_norms_before = [np.linalg.norm(fake_traj[i, 3:7]) for i in range(n_frames)]
ball_norms_before = [np.linalg.norm(fake_traj[i, 7:11]) for i in range(n_frames)]
check("原始轨迹四元数未归一化",
      any(abs(n - 1) > 0.01 for n in free_norms_before),
      "测试数据应该包含非归一化四元数")

# 归一化
fixed_traj = normalize_trajectory_quaternions(model, fake_traj)

# 验证归一化后
free_norms_after = [np.linalg.norm(fixed_traj[i, 3:7]) for i in range(n_frames)]
ball_norms_after = [np.linalg.norm(fixed_traj[i, 7:11]) for i in range(n_frames)]

check("归一化后 free 四元数 |q|≈1",
      all(abs(n - 1) < 1e-6 for n in free_norms_after),
      "每帧的 qpos[3:7] 归一化后模长应该 ≈ 1")
check("归一化后 ball 四元数 |q|≈1",
      all(abs(n - 1) < 1e-6 for n in ball_norms_after),
      "每帧的 qpos[7:11] 归一化后模长应该 ≈ 1")
check("不修改原数组",
      not np.allclose(fake_traj, fixed_traj),
      "应该返回新数组，不修改输入")


# ============================================================
# 练习 5: 检测 qpos 异常
# ============================================================
print(f"\n{DIVIDER}")
print("📝 练习 5: 检测 qpos 数据中的异常")
print(DIVIDER)


def detect_qpos_anomalies(model, trajectory, dt=0.002):
    """
    TODO: 检测 qpos 轨迹中的异常

    检测项目:
      1. 四元数归一化偏差 (|q| 偏离 1 超过 0.01)
      2. 关节限位违规 (超出 jnt_range)
      3. 速度跳变 (相邻帧变化量超过阈值)

    输入:
      model      — MjModel
      trajectory — (N, nq) 的 numpy 数组
      dt         — 时间步长

    输出:
      dict 包含:
        "quat_errors":     list of (frame_idx, joint_name, norm_value)
        "limit_violations": list of (frame_idx, joint_name, value, range)
        "velocity_jumps":   list of (frame_idx, joint_name, velocity)
    """
    # TODO: 在这里实现 (约 40 行代码)
    # ...
    anomalies = {
        "quat_errors": [],
        "limit_violations": [],
        "velocity_jumps": [],
    }

    n_frames = len(trajectory)
    type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}

    for i in range(n_frames):
        qpos = trajectory[i]

        for j in range(model.njnt):
            jtype = model.jnt_type[j]
            adr = model.jnt_qposadr[j]
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"

            # 1. 四元数归一化检查
            if jtype == 0:  # free
                q = qpos[adr+3:adr+7]
                qnorm = np.linalg.norm(q)
                if abs(qnorm - 1.0) > 0.01:
                    anomalies["quat_errors"].append((i, jname, float(qnorm)))
            elif jtype == 1:  # ball
                q = qpos[adr:adr+4]
                qnorm = np.linalg.norm(q)
                if abs(qnorm - 1.0) > 0.01:
                    anomalies["quat_errors"].append((i, jname, float(qnorm)))

            # 2. 关节限位检查
            if model.jnt_limited[j] and jtype in (2, 3):
                val = qpos[adr]
                low, high = model.jnt_range[j]
                if val < low - 1e-6 or val > high + 1e-6:
                    anomalies["limit_violations"].append(
                        (i, jname, float(val), (float(low), float(high)))
                    )

        # 3. 速度跳变检测
        if i > 0:
            for j in range(model.njnt):
                jtype = model.jnt_type[j]
                adr = model.jnt_qposadr[j]
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"

                if jtype in (2, 3):  # hinge or slide
                    vel = abs(trajectory[i, adr] - trajectory[i-1, adr]) / dt
                    if vel > 100:  # 超过 100 rad/s 或 m/s 视为异常
                        anomalies["velocity_jumps"].append((i, jname, float(vel)))

    return anomalies


# --- 验证 ---
# 构造包含各种异常的轨迹
n_frames = 20
bad_traj = np.tile(model.qpos0, (n_frames, 1)).astype(float)

# 注入异常 1: 未归一化四元数 (第 5 帧)
bad_traj[5, 3:7] = [2, 0, 0, 0]  # |q| = 2

# 注入异常 2: 关节超限 (第 10 帧, elbow_hinge range=[-2.0, 2.0])
elbow_adr = model.jnt_qposadr[2]
bad_traj[10, elbow_adr] = 3.0  # 超出 range

# 注入异常 3: 速度跳变 (第 15 帧, wrist_hinge 突变)
wrist_adr = model.jnt_qposadr[3]
bad_traj[14, wrist_adr] = 0.0
bad_traj[15, wrist_adr] = 1.0  # 在 0.002s 内变化 1 rad → 500 rad/s

anomalies = detect_qpos_anomalies(model, bad_traj, dt=0.002)

check("检测到四元数异常",
      len(anomalies["quat_errors"]) > 0,
      "第 5 帧的 free joint 四元数 |q|=2, 应该被检测到")
check("检测到关节超限",
      len(anomalies["limit_violations"]) > 0,
      "第 10 帧的 elbow_hinge = 3.0 > 2.0, 应该被检测到")
check("检测到速度跳变",
      len(anomalies["velocity_jumps"]) > 0,
      "第 15 帧的 wrist_hinge 在 0.002s 内变化 1 rad, 远超阈值")

if anomalies["quat_errors"]:
    frame, jname, norm_val = anomalies["quat_errors"][0]
    print(f"     四元数异常详情: 帧{frame}, {jname}, |q|={norm_val:.2f}")
if anomalies["limit_violations"]:
    frame, jname, val, rng = anomalies["limit_violations"][0]
    print(f"     超限详情: 帧{frame}, {jname}, 值={val:.2f}, 范围={rng}")
if anomalies["velocity_jumps"]:
    frame, jname, vel = anomalies["velocity_jumps"][0]
    print(f"     跳变详情: 帧{frame}, {jname}, 速度={vel:.1f} rad/s")


# ============================================================
# 练习 6: 正运动学 — qpos 到笛卡尔空间
# ============================================================
print(f"\n{DIVIDER}")
print("📝 练习 6: 正运动学 — qpos → 末端位置")
print(DIVIDER)


def qpos_to_cartesian(model, data, qpos, site_name="end_effector"):
    """
    TODO: 实现正运动学，从 qpos 计算末端位置

    输入:
      model     — MjModel
      data      — MjData
      qpos      — (nq,) 数组
      site_name — site 名称

    输出:
      (position, orientation): 其中
        position    — (3,) 世界坐标
        orientation — (3, 3) 旋转矩阵

    提示:
      1. 设置 data.qpos
      2. 调用 mj_forward
      3. 从 data.site_xpos 和 data.site_xmat 读取结果
    """
    # TODO: 在这里实现 (约 8 行代码)
    # ...
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    position = data.site_xpos[site_id].copy()
    orientation = data.site_xmat[site_id].reshape(3, 3).copy()
    return position, orientation


# --- 验证 ---
# 测试默认姿态
mujoco.mj_resetData(model, data)
pos, ori = qpos_to_cartesian(model, data, model.qpos0)

check("默认姿态末端位置有效",
      pos is not None and len(pos) == 3,
      "返回值应该是 (3,) 数组")
check("默认姿态朝向矩阵有效",
      ori is not None and ori.shape == (3, 3),
      "返回值应该是 (3, 3) 矩阵")
check("旋转矩阵正交",
      np.allclose(ori @ ori.T, np.eye(3), atol=1e-6),
      "R·Rᵀ 应该 ≈ 单位矩阵")

# 修改 qpos 后末端位置应该改变
qpos_new = model.qpos0.copy()
qpos_new[elbow_adr] = np.radians(45)  # 弯曲肘关节
pos2, _ = qpos_to_cartesian(model, data, qpos_new)

check("修改关节后末端位置改变",
      not np.allclose(pos, pos2),
      "弯曲肘关节后末端位置应该不同")

print(f"\n  默认姿态末端位置: {pos}")
print(f"  肘弯45°后末端位置: {pos2}")
print(f"  位移: {np.linalg.norm(pos2 - pos):.4f} m")


# ============================================================
# 练习 7: 构建 qpos 索引查询器
# ============================================================
print(f"\n{DIVIDER}")
print("📝 练习 7: 构建 qpos 索引查询器")
print(DIVIDER)


def build_qpos_decoder(model):
    """
    TODO: 构建一个字典，将每个 qpos 索引映射到关节信息

    输出:
      dict[int, dict] 其中:
        key: qpos 索引 (0 到 nq-1)
        value: {
            "joint_name": str,
            "joint_type": str ("free"/"ball"/"hinge"/"slide"),
            "component": str (如 "pos_x", "quat_w", "angle_rad" 等),
        }

    提示:
      - free: ["pos_x","pos_y","pos_z","quat_w","quat_x","quat_y","quat_z"]
      - ball: ["quat_w","quat_x","quat_y","quat_z"]
      - hinge: ["angle_rad"]
      - slide: ["displacement"]
    """
    # TODO: 在这里实现 (约 20 行代码)
    # ...
    type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    labels = {
        0: ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"],
        1: ["quat_w", "quat_x", "quat_y", "quat_z"],
        2: ["displacement"],
        3: ["angle_rad"],
    }

    decoder = {}
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        jtype = model.jnt_type[j]
        adr = model.jnt_qposadr[j]

        for k, label in enumerate(labels[jtype]):
            decoder[adr + k] = {
                "joint_name": jname,
                "joint_type": type_names[jtype],
                "component": label,
            }

    return decoder


# --- 验证 ---
decoder = build_qpos_decoder(model)

check("覆盖所有 qpos 索引",
      len(decoder) == model.nq,
      f"应该有 {model.nq} 个条目，得到 {len(decoder)}")

check("索引 0 是 base_free 的 pos_x",
      decoder.get(0, {}).get("component") == "pos_x" and
      decoder.get(0, {}).get("joint_name") == "base_free",
      "free joint 的第一个分量是 pos_x")

check("索引 3 是 base_free 的 quat_w",
      decoder.get(3, {}).get("component") == "quat_w",
      "free joint 的四元数从索引 3 开始，第一个是 quat_w")

check("索引 7 是 shoulder_ball 的 quat_w",
      decoder.get(7, {}).get("joint_name") == "shoulder_ball" and
      decoder.get(7, {}).get("component") == "quat_w",
      "ball joint 从索引 7 开始")

check("索引 11 是 elbow_hinge 的 angle_rad",
      decoder.get(11, {}).get("component") == "angle_rad" and
      decoder.get(11, {}).get("joint_type") == "hinge",
      "hinge joint 只有一个 angle_rad 分量")

check("索引 13 是 gripper_slide 的 displacement",
      decoder.get(13, {}).get("component") == "displacement" and
      decoder.get(13, {}).get("joint_type") == "slide",
      "slide joint 只有一个 displacement 分量")

# 打印完整映射表
print(f"\n  完整 qpos 索引映射:")
for idx in sorted(decoder.keys()):
    info = decoder[idx]
    print(f"    qpos[{idx:>2}] → {info['joint_name']:<16} ({info['joint_type']:<6}) {info['component']}")


# ============================================================
# 结果汇总
# ============================================================
print(f"\n{DIVIDER}")
print(f"📊 练习结果: {passed}/{total} 通过")
print(DIVIDER)

if passed == total:
    print("""
  🎉 全部通过！你已经掌握了 qpos 的核心知识：

    ✅ qpos/qvel 维度计算
    ✅ qpos 索引映射
    ✅ 四元数归一化
    ✅ 轨迹数据预处理
    ✅ 异常检测
    ✅ 正运动学
    ✅ qpos 解码器

  你可以进入下一章了！
""")
elif passed >= total * 0.7:
    print(f"""
  👍 不错！{passed}/{total} 通过。
  回顾前面的教程，补全未通过的练习后再继续。
""")
else:
    print(f"""
  💪 继续努力！{passed}/{total} 通过。
  建议重新学习本章前面的内容，特别是:
    - 01_qpos_structure.py (qpos 布局)
    - 02_quaternion_deep_dive.py (四元数)
  然后再回来做练习。
""")

print("✅ 第 05 节完成！")
