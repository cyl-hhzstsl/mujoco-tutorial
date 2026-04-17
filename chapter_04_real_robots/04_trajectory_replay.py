"""
第 4 章 · 04 - 轨迹回放

目标: 加载已录制的轨迹数据，在仿真中逐帧回放，
     支持速度调节、格式自动检测、统计分析。

核心知识点:
  1. 从 .pkl 文件加载完整轨迹
  2. 通过设置 qpos + mj_forward 实现回放（无需物理仿真）
  3. 轨迹数据格式的自动检测
  4. 回放速度控制
  5. 轨迹质量统计

前置: 先运行 03_trajectory_recording.py 生成数据

运行: python 04_trajectory_replay.py
"""

import mujoco
import numpy as np
import os
import time

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# 与 03 脚本一致的机械臂模型
ARM_XML = """
<mujoco model="6dof_arm">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="2 2 0.01" rgba="0.9 0.9 0.9 1"/>

    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.08 0.05" rgba="0.2 0.2 0.2 1" mass="5"/>

      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="2"/>
        <geom type="cylinder" size="0.05 0.08" rgba="0.8 0.3 0.3 1" mass="3"/>

        <body name="link2" pos="0 0 0.08">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2.36 2.36" damping="2"/>
          <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.3" rgba="0.3 0.3 0.8 1" mass="2.5"/>

          <body name="link3" pos="0 0 0.3">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-2.36 2.36" damping="1.5"/>
            <geom type="capsule" size="0.035" fromto="0 0 0 0 0 0.25" rgba="0.3 0.8 0.3 1" mass="2"/>

            <body name="link4" pos="0 0 0.25">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1"/>
              <geom type="cylinder" size="0.03 0.04" rgba="0.8 0.8 0.3 1" mass="1"/>

              <body name="link5" pos="0 0 0.04">
                <joint name="joint5" type="hinge" axis="0 1 0" range="-2.09 2.09" damping="0.5"/>
                <geom type="cylinder" size="0.025 0.03" rgba="0.8 0.3 0.8 1" mass="0.5"/>

                <body name="link6" pos="0 0 0.03">
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.5"/>
                  <geom type="cylinder" size="0.02 0.02" rgba="0.3 0.8 0.8 1" mass="0.3"/>
                  <site name="end_effector" pos="0 0 0.04" size="0.015" rgba="1 0 0 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="act1" joint="joint1" kp="500" kv="50"/>
    <position name="act2" joint="joint2" kp="500" kv="50"/>
    <position name="act3" joint="joint3" kp="500" kv="50"/>
    <position name="act4" joint="joint4" kp="200" kv="20"/>
    <position name="act5" joint="joint5" kp="200" kv="20"/>
    <position name="act6" joint="joint6" kp="100" kv="10"/>
  </actuator>

  <sensor>
    <jointpos name="jpos1" joint="joint1"/>
    <jointpos name="jpos2" joint="joint2"/>
    <jointpos name="jpos3" joint="joint3"/>
    <jointpos name="jpos4" joint="joint4"/>
    <jointpos name="jpos5" joint="joint5"/>
    <jointpos name="jpos6" joint="joint6"/>
  </sensor>
</mujoco>
"""


# ============================================================
# 轨迹加载与格式自动检测
# ============================================================
def load_trajectory(filepath):
    """
    自动检测并加载轨迹数据。
    支持: .pkl, .npz, .npy, .json

    返回: (trajectory_dict, format_name)
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pkl":
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return detect_and_normalize(data, "pickle")

    elif ext == ".npz":
        npz = np.load(filepath, allow_pickle=True)
        data = {key: npz[key] for key in npz.files}
        return detect_and_normalize(data, "npz")

    elif ext == ".npy":
        arr = np.load(filepath, allow_pickle=True)
        # .npy 可能是纯数组或 allow_pickle 的字典
        if isinstance(arr, np.ndarray) and arr.ndim == 0:
            data = arr.item()
        elif isinstance(arr, np.ndarray):
            data = {"qpos": arr}
        else:
            data = arr
        return detect_and_normalize(data, "npy")

    elif ext == ".json":
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return detect_and_normalize(data, "json")

    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def detect_and_normalize(data, source_format):
    """
    自动检测数据结构并归一化为标准格式。

    标准格式:
      {
        "time":  ndarray (T,),
        "qpos":  ndarray (T, nq),
        "qvel":  ndarray (T, nv) or None,
        "ctrl":  ndarray (T, nu) or None,
        "ee_pos": ndarray (T, 3) or None,
        "metadata": dict or None,
      }
    """
    result = {
        "time": None, "qpos": None, "qvel": None,
        "ctrl": None, "ee_pos": None, "metadata": None,
    }
    detected = []

    if isinstance(data, dict):
        # 情况 1: 标准字典格式
        for key in ["time", "qpos", "qvel", "ctrl", "ee_pos"]:
            if key in data:
                val = data[key]
                if not isinstance(val, np.ndarray):
                    val = np.array(val)
                result[key] = val
                detected.append(key)
        if "metadata" in data:
            result["metadata"] = data["metadata"]
            detected.append("metadata")

    elif isinstance(data, list):
        # 情况 2: 帧列表 [{"qpos": [...], ...}, ...]
        if len(data) > 0 and isinstance(data[0], dict):
            n = len(data)
            if "qpos" in data[0]:
                nq = len(data[0]["qpos"])
                result["qpos"] = np.array([f["qpos"] for f in data])
                detected.append(f"qpos (from {n} frames)")
            if "time" in data[0]:
                result["time"] = np.array([f["time"] for f in data])
                detected.append("time")

    elif isinstance(data, np.ndarray):
        # 情况 3: 纯数组，假设是 qpos
        result["qpos"] = data
        detected.append(f"qpos array {data.shape}")

    # 如果没有 time，自动生成
    if result["qpos"] is not None and result["time"] is None:
        n = len(result["qpos"])
        dt = 0.02  # 默认 50 Hz
        if result["metadata"] and "record_dt" in result["metadata"]:
            dt = result["metadata"]["record_dt"]
        result["time"] = np.arange(n) * dt
        detected.append(f"time (auto-generated, dt={dt})")

    print(f"    数据源格式: {source_format}")
    print(f"    检测到的字段: {', '.join(detected)}")
    if result["qpos"] is not None:
        print(f"    轨迹帧数: {len(result['qpos'])}")
        print(f"    qpos 维度: {result['qpos'].shape[1] if result['qpos'].ndim > 1 else 1}")

    return result, source_format


# ============================================================
# 轨迹回放（无可视化）
# ============================================================
def replay_headless(model, data, trajectory, speed=1.0, verbose_every=50):
    """
    无可视化回放：逐帧设置 qpos，调用 mj_forward，打印信息。

    参数:
      speed         - 回放速度倍率 (1.0=实时, 2.0=两倍速, 0.5=半速)
      verbose_every - 每隔多少帧打印一次详细信息
    """
    qpos_traj = trajectory["qpos"]
    time_traj = trajectory["time"]
    n_frames = len(qpos_traj)

    if qpos_traj.shape[1] != model.nq:
        print(f"  ⚠ 轨迹 qpos 维度 ({qpos_traj.shape[1]}) != 模型 nq ({model.nq})")
        print(f"    将截断/填充到 nq={model.nq}")

    ee_site_id = -1
    try:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    except Exception:
        pass

    print(f"\n  开始回放: {n_frames} 帧, 速度 {speed}x")
    print(f"  {'帧号':>6}  {'时间(s)':>8}  {'qpos[0]':>9}  {'qpos[1]':>9}  {'qpos[2]':>9}  {'末端 z':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")

    replay_ee_positions = []
    t_start = time.time()

    for i in range(n_frames):
        # 设置 qpos
        nq = min(qpos_traj.shape[1], model.nq)
        data.qpos[:nq] = qpos_traj[i, :nq]

        # 设置 qvel（如果有）
        if trajectory["qvel"] is not None:
            nv = min(trajectory["qvel"].shape[1], model.nv)
            data.qvel[:nv] = trajectory["qvel"][i, :nv]

        # 前向运动学（不做物理仿真）
        mujoco.mj_forward(model, data)

        # 记录末端位置
        ee_z = 0.0
        if ee_site_id >= 0:
            ee_pos = data.site_xpos[ee_site_id].copy()
            replay_ee_positions.append(ee_pos)
            ee_z = ee_pos[2]

        # 打印
        if i % verbose_every == 0 or i == n_frames - 1:
            t = time_traj[i] if i < len(time_traj) else 0
            q = qpos_traj[i]
            print(f"  {i:6d}  {t:8.3f}  {q[0]:9.4f}  {q[1]:9.4f}  {q[2]:9.4f}  {ee_z:8.4f}")

        # 速度控制
        if speed > 0 and speed != float("inf") and i < n_frames - 1:
            dt_data = time_traj[i + 1] - time_traj[i] if i + 1 < len(time_traj) else 0.02
            sleep_time = dt_data / speed
            if sleep_time > 0.001:  # 避免过于频繁的 sleep
                time.sleep(sleep_time)

    wall_time = time.time() - t_start
    sim_time = time_traj[-1] - time_traj[0] if len(time_traj) > 1 else 0
    print(f"\n  回放完成:")
    print(f"    仿真时间:  {sim_time:.3f} s")
    print(f"    墙钟时间:  {wall_time:.3f} s")
    print(f"    实际速度:  {sim_time / wall_time:.2f}x" if wall_time > 0 else "    (瞬间完成)")

    return np.array(replay_ee_positions) if replay_ee_positions else None


# ============================================================
# 轨迹回放（带可视化）
# ============================================================
def replay_with_viewer(model, data, trajectory, speed=1.0):
    """尝试使用 MuJoCo viewer 回放（需要显示环境）。"""
    try:
        import mujoco.viewer
    except ImportError:
        print("  ⚠ mujoco.viewer 不可用，跳过可视化回放")
        return False

    qpos_traj = trajectory["qpos"]
    time_traj = trajectory["time"]
    n_frames = len(qpos_traj)

    print(f"\n  启动可视化回放 ({n_frames} 帧, {speed}x)...")
    print(f"  按 ESC 或关闭窗口退出")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for i in range(n_frames):
                if not viewer.is_running():
                    print(f"  用户关闭了 viewer (帧 {i}/{n_frames})")
                    break

                nq = min(qpos_traj.shape[1], model.nq)
                data.qpos[:nq] = qpos_traj[i, :nq]
                if trajectory["qvel"] is not None:
                    nv = min(trajectory["qvel"].shape[1], model.nv)
                    data.qvel[:nv] = trajectory["qvel"][i, :nv]

                mujoco.mj_forward(model, data)
                viewer.sync()

                if speed > 0 and i < n_frames - 1:
                    dt_data = time_traj[i + 1] - time_traj[i] if i + 1 < len(time_traj) else 0.02
                    time.sleep(dt_data / speed)

        print(f"  ✓ 可视化回放完成")
        return True

    except Exception as e:
        print(f"  ⚠ 可视化回放失败: {e}")
        print(f"    这通常是因为没有显示环境 (headless 服务器)")
        return False


# ============================================================
# 轨迹统计
# ============================================================
def compute_trajectory_statistics(trajectory):
    """计算并打印轨迹的详细统计信息。"""
    print(f"\n{DIVIDER}")
    print(f"  【轨迹统计分析】")
    print(DIVIDER)

    qpos = trajectory["qpos"]
    time_arr = trajectory["time"]
    n_frames, nq = qpos.shape

    meta = trajectory.get("metadata") or {}
    joint_names = meta.get("joint_names", [f"q{i}" for i in range(nq)])

    # 1. 基本信息
    duration = time_arr[-1] - time_arr[0] if len(time_arr) > 1 else 0
    dt = np.diff(time_arr)
    print(f"\n  基本信息:")
    print(f"    总帧数:          {n_frames}")
    print(f"    时间跨度:        {duration:.3f} s")
    print(f"    平均帧间隔:      {dt.mean():.6f} s ({1.0 / dt.mean():.1f} Hz)" if len(dt) > 0 else "")
    print(f"    帧间隔标准差:    {dt.std():.6f} s" if len(dt) > 0 else "")

    # 2. 各关节统计
    print(f"\n  各关节 qpos 统计:")
    print(f"  {'关节':14s}  {'均值':>8}  {'标准差':>8}  {'最小':>8}  {'最大':>8}  {'变化范围':>8}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for j in range(nq):
        jname = joint_names[j] if j < len(joint_names) else f"q{j}"
        col = qpos[:, j]
        print(f"  {jname:14s}  {col.mean():8.4f}  {col.std():8.4f}  "
              f"{col.min():8.4f}  {col.max():8.4f}  {col.max() - col.min():8.4f}")

    # 3. 速度统计（通过数值差分）
    if len(time_arr) > 1:
        print(f"\n  数值差分速度 (dq/dt) 统计:")
        dq = np.diff(qpos, axis=0)
        dt_col = np.diff(time_arr)[:, None]
        velocity = dq / dt_col
        print(f"  {'关节':14s}  {'均值速度':>10}  {'最大|速度|':>12}  {'平滑度':>10}")
        print(f"  {'-'*14}  {'-'*10}  {'-'*12}  {'-'*10}")
        for j in range(nq):
            jname = joint_names[j] if j < len(joint_names) else f"q{j}"
            v = velocity[:, j]
            smoothness = np.abs(np.diff(v)).mean()  # 加速度均值 ≈ 平滑度
            print(f"  {jname:14s}  {v.mean():10.4f}  {np.abs(v).max():12.4f}  {smoothness:10.6f}")

    # 4. 相关性（是否有关节高度相关）
    if nq >= 2:
        print(f"\n  关节相关性矩阵 (|r| > 0.7 标为 ★):")
        corr = np.corrcoef(qpos.T)
        header = "  " + " " * 14
        for j in range(min(nq, 8)):
            header += f"  {joint_names[j][:6]:>6}" if j < len(joint_names) else f"  q{j:>4}"
        print(header)
        for i in range(min(nq, 8)):
            iname = joint_names[i][:14] if i < len(joint_names) else f"q{i}"
            line = f"  {iname:14s}"
            for j in range(min(nq, 8)):
                r = corr[i, j]
                marker = "★" if abs(r) > 0.7 and i != j else " "
                line += f" {r:6.2f}{marker}"
            print(line)

    # 5. 轨迹录制的 ctrl 回放误差（如果有 qvel）
    if trajectory["qvel"] is not None:
        qvel = trajectory["qvel"]
        print(f"\n  qvel 统计:")
        print(f"    最大绝对速度: {np.abs(qvel).max():.4f} rad/s")
        print(f"    平均绝对速度: {np.abs(qvel).mean():.4f} rad/s")
        static_ratio = (np.abs(qvel) < 0.01).mean() * 100
        print(f"    静止帧比例 (|qvel| < 0.01): {static_ratio:.1f}%")

    # 6. 末端执行器
    if trajectory["ee_pos"] is not None and np.any(trajectory["ee_pos"] != 0):
        ee = trajectory["ee_pos"]
        print(f"\n  末端执行器轨迹:")
        distances = np.sqrt(np.sum(np.diff(ee, axis=0)**2, axis=1))
        total_dist = distances.sum()
        print(f"    总行程距离:   {total_dist:.4f} m")
        print(f"    工作空间 X:   [{ee[:, 0].min():.4f}, {ee[:, 0].max():.4f}] m")
        print(f"    工作空间 Y:   [{ee[:, 1].min():.4f}, {ee[:, 1].max():.4f}] m")
        print(f"    工作空间 Z:   [{ee[:, 2].min():.4f}, {ee[:, 2].max():.4f}] m")


# ============================================================
# 多速度回放
# ============================================================
def replay_at_multiple_speeds(model, data, trajectory):
    """以不同速度回放，展示速度控制效果。"""
    print(f"\n{DIVIDER}")
    print(f"  【多速度回放测试】")
    print(DIVIDER)

    # 为了快速演示，只取前 100 帧
    short_traj = {k: v for k, v in trajectory.items()}
    max_frames = min(100, len(trajectory["qpos"]))
    short_traj["qpos"] = trajectory["qpos"][:max_frames]
    short_traj["time"] = trajectory["time"][:max_frames]
    if trajectory["qvel"] is not None:
        short_traj["qvel"] = trajectory["qvel"][:max_frames]

    speeds = [float("inf"), 2.0, 1.0, 0.5]
    speed_names = ["最快 (∞)", "2x 倍速", "1x 实时", "0.5x 慢放"]

    results = []
    for speed, sname in zip(speeds, speed_names):
        print(f"\n  ─── {sname} ───")
        t0 = time.time()
        for i in range(max_frames):
            nq = min(short_traj["qpos"].shape[1], model.nq)
            data.qpos[:nq] = short_traj["qpos"][i, :nq]
            mujoco.mj_forward(model, data)

            if speed != float("inf") and speed > 0 and i < max_frames - 1:
                dt_data = short_traj["time"][i + 1] - short_traj["time"][i]
                sleep_time = dt_data / speed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

        wall_time = time.time() - t0
        sim_time = short_traj["time"][max_frames - 1] - short_traj["time"][0]
        actual_speed = sim_time / wall_time if wall_time > 0 else float("inf")
        results.append((sname, speed, wall_time, actual_speed))
        print(f"    {max_frames} 帧, 仿真时间 {sim_time:.3f}s, "
              f"墙钟 {wall_time:.3f}s, 实际速度 {actual_speed:.1f}x")

    print(f"\n  速度对比:")
    print(f"  {'模式':12s}  {'目标速度':>8}  {'墙钟时间':>10}  {'实际速度':>10}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*10}")
    for sname, target, wall, actual in results:
        ts = f"{target:.1f}x" if target != float("inf") else "∞"
        print(f"  {sname:12s}  {ts:>8}  {wall:10.3f}s  {actual:9.1f}x")


# ============================================================
# 主程序
# ============================================================
def main():
    print(DIVIDER)
    print("  第 4 章 · 04 - 轨迹回放")
    print(DIVIDER)

    # 1. 加载模型
    print(f"\n【步骤 1】加载机械臂模型...")
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)
    print(f"  模型: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # 2. 查找轨迹文件
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recorded_data")
    pkl_path = os.path.join(data_dir, "trajectory.pkl")

    if not os.path.exists(pkl_path):
        print(f"\n  ⚠ 未找到轨迹文件: {pkl_path}")
        print(f"    请先运行: python 03_trajectory_recording.py")
        print(f"\n  将生成演示轨迹以继续...")

        # 自动生成一段简单轨迹
        n_frames = 250
        t = np.linspace(0, 5.0, n_frames)
        qpos = np.zeros((n_frames, model.nq))
        qvel = np.zeros((n_frames, model.nv))
        for i in range(model.nq):
            freq = 0.3 + i * 0.15
            amp = 0.5 / (1 + i * 0.3)
            qpos[:, i] = amp * np.sin(2 * np.pi * freq * t + i * np.pi / 4)
            qvel[:, i] = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t + i * np.pi / 4)

        trajectory = {
            "time": t,
            "qpos": qpos,
            "qvel": qvel,
            "ctrl": None,
            "ee_pos": np.zeros((n_frames, 3)),
            "metadata": {
                "model_name": "6dof_arm",
                "nq": model.nq,
                "nv": model.nv,
                "nu": model.nu,
                "record_dt": t[1] - t[0],
                "duration": 5.0,
                "n_frames": n_frames,
                "record_hz": n_frames / 5.0,
                "joint_names": [model.joint(j).name for j in range(model.njnt)],
                "actuator_names": [model.actuator(a).name for a in range(model.nu)],
            },
        }
        fmt = "auto-generated"
        print(f"  ✓ 生成了 {n_frames} 帧演示轨迹")
    else:
        print(f"\n【步骤 2】加载轨迹文件...")
        print(f"  文件: {pkl_path}")
        trajectory, fmt = load_trajectory(pkl_path)
        print(f"  ✓ 加载完成 (格式: {fmt})")

    # 3. 轨迹统计
    compute_trajectory_statistics(trajectory)

    # 4. 无可视化回放（最快速度，打印关键帧）
    print(f"\n{DIVIDER}")
    print(f"  【无可视化回放】")
    print(DIVIDER)
    print(f"\n  模式: headless (最快速度)")
    replay_ee = replay_headless(model, data, trajectory, speed=float("inf"), verbose_every=50)

    # 5. 回放 vs 原始末端位置对比
    if (replay_ee is not None and trajectory["ee_pos"] is not None
            and np.any(trajectory["ee_pos"] != 0)):
        orig_ee = trajectory["ee_pos"]
        min_len = min(len(replay_ee), len(orig_ee))
        if min_len > 0:
            error = np.sqrt(np.sum((replay_ee[:min_len] - orig_ee[:min_len]) ** 2, axis=1))
            print(f"\n  回放误差分析 (末端执行器位置):")
            print(f"    平均误差: {error.mean():.6f} m")
            print(f"    最大误差: {error.max():.6f} m")
            if error.max() < 1e-6:
                print(f"    ✓ 完美回放 (设置 qpos 方式)")
            else:
                print(f"    ℹ 误差来源: 回放时通过 mj_forward 重算，与录制时可能有微小差异")

    # 6. 多速度回放
    replay_at_multiple_speeds(model, data, trajectory)

    # 7. 可视化回放（可选）
    print(f"\n{DIVIDER}")
    print(f"  【可视化回放 (可选)】")
    print(DIVIDER)
    try:
        use_viewer = False
        env_display = os.environ.get("DISPLAY", "")
        env_wayland = os.environ.get("WAYLAND_DISPLAY", "")
        if env_display or env_wayland or os.name == "nt" or os.uname().sysname == "Darwin":
            use_viewer = True

        if use_viewer:
            print(f"  检测到图形环境，尝试启动 viewer...")
            replay_with_viewer(model, data, trajectory, speed=1.0)
        else:
            print(f"  未检测到图形环境，跳过可视化回放")
            print(f"  提示: 在有显示器的环境中运行可以看到 3D 回放")
    except Exception as e:
        print(f"  可视化回放跳过: {e}")

    # 总结
    print(f"\n{DIVIDER}")
    print("  【本章总结】")
    print(DIVIDER)
    print(f"""
  本章学到了:
    1. 如何加载真实机器人模型（从 robot_descriptions 或内置模型）
    2. 不同机器人类型的 qpos 结构差异（固定 vs 浮动基座）
    3. 如何在仿真中录制轨迹数据（qpos, qvel, ctrl, 末端位置）
    4. 多种数据格式的保存与加载（.npy, .npz, .pkl, .json）
    5. 轨迹回放的两种方式:
       - 设置 qpos + mj_forward（精确重现，推荐）
       - 设置 ctrl + mj_step（物理仿真，有累积误差）
    6. 轨迹质量的统计分析方法

  下一步:
    → 第 5 章: 机器人数据格式（HDF5, PKL, LeRobot 格式）
    """)
    print(DIVIDER)


if __name__ == "__main__":
    main()
