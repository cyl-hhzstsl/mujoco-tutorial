"""
第 4 章 · 03 - 仿真轨迹录制

目标: 在仿真中运行机器人，录制完整的轨迹数据，
     并保存为多种格式（.npy, .npz, .pkl, .json），
     对比不同格式的文件大小和特点。

核心知识点:
  1. 控制信号的生成（正弦波扫频）
  2. 仿真循环中的数据采集
  3. 多种序列化格式的优劣
  4. 数据完整性验证

运行: python 03_trajectory_recording.py
"""

import mujoco
import numpy as np
import os
import time

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# ============================================================
# 内置 6DOF 机械臂模型
# ============================================================
ARM_XML = """
<mujoco model="6dof_arm">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="2 2 0.01" rgba="0.9 0.9 0.9 1"/>

    <!-- 基座 -->
    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.08 0.05" rgba="0.2 0.2 0.2 1" mass="5"/>

      <!-- 关节 1: 基座旋转 -->
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="2"/>
        <geom type="cylinder" size="0.05 0.08" rgba="0.8 0.3 0.3 1" mass="3"/>

        <!-- 关节 2: 肩部 -->
        <body name="link2" pos="0 0 0.08">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2.36 2.36" damping="2"/>
          <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.3" rgba="0.3 0.3 0.8 1" mass="2.5"/>

          <!-- 关节 3: 肘部 -->
          <body name="link3" pos="0 0 0.3">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-2.36 2.36" damping="1.5"/>
            <geom type="capsule" size="0.035" fromto="0 0 0 0 0 0.25" rgba="0.3 0.8 0.3 1" mass="2"/>

            <!-- 关节 4: 腕1 -->
            <body name="link4" pos="0 0 0.25">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1"/>
              <geom type="cylinder" size="0.03 0.04" rgba="0.8 0.8 0.3 1" mass="1"/>

              <!-- 关节 5: 腕2 -->
              <body name="link5" pos="0 0 0.04">
                <joint name="joint5" type="hinge" axis="0 1 0" range="-2.09 2.09" damping="0.5"/>
                <geom type="cylinder" size="0.025 0.03" rgba="0.8 0.3 0.8 1" mass="0.5"/>

                <!-- 关节 6: 腕3 -->
                <body name="link6" pos="0 0 0.03">
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.5"/>
                  <geom type="cylinder" size="0.02 0.02" rgba="0.3 0.8 0.8 1" mass="0.3"/>

                  <!-- 末端执行器标记 -->
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
    <jointvel name="jvel1" joint="joint1"/>
    <jointvel name="jvel2" joint="joint2"/>
    <jointvel name="jvel3" joint="joint3"/>
    <jointvel name="jvel4" joint="joint4"/>
    <jointvel name="jvel5" joint="joint5"/>
    <jointvel name="jvel6" joint="joint6"/>
  </sensor>
</mujoco>
"""


# ============================================================
# 控制信号生成
# ============================================================
def generate_control_signal(t, nu):
    """
    为每个关节生成不同频率和幅度的正弦波控制信号。
    模拟一个 "扫频" 运动，让机械臂各关节同时运动。

    参数:
      t  - 当前仿真时间（秒）
      nu - 执行器数量
    返回:
      ctrl - 长度为 nu 的控制向量
    """
    ctrl = np.zeros(nu)
    for i in range(nu):
        freq = 0.3 + i * 0.15      # 递增频率: 0.3, 0.45, 0.6, ...
        amp = 0.5 / (1 + i * 0.3)   # 递减幅度: 越远端越小
        phase = i * np.pi / 4       # 各关节相位差
        ctrl[i] = amp * np.sin(2 * np.pi * freq * t + phase)
    return ctrl


# ============================================================
# 仿真与录制
# ============================================================
def run_simulation_and_record(model, data, duration=5.0):
    """
    运行仿真并录制轨迹数据。

    参数:
      model    - MjModel
      data     - MjData
      duration - 仿真时长（秒）
    返回:
      trajectory - 字典，包含所有录制的数据
    """
    dt = model.opt.timestep
    n_steps = int(duration / dt)
    record_every = 10  # 每 10 步记录一次（降采样）
    n_records = n_steps // record_every

    print(f"\n  仿真参数:")
    print(f"    时间步长 dt    = {dt} s")
    print(f"    仿真时长       = {duration} s")
    print(f"    总步数         = {n_steps}")
    print(f"    记录间隔       = 每 {record_every} 步")
    print(f"    预计记录帧数   = {n_records}")
    print(f"    记录频率       = {1.0 / (dt * record_every):.1f} Hz")

    # 查找末端执行器 site
    ee_site_id = -1
    try:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    except Exception:
        pass

    # 预分配数组
    times = np.zeros(n_records)
    qpos_log = np.zeros((n_records, model.nq))
    qvel_log = np.zeros((n_records, model.nv))
    ctrl_log = np.zeros((n_records, model.nu))
    sensor_log = np.zeros((n_records, model.nsensordata))
    ee_pos_log = np.zeros((n_records, 3))

    # 仿真循环
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    record_idx = 0

    print(f"\n  开始仿真...")
    t_start = time.time()

    for step in range(n_steps):
        # 生成控制信号
        ctrl = generate_control_signal(data.time, model.nu)
        data.ctrl[:] = ctrl

        # 仿真一步
        mujoco.mj_step(model, data)

        # 按间隔记录
        if step % record_every == 0 and record_idx < n_records:
            times[record_idx] = data.time
            qpos_log[record_idx] = data.qpos.copy()
            qvel_log[record_idx] = data.qvel.copy()
            ctrl_log[record_idx] = ctrl.copy()
            if model.nsensordata > 0:
                sensor_log[record_idx] = data.sensordata.copy()
            if ee_site_id >= 0:
                ee_pos_log[record_idx] = data.site_xpos[ee_site_id].copy()
            record_idx += 1

    wall_time = time.time() - t_start
    actual_records = record_idx

    print(f"  仿真完成!")
    print(f"    实际录制帧数   = {actual_records}")
    print(f"    墙钟时间       = {wall_time:.3f} s")
    print(f"    实时倍率       = {duration / wall_time:.1f}x")

    # 裁剪到实际录制长度
    trajectory = {
        "time": times[:actual_records],
        "qpos": qpos_log[:actual_records],
        "qvel": qvel_log[:actual_records],
        "ctrl": ctrl_log[:actual_records],
        "sensor_data": sensor_log[:actual_records],
        "ee_pos": ee_pos_log[:actual_records],
        "metadata": {
            "model_name": "6dof_arm",
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "dt": dt,
            "record_dt": dt * record_every,
            "duration": duration,
            "n_frames": actual_records,
            "record_hz": 1.0 / (dt * record_every),
            "joint_names": [model.joint(j).name for j in range(model.njnt)],
            "actuator_names": [model.actuator(a).name for a in range(model.nu)],
        },
    }
    return trajectory


# ============================================================
# 多格式保存
# ============================================================
def save_trajectory(trajectory, output_dir):
    """将轨迹保存为多种格式，对比文件大小。"""
    import json
    import pickle

    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # --- 1. NumPy .npy (只存 qpos) ---
    path_npy = os.path.join(output_dir, "trajectory_qpos.npy")
    np.save(path_npy, trajectory["qpos"])
    saved_files[".npy (qpos only)"] = path_npy

    # --- 2. NumPy .npz (多个数组) ---
    path_npz = os.path.join(output_dir, "trajectory.npz")
    np.savez_compressed(
        path_npz,
        time=trajectory["time"],
        qpos=trajectory["qpos"],
        qvel=trajectory["qvel"],
        ctrl=trajectory["ctrl"],
        sensor_data=trajectory["sensor_data"],
        ee_pos=trajectory["ee_pos"],
    )
    saved_files[".npz (compressed)"] = path_npz

    # --- 3. Pickle .pkl (完整字典，含 metadata) ---
    path_pkl = os.path.join(output_dir, "trajectory.pkl")
    with open(path_pkl, "wb") as f:
        pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
    saved_files[".pkl (full dict)"] = path_pkl

    # --- 4. JSON (只存 metadata + 统计摘要) ---
    path_json = os.path.join(output_dir, "trajectory_meta.json")
    json_data = {
        "metadata": trajectory["metadata"],
        "statistics": {
            "qpos_mean": trajectory["qpos"].mean(axis=0).tolist(),
            "qpos_std": trajectory["qpos"].std(axis=0).tolist(),
            "qpos_min": trajectory["qpos"].min(axis=0).tolist(),
            "qpos_max": trajectory["qpos"].max(axis=0).tolist(),
            "qvel_abs_max": float(np.abs(trajectory["qvel"]).max()),
            "ee_pos_range": {
                "x": [float(trajectory["ee_pos"][:, 0].min()),
                       float(trajectory["ee_pos"][:, 0].max())],
                "y": [float(trajectory["ee_pos"][:, 1].min()),
                       float(trajectory["ee_pos"][:, 1].max())],
                "z": [float(trajectory["ee_pos"][:, 2].min()),
                       float(trajectory["ee_pos"][:, 2].max())],
            },
        },
    }
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    saved_files[".json (metadata)"] = path_json

    return saved_files


# ============================================================
# 文件大小对比
# ============================================================
def print_file_sizes(saved_files):
    """打印各文件的大小对比。"""
    print(f"\n  【文件大小对比】")
    print(f"  {'格式':25s}  {'文件名':35s}  {'大小':>12s}")
    print(f"  {'-'*25}  {'-'*35}  {'-'*12}")

    for fmt, path in saved_files.items():
        size = os.path.getsize(path)
        fname = os.path.basename(path)
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.2f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {fmt:25s}  {fname:35s}  {size_str:>12s}")


# ============================================================
# 数据完整性验证
# ============================================================
def verify_saved_data(trajectory, output_dir):
    """从各格式重新加载数据，验证完整性。"""
    import pickle

    print(f"\n  【数据完整性验证】")
    all_ok = True

    # 验证 .npy
    print(f"\n  1. 验证 .npy ...")
    loaded_qpos = np.load(os.path.join(output_dir, "trajectory_qpos.npy"))
    if np.allclose(loaded_qpos, trajectory["qpos"]):
        print(f"     ✓ qpos 数据完全一致  shape={loaded_qpos.shape}")
    else:
        print(f"     ✗ qpos 数据不一致!")
        all_ok = False

    # 验证 .npz
    print(f"  2. 验证 .npz ...")
    with np.load(os.path.join(output_dir, "trajectory.npz")) as npz:
        keys = list(npz.keys())
        print(f"     包含的数组: {keys}")
        for key in ["time", "qpos", "qvel", "ctrl"]:
            if key in npz:
                if np.allclose(npz[key], trajectory[key]):
                    print(f"     ✓ {key:12s} 一致  shape={npz[key].shape}")
                else:
                    print(f"     ✗ {key} 不一致!")
                    all_ok = False

    # 验证 .pkl
    print(f"  3. 验证 .pkl ...")
    with open(os.path.join(output_dir, "trajectory.pkl"), "rb") as f:
        loaded_traj = pickle.load(f)
    for key in ["time", "qpos", "qvel", "ctrl"]:
        if np.allclose(loaded_traj[key], trajectory[key]):
            print(f"     ✓ {key:12s} 一致  shape={loaded_traj[key].shape}")
        else:
            print(f"     ✗ {key} 不一致!")
            all_ok = False
    if "metadata" in loaded_traj:
        print(f"     ✓ metadata 已保存  模型={loaded_traj['metadata']['model_name']}")

    # 验证 .json
    print(f"  4. 验证 .json ...")
    import json
    with open(os.path.join(output_dir, "trajectory_meta.json"), "r") as f:
        meta = json.load(f)
    if "metadata" in meta and "statistics" in meta:
        print(f"     ✓ JSON 结构完整  关节数={meta['metadata']['nq']}")
        print(f"     ✓ 统计信息包含: {list(meta['statistics'].keys())}")
    else:
        print(f"     ✗ JSON 结构不完整!")
        all_ok = False

    if all_ok:
        print(f"\n  ✅ 所有格式验证通过!")
    else:
        print(f"\n  ⚠ 部分格式验证失败，请检查")


# ============================================================
# 打印轨迹摘要
# ============================================================
def print_trajectory_summary(trajectory):
    """打印轨迹数据的统计摘要。"""
    print(f"\n{SUB_DIVIDER}")
    print(f"  【轨迹统计摘要】")
    print(SUB_DIVIDER)

    t = trajectory["time"]
    qpos = trajectory["qpos"]
    qvel = trajectory["qvel"]
    ctrl = trajectory["ctrl"]
    ee = trajectory["ee_pos"]
    meta = trajectory["metadata"]

    print(f"    帧数:        {meta['n_frames']}")
    print(f"    时间范围:    [{t[0]:.3f}, {t[-1]:.3f}] s")
    print(f"    记录频率:    {meta['record_hz']:.1f} Hz")

    print(f"\n    qpos 统计 (shape {qpos.shape}):")
    for j in range(min(qpos.shape[1], 6)):
        jname = meta["joint_names"][j] if j < len(meta["joint_names"]) else f"q{j}"
        print(f"      {jname:12s}: mean={qpos[:, j].mean():7.3f}  "
              f"std={qpos[:, j].std():6.3f}  "
              f"range=[{qpos[:, j].min():7.3f}, {qpos[:, j].max():7.3f}]")

    print(f"\n    qvel 绝对值最大: {np.abs(qvel).max():.4f} rad/s")
    print(f"    ctrl 绝对值最大: {np.abs(ctrl).max():.4f}")

    if np.any(ee != 0):
        print(f"\n    末端执行器运动范围:")
        for dim, label in enumerate(["x", "y", "z"]):
            print(f"      {label}: [{ee[:, dim].min():.4f}, {ee[:, dim].max():.4f}] m")


# ============================================================
# 主程序
# ============================================================
def main():
    print(DIVIDER)
    print("  第 4 章 · 03 - 仿真轨迹录制")
    print(DIVIDER)

    # 1. 加载模型
    print(f"\n【步骤 1】加载机械臂模型...")
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)
    print(f"  模型: {model.nq} DOF 机械臂")
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}, nsensor={model.nsensor}")

    # 2. 运行仿真并录制
    print(f"\n【步骤 2】运行仿真并录制轨迹...")
    trajectory = run_simulation_and_record(model, data, duration=5.0)

    # 3. 打印统计摘要
    print_trajectory_summary(trajectory)

    # 4. 保存到多种格式
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recorded_data")
    print(f"\n【步骤 3】保存轨迹数据到 {output_dir}/")
    saved_files = save_trajectory(trajectory, output_dir)
    print_file_sizes(saved_files)

    # 5. 验证数据完整性
    print(f"\n【步骤 4】验证保存的数据...")
    verify_saved_data(trajectory, output_dir)

    # 6. 格式选择建议
    print(f"\n{SUB_DIVIDER}")
    print(f"  【格式选择建议】")
    print(SUB_DIVIDER)
    print(f"""
    格式        适合场景                    优点                   缺点
    ──────────────────────────────────────────────────────────────────────────
    .npy       单个数组快速读写            极快，零开销            只能存单个数组
    .npz       多个数组打包                可压缩，支持懒加载      不支持任意对象
    .pkl       完整 Python 对象            灵活，保留类型          仅 Python 可读
    .json      元数据/配置                 人类可读，跨语言        不适合大数组
    .hdf5      大规模数据集（第 5 章）      分层存储，部分读取      需要 h5py 库
    """)

    print(f"\n{DIVIDER}")
    print(f"  ✅ 录制完成！下一步: python 04_trajectory_replay.py")
    print(DIVIDER)


if __name__ == "__main__":
    main()
