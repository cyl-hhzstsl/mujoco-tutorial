"""
第 8 章 · 05 - mink: 专业级逆运动学库

目标: 掌握 mink 库的核心 API（Configuration / FrameTask / solve_ik），
     对比手写 Jacobian IK 与 mink 的差异，理解约束优化在 IK 中的应用。

核心知识点:
  1. Configuration: 机器人状态管理
  2. FrameTask: 任务空间目标定义（位置 + 姿态）
  3. solve_ik: QP 求解器一次搞定多任务 + 多约束
  4. Limits: ConfigurationLimit / VelocityLimit 约束
  5. 与手写 IK 的性能和代码量对比

运行: python 05_mink_ik.py
依赖: pip install numpy matplotlib mujoco mink
"""

import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from typing import Dict, List

try:
    from mink import Configuration, FrameTask, solve_ik
    from mink import ConfigurationLimit, VelocityLimit
    from mink.lie import SE3

    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# ============================================================
# 3 连杆平面机械臂 MJCF 模型 (与 01_kinematics.py 一致)
# ============================================================

THREE_LINK_ARM_XML = """
<mujoco model="three_link_arm">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <default>
    <joint type="hinge" axis="0 1 0" limited="true" damping="0.5"/>
    <geom type="capsule" size="0.02" rgba="0.4 0.6 0.8 1"/>
  </default>

  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="sphere" size="0.04" rgba="0.3 0.3 0.3 1"/>

      <body name="link1" pos="0 0 0">
        <joint name="shoulder" range="-180 180"/>
        <geom fromto="0 0 0  0.3 0 0" size="0.02"/>

        <body name="link2" pos="0.3 0 0">
          <joint name="elbow" range="-150 150"/>
          <geom fromto="0 0 0  0.25 0 0" size="0.018"/>

          <body name="link3" pos="0.25 0 0">
            <joint name="wrist" range="-120 120"/>
            <geom fromto="0 0 0  0.2 0 0" size="0.015"/>

            <site name="end_effector" pos="0.2 0 0" size="0.025"
                  rgba="1 0.2 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="shoulder" ctrlrange="-50 50"/>
    <motor joint="elbow" ctrlrange="-30 30"/>
    <motor joint="wrist" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""

# ============================================================
# 辅助: 手写 Jacobian IK (简化版，用于对比)
# ============================================================


def jacobian_ik(model: mujoco.MjModel, data: mujoco.MjData,
                target: np.ndarray, site_name: str = "end_effector",
                max_iter: int = 300, tol: float = 1e-4,
                step_size: float = 0.5, damping: float = 1e-4
                ) -> Dict:
    """手写阻尼最小二乘 IK，用于基准对比。"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    nv = model.nv
    errors = []

    for i in range(max_iter):
        mujoco.mj_forward(model, data)
        current = data.site_xpos[site_id].copy()
        err_vec = target - current
        err = np.linalg.norm(err_vec)
        errors.append(err)

        if err < tol:
            return {"qpos": data.qpos.copy(), "error": err,
                    "iters": i + 1, "success": True, "errors": errors}

        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        JJT = jacp @ jacp.T + damping ** 2 * np.eye(3)
        dq = jacp.T @ np.linalg.solve(JJT, err_vec)
        data.qpos[:] += step_size * dq

        for j in range(model.njnt):
            if model.jnt_limited[j]:
                idx = model.jnt_qposadr[j]
                lo, hi = model.jnt_range[j]
                data.qpos[idx] = np.clip(data.qpos[idx], lo, hi)

    mujoco.mj_forward(model, data)
    final = data.site_xpos[site_id].copy()
    return {"qpos": data.qpos.copy(), "error": np.linalg.norm(target - final),
            "iters": max_iter, "success": False, "errors": errors}


# ============================================================
# 第 1 节: mink 基础 — Configuration + FrameTask + solve_ik
# ============================================================

def section_1_mink_basics(model: mujoco.MjModel) -> None:
    """mink 核心 API 演示。"""
    print(f"\n{DIVIDER}")
    print("第 1 节: mink 基础 — Configuration / FrameTask / solve_ik")
    print(DIVIDER)

    configuration = Configuration(model)
    configuration.update(q=np.array([0.1, -0.1, 0.05]))
    print(f"\n  Configuration 创建成功")
    print(f"    nq = {model.nq}, nv = {model.nv}")
    print(f"    初始 qpos = {configuration.data.qpos}")

    task = FrameTask(
        frame_name="end_effector",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,
    )
    print(f"\n  FrameTask 创建成功")
    print(f"    frame: end_effector (site)")
    print(f"    position_cost = 1.0, orientation_cost = 0.0")

    target = np.array([0.5, 0.0, 0.0])
    task.set_target(SE3.from_translation(target))
    print(f"    target position = {target}")

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    dt = 0.01

    print(f"\n  开始 IK 求解 (dt={dt}) ...")

    for i in range(500):
        vel = solve_ik(configuration, [task], dt, solver="daqp", damping=1e-3)
        configuration.integrate_inplace(vel, dt)

        ee = configuration.data.site_xpos[site_id]
        err = np.linalg.norm(target - ee)

        if i % 50 == 0 or err < 1e-4:
            print(f"    iter {i:>4d}: pos = [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]  "
                  f"error = {err:.6f}")

        if err < 1e-4:
            print(f"\n  收敛! 迭代 {i+1} 次, 最终误差 = {err:.6f}")
            q_deg = np.degrees(configuration.data.qpos)
            print(f"  关节角度: [{q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f}] 度")
            return

    print(f"  未收敛 (500 次迭代)")


# ============================================================
# 第 2 节: 多目标测试
# ============================================================

def section_2_multi_target(model: mujoco.MjModel) -> List[Dict]:
    """测试多个目标位置，收集结果用于后续对比。"""
    print(f"\n{DIVIDER}")
    print("第 2 节: 多目标 IK 测试")
    print(DIVIDER)

    targets = [
        ("正前方",         np.array([0.5, 0.0, 0.0])),
        ("右上方",         np.array([0.3, 0.0, 0.3])),
        ("近处",           np.array([0.15, 0.0, 0.1])),
        ("不可达 (太远)",  np.array([1.0, 0.0, 0.0])),
    ]

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    dt = 0.01
    results = []

    print(f"\n  {'目标':<18} {'目标位置':<24} {'实际位置':<24} "
          f"{'误差':<12} {'迭代':<8} {'状态'}")
    print(f"  {SUB_DIVIDER}")

    for name, target in targets:
        configuration = Configuration(model)
        configuration.data.qpos[:] = [0.3, -0.3, 0.1]
        configuration.update()

        task = FrameTask(
            frame_name="end_effector",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        task.set_target(SE3.from_translation(target))

        errors = []
        converged = False
        final_iter = 500

        for i in range(500):
            vel = solve_ik(configuration, [task], dt, solver="daqp", damping=1e-3)
            configuration.integrate_inplace(vel, dt)

            ee = configuration.data.site_xpos[site_id]
            err = np.linalg.norm(target - ee)
            errors.append(err)

            if err < 1e-4:
                converged = True
                final_iter = i + 1
                break

        ee = configuration.data.site_xpos[site_id].copy()
        err = np.linalg.norm(target - ee)
        status = "OK" if converged else "FAIL"

        t_str = f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]"
        e_str = f"[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]"
        print(f"  {name:<18} {t_str:<24} {e_str:<24} "
              f"{err:<12.6f} {final_iter:<8} {status}")

        results.append({
            "name": name, "target": target, "final_pos": ee,
            "error": err, "iters": final_iter, "success": converged,
            "errors": errors,
        })

    return results


# ============================================================
# 第 3 节: 带约束的 IK
# ============================================================

def section_3_constrained_ik(model: mujoco.MjModel) -> None:
    """演示 ConfigurationLimit 和 VelocityLimit。"""
    print(f"\n{DIVIDER}")
    print("第 3 节: 带约束的 IK (ConfigurationLimit + VelocityLimit)")
    print(DIVIDER)

    target = np.array([0.3, 0.0, 0.3])
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    dt = 0.01

    config_limit = ConfigurationLimit(model=model)
    velocity_limit = VelocityLimit(
        model,
        velocities={"shoulder": np.pi, "elbow": np.pi, "wrist": np.pi},
    )
    limits = [config_limit, velocity_limit]

    print(f"\n  约束:")
    print(f"    ConfigurationLimit: 从 MJCF 模型读取关节范围")
    print(f"    VelocityLimit: 所有关节最大 pi rad/s")

    scenarios = [
        ("无约束", []),
        ("有约束", limits),
    ]

    for label, lims in scenarios:
        configuration = Configuration(model)
        configuration.data.qpos[:] = [0.0, 0.0, 0.0]
        configuration.update()

        task = FrameTask(
            frame_name="end_effector",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        task.set_target(SE3.from_translation(target))

        max_dq_per_step = []

        for i in range(300):
            vel = solve_ik(configuration, [task], dt, solver="daqp",
                           damping=1e-3, limits=lims)
            dq = np.abs(vel[:model.nv] * dt)
            max_dq_per_step.append(dq.max())
            configuration.integrate_inplace(vel, dt)

            ee = configuration.data.site_xpos[site_id]
            if np.linalg.norm(target - ee) < 1e-4:
                break

        ee = configuration.data.site_xpos[site_id].copy()
        err = np.linalg.norm(target - ee)
        q_deg = np.degrees(configuration.data.qpos)
        peak_dq = max(max_dq_per_step) if max_dq_per_step else 0

        print(f"\n  [{label}]")
        print(f"    最终误差: {err:.6f}")
        print(f"    迭代次数: {i+1}")
        print(f"    关节角度: [{q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f}] 度")
        print(f"    单步最大 |dq|: {peak_dq:.4f} rad")

        for j in range(model.njnt):
            if model.jnt_limited[j]:
                idx = model.jnt_qposadr[j]
                lo, hi = model.jnt_range[j]
                q = configuration.data.qpos[idx]
                in_range = lo <= q <= hi
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                print(f"    关节 {jname}: {np.degrees(q):>7.1f} deg  "
                      f"范围 [{np.degrees(lo):.0f}, {np.degrees(hi):.0f}]  "
                      f"{'OK' if in_range else 'VIOLATED!'}")


# ============================================================
# 第 4 节: mink vs 手写 IK 性能对比
# ============================================================

def section_4_comparison(model: mujoco.MjModel) -> Dict:
    """对比 mink 和手写 Jacobian IK 的收敛速度和耗时。"""
    print(f"\n{DIVIDER}")
    print("第 4 节: mink vs 手写 Jacobian IK 对比")
    print(DIVIDER)

    targets = [
        ("正前方", np.array([0.5, 0.0, 0.0])),
        ("右上方", np.array([0.3, 0.0, 0.3])),
        ("近处",   np.array([0.15, 0.0, 0.1])),
    ]

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    dt = 0.01
    comparison = {}

    print(f"\n  {'目标':<12} {'方法':<16} {'误差':<12} {'迭代':<8} {'耗时(ms)':<12} {'状态'}")
    print(f"  {SUB_DIVIDER}")

    for name, target in targets:
        # --- 手写 IK ---
        data_manual = mujoco.MjData(model)
        data_manual.qpos[:] = [0.3, -0.3, 0.1]

        t0 = time.perf_counter()
        manual_result = jacobian_ik(model, data_manual, target)
        t_manual = (time.perf_counter() - t0) * 1000

        # --- mink IK ---
        config = Configuration(model)
        config.data.qpos[:] = [0.3, -0.3, 0.1]
        config.update()

        task = FrameTask(
            frame_name="end_effector",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        task.set_target(SE3.from_translation(target))

        mink_errors = []
        t0 = time.perf_counter()
        mink_iters = 500
        mink_success = False

        for i in range(500):
            vel = solve_ik(config, [task], dt, solver="daqp", damping=1e-3)
            config.integrate_inplace(vel, dt)

            ee = config.data.site_xpos[site_id]
            err = np.linalg.norm(target - ee)
            mink_errors.append(err)

            if err < 1e-4:
                mink_iters = i + 1
                mink_success = True
                break

        t_mink = (time.perf_counter() - t0) * 1000
        mink_err = mink_errors[-1]

        print(f"  {name:<12} {'手写 Jacobian':<16} {manual_result['error']:<12.6f} "
              f"{manual_result['iters']:<8} {t_manual:<12.2f} "
              f"{'OK' if manual_result['success'] else 'FAIL'}")
        print(f"  {'':<12} {'mink':<16} {mink_err:<12.6f} "
              f"{mink_iters:<8} {t_mink:<12.2f} "
              f"{'OK' if mink_success else 'FAIL'}")

        comparison[name] = {
            "manual_errors": manual_result["errors"],
            "mink_errors": mink_errors,
        }

    return comparison


# ============================================================
# 第 5 节: 可视化收敛对比
# ============================================================

def plot_comparison(comparison: Dict, save_path: str = "mink_vs_manual_ik.png") -> None:
    """绘制 mink vs 手写 IK 的收敛曲线。"""
    n = len(comparison)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, comparison.items()):
        manual = data["manual_errors"]
        mink = data["mink_errors"]

        ax.semilogy(manual, label="Manual Jacobian IK", color="steelblue",
                     linewidth=2)
        ax.semilogy(mink, label="mink (QP)", color="coral",
                     linewidth=2)
        ax.axhline(y=1e-4, color="gray", linestyle="--", alpha=0.5,
                    label="Tolerance (1e-4)")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Position Error (m)", fontsize=11)
        ax.set_title(f"Target: {name}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("IK Convergence: mink vs Manual Jacobian", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  图表已保存: {save_path}")


# ============================================================
# 第 6 节: 用 mink 生成轨迹数据
# ============================================================

def section_6_trajectory_generation(model: mujoco.MjModel) -> None:
    """演示用 mink 生成机器人轨迹数据（数据工程实用场景）。"""
    print(f"\n{DIVIDER}")
    print("第 6 节: 用 mink 生成轨迹数据 (数据工程实战)")
    print(DIVIDER)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    dt = 0.01

    n_waypoints = 6
    t_vals = np.linspace(0, 2 * np.pi, n_waypoints, endpoint=False)
    radius = 0.15
    center = np.array([0.4, 0.0, 0.0])
    waypoints = []
    for t in t_vals:
        wp = center + np.array([radius * np.cos(t), 0, radius * np.sin(t)])
        waypoints.append(wp)
    waypoints.append(waypoints[0])

    print(f"\n  圆形轨迹: 中心 = {center}, 半径 = {radius} m")
    print(f"  路径点数: {len(waypoints)}")

    trajectory = {"qpos": [], "ee_pos": [], "timestamp": []}
    configuration = Configuration(model)
    configuration.data.qpos[:] = [0.3, -0.3, 0.1]
    configuration.update()

    task = FrameTask(
        frame_name="end_effector",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,
    )

    config_limit = ConfigurationLimit(model=model)
    velocity_limit = VelocityLimit(
        model,
        velocities={"shoulder": np.pi, "elbow": np.pi, "wrist": np.pi},
    )
    limits = [config_limit, velocity_limit]

    step_count = 0
    for wp_idx, wp in enumerate(waypoints):
        task.set_target(SE3.from_translation(wp))

        for i in range(200):
            vel = solve_ik(configuration, [task], dt, solver="daqp",
                           damping=1e-3, limits=limits)
            configuration.integrate_inplace(vel, dt)

            ee = configuration.data.site_xpos[site_id].copy()
            trajectory["qpos"].append(configuration.data.qpos.copy())
            trajectory["ee_pos"].append(ee)
            trajectory["timestamp"].append(step_count * dt)
            step_count += 1

            if np.linalg.norm(wp - ee) < 5e-4:
                break

        ee = configuration.data.site_xpos[site_id]
        err = np.linalg.norm(wp - ee)
        print(f"  路径点 {wp_idx}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]  "
              f"误差 = {err:.5f}  步数 = {i+1}")

    trajectory["qpos"] = np.array(trajectory["qpos"])
    trajectory["ee_pos"] = np.array(trajectory["ee_pos"])
    trajectory["timestamp"] = np.array(trajectory["timestamp"])

    print(f"\n  轨迹统计:")
    print(f"    总步数: {len(trajectory['timestamp'])}")
    print(f"    总时长: {trajectory['timestamp'][-1]:.2f} s")
    print(f"    qpos shape: {trajectory['qpos'].shape}")
    print(f"    ee_pos shape: {trajectory['ee_pos'].shape}")

    plot_trajectory(trajectory)


def plot_trajectory(trajectory: Dict, save_path: str = "mink_trajectory.png") -> None:
    """绘制 mink 生成的轨迹。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ee = trajectory["ee_pos"]
    qpos = trajectory["qpos"]
    t = trajectory["timestamp"]

    ax = axes[0]
    ax.plot(ee[:, 0], ee[:, 2], "b-", linewidth=1.5, alpha=0.8)
    ax.plot(ee[0, 0], ee[0, 2], "go", markersize=10, label="Start")
    ax.plot(ee[-1, 0], ee[-1, 2], "r*", markersize=12, label="End")
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Z (m)", fontsize=11)
    ax.set_title("End-Effector Path (XZ plane)", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    joint_names = ["shoulder", "elbow", "wrist"]
    for j in range(qpos.shape[1]):
        ax.plot(t, np.degrees(qpos[:, j]), linewidth=1.5, label=joint_names[j])
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Joint Angle (deg)", fontsize=11)
    ax.set_title("Joint Trajectories", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    dq = np.diff(qpos, axis=0) / np.diff(t).reshape(-1, 1)
    t_vel = t[:-1]
    for j in range(dq.shape[1]):
        ax.plot(t_vel, np.degrees(dq[:, j]), linewidth=1.5, label=joint_names[j])
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Joint Velocity (deg/s)", fontsize=11)
    ax.set_title("Joint Velocities", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("mink IK Trajectory Generation", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {save_path}")


# ============================================================
# 主程序
# ============================================================

def main():
    print(DIVIDER)
    print("第 8 章 · 05 - mink: 专业级逆运动学库")
    print(DIVIDER)

    if not MINK_AVAILABLE:
        print("\n  mink 未安装。请先运行:")
        print("    pip install mink")
        print("\n  安装后重新运行本脚本。")
        print(f"\n  以下演示手写 Jacobian IK 作为参考...\n")

        model = mujoco.MjModel.from_xml_string(THREE_LINK_ARM_XML)
        data = mujoco.MjData(model)

        target = np.array([0.5, 0.0, 0.0])
        data.qpos[:] = [0.3, -0.3, 0.1]
        result = jacobian_ik(model, data, target)
        print(f"  手写 IK 结果: 误差={result['error']:.6f}, "
              f"迭代={result['iters']}, "
              f"成功={'是' if result['success'] else '否'}")
        print(f"\n  安装 mink 后可以看到完整对比!")
        return

    model = mujoco.MjModel.from_xml_string(THREE_LINK_ARM_XML)

    print(f"\n模型概要:")
    print(f"  关节数: {model.njnt}")
    print(f"  自由度: {model.nv}")
    print(f"  连杆长度: 0.3 + 0.25 + 0.2 = 0.75 m")

    section_1_mink_basics(model)

    section_2_multi_target(model)

    section_3_constrained_ik(model)

    comparison = section_4_comparison(model)
    plot_comparison(comparison)

    section_6_trajectory_generation(model)

    print(f"\n{DIVIDER}")
    print("完成! 关键收获:")
    print("  1. mink 把 IK 表述为 QP 问题，比手写 Jacobian 伪逆更安全、更强大")
    print("  2. Configuration + FrameTask + solve_ik 三件套覆盖大多数场景")
    print("  3. Limits 让关节限位和速度约束参与优化，而非事后 clip")
    print("  4. 用 mink 生成轨迹数据比手写 IK 更可靠，适合数据工程流水线")
    print(DIVIDER)


if __name__ == "__main__":
    main()
