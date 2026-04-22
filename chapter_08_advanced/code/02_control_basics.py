"""
第 8 章 · 02 - 机器人控制基础 (Robot Control Fundamentals)

目标: 理解 PD/PID 控制器如何驱动机器人，以及控制参数如何影响数据特征。

核心知识点:
  1. PD 控制: τ = Kp·(q_des - q) + Kd·(q̇_des - q̇)
  2. 重力补偿: 抵消重力对控制的影响
  3. 轨迹跟踪: 跟踪正弦波参考轨迹
  4. P / PD / PID 控制器对比: 不同控制器的数据特征差异
  5. 控制参数对数据质量的影响

运行: python 02_control_basics.py
输出: control_comparison.png (控制器性能对比图)
依赖: pip install numpy matplotlib mujoco
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# ============================================================
# 单关节机械臂 MJCF（用于控制演示）
# ============================================================

SINGLE_JOINT_ARM_XML = """
<mujoco model="control_demo_arm">
  <option gravity="0 0 -9.81" timestep="0.001"/>

  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom type="sphere" size="0.04" rgba="0.3 0.3 0.3 1" mass="0"/>

      <body name="arm" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 1 0"
               limited="true" range="-180 180" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.02"
              rgba="0.4 0.6 0.8 1" mass="1.0"/>

        <body name="forearm" pos="0.4 0 0">
          <joint name="joint2" type="hinge" axis="0 1 0"
                 limited="true" range="-150 150" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.018"
                rgba="0.6 0.8 0.4 1" mass="0.7"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="joint1" ctrlrange="-50 50"/>
    <motor joint="joint2" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""


# ============================================================
# 第 1 节：控制器实现
# ============================================================

@dataclass
class ControllerGains:
    """控制器增益参数。"""
    kp: np.ndarray       # 比例增益
    kd: np.ndarray       # 微分增益
    ki: np.ndarray       # 积分增益

    @classmethod
    def p_only(cls, kp: List[float]) -> "ControllerGains":
        """仅比例控制。"""
        kp = np.array(kp)
        return cls(kp=kp, kd=np.zeros_like(kp), ki=np.zeros_like(kp))

    @classmethod
    def pd(cls, kp: List[float], kd: List[float]) -> "ControllerGains":
        """比例 + 微分控制。"""
        kp = np.array(kp)
        return cls(kp=kp, kd=np.array(kd), ki=np.zeros_like(kp))

    @classmethod
    def pid(cls, kp: List[float], kd: List[float],
            ki: List[float]) -> "ControllerGains":
        """比例 + 积分 + 微分控制。"""
        return cls(kp=np.array(kp), kd=np.array(kd), ki=np.array(ki))


class RobotController:
    """
    通用机器人控制器，支持 P / PD / PID + 可选重力补偿。

    控制律:
      τ = Kp·(q_des - q) + Kd·(q̇_des - q̇) + Ki·∫(q_des - q)dt + τ_gravity

    数据工程视角:
      不同控制器产生的数据有明显不同的统计特征:
      - P 控制: 稳态误差大，数据有系统性偏移
      - PD 控制: 振荡少但可能有稳态误差
      - PID 控制: 无稳态误差，但可能有积分饱和问题
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 gains: ControllerGains,
                 gravity_comp: bool = False,
                 integral_limit: float = 10.0):
        self.model = model
        self.gains = gains
        self.gravity_comp = gravity_comp
        self.integral_limit = integral_limit

        self.integral_error = np.zeros(model.nv)

    def reset(self) -> None:
        """重置积分项。"""
        self.integral_error[:] = 0

    def compute(self,
                data: mujoco.MjData,
                q_des: np.ndarray,
                qd_des: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算控制力矩。

        参数:
          data: 当前仿真状态
          q_des: 目标关节角度
          qd_des: 目标关节速度（默认为 0）
        """
        nv = self.model.nv
        if qd_des is None:
            qd_des = np.zeros(nv)

        # 误差计算
        error_pos = q_des - data.qpos[:nv]
        error_vel = qd_des - data.qvel[:nv]

        # 积分项（带防饱和）
        dt = self.model.opt.timestep
        self.integral_error += error_pos * dt
        self.integral_error = np.clip(
            self.integral_error, -self.integral_limit, self.integral_limit
        )

        # PID 控制
        tau = (self.gains.kp * error_pos +
               self.gains.kd * error_vel +
               self.gains.ki * self.integral_error)

        # 重力补偿
        if self.gravity_comp:
            tau += data.qfrc_bias[:nv]

        # 执行器力矩限制
        for i in range(min(nv, self.model.nu)):
            low, high = self.model.actuator_ctrlrange[i]
            tau[i] = np.clip(tau[i], low, high)

        return tau


# ============================================================
# 第 2 节：重力补偿演示
# ============================================================

def demo_gravity_compensation(model: mujoco.MjModel,
                              data: mujoco.MjData) -> Dict:
    """
    重力补偿演示：对比有/无重力补偿时的位置保持。

    没有重力补偿时，机械臂会因为重力而下垂（数据中表现为稳态偏移）。
    """
    print(f"\n{DIVIDER}")
    print("第 2 节：重力补偿")
    print(DIVIDER)

    target_pos = np.array([0.5, -0.3])
    duration = 3.0  # 秒
    dt = model.opt.timestep
    n_steps = int(duration / dt)

    results = {}

    for use_grav_comp in [False, True]:
        label = "有重力补偿" if use_grav_comp else "无重力补偿"
        gains = ControllerGains.pd(kp=[100, 80], kd=[10, 8])
        ctrl = RobotController(model, gains, gravity_comp=use_grav_comp)

        # 重置仿真
        mujoco.mj_resetData(model, data)
        data.qpos[:] = target_pos
        mujoco.mj_forward(model, data)

        positions = []
        for step in range(n_steps):
            tau = ctrl.compute(data, target_pos)
            data.ctrl[:] = tau[:model.nu]
            mujoco.mj_step(model, data)
            positions.append(data.qpos[:model.nv].copy())

        positions = np.array(positions)
        final_error = np.abs(positions[-1] - target_pos)

        results[label] = positions
        print(f"\n  {label}:")
        print(f"    最终误差: joint1 = {final_error[0]:.6f} rad, "
              f"joint2 = {final_error[1]:.6f} rad")

    return results


# ============================================================
# 第 3 节：轨迹跟踪（正弦波）
# ============================================================

def generate_sine_trajectory(duration: float, dt: float,
                             amplitude: float = 0.8,
                             frequency: float = 0.5) -> Dict[str, np.ndarray]:
    """
    生成正弦波参考轨迹。

    返回: 位置、速度、加速度的时间序列。
    """
    t = np.arange(0, duration, dt)
    omega = 2 * np.pi * frequency

    q_ref = amplitude * np.sin(omega * t)
    qd_ref = amplitude * omega * np.cos(omega * t)
    qdd_ref = -amplitude * omega ** 2 * np.sin(omega * t)

    return {
        "time": t,
        "q_ref": q_ref,
        "qd_ref": qd_ref,
        "qdd_ref": qdd_ref,
    }


def run_tracking_experiment(model: mujoco.MjModel,
                            data: mujoco.MjData,
                            controller: RobotController,
                            trajectory: Dict[str, np.ndarray],
                            label: str) -> Dict:
    """
    执行轨迹跟踪实验，记录所有数据。
    """
    controller.reset()
    mujoco.mj_resetData(model, data)

    t_arr = trajectory["time"]
    n_steps = len(t_arr)

    q_actual = np.zeros(n_steps)
    q_ref = np.zeros(n_steps)
    ctrl_signal = np.zeros(n_steps)
    errors = np.zeros(n_steps)

    for i in range(n_steps):
        # 两个关节: 第一个跟踪正弦波，第二个保持在 0
        q_des = np.array([trajectory["q_ref"][i], 0.0])
        qd_des = np.array([trajectory["qd_ref"][i], 0.0])

        tau = controller.compute(data, q_des, qd_des)
        data.ctrl[:] = tau[:model.nu]
        mujoco.mj_step(model, data)

        q_actual[i] = data.qpos[0]
        q_ref[i] = trajectory["q_ref"][i]
        ctrl_signal[i] = tau[0]
        errors[i] = q_ref[i] - q_actual[i]

    return {
        "label": label,
        "time": t_arr,
        "q_actual": q_actual,
        "q_ref": q_ref,
        "ctrl_signal": ctrl_signal,
        "errors": errors,
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "max_error": np.max(np.abs(errors)),
        "mean_abs_error": np.mean(np.abs(errors)),
    }


# ============================================================
# 第 4 节：P / PD / PID 控制器对比
# ============================================================

def demo_controller_comparison(model: mujoco.MjModel,
                               data: mujoco.MjData) -> List[Dict]:
    """
    对比三种控制器在轨迹跟踪上的表现。
    """
    print(f"\n{DIVIDER}")
    print("第 4 节：P / PD / PID 控制器对比")
    print(DIVIDER)

    duration = 6.0
    dt = model.opt.timestep
    traj = generate_sine_trajectory(duration, dt, amplitude=0.8, frequency=0.3)

    controllers = [
        ("P-only",  ControllerGains.p_only(kp=[200, 150]),        True),
        ("PD",      ControllerGains.pd(kp=[200, 150], kd=[20, 15]), True),
        ("PID",     ControllerGains.pid(kp=[200, 150], kd=[20, 15], ki=[50, 40]), True),
    ]

    all_results = []

    print(f"\n{'控制器':<12} {'RMSE (rad)':<15} {'最大误差':<15} {'平均误差':<15}")
    print(SUB_DIVIDER)

    for label, gains, grav_comp in controllers:
        ctrl = RobotController(model, gains, gravity_comp=grav_comp)
        result = run_tracking_experiment(model, data, ctrl, traj, label)
        all_results.append(result)

        print(f"{label:<12} {result['rmse']:<15.6f} "
              f"{result['max_error']:<15.6f} {result['mean_abs_error']:<15.6f}")

    return all_results


def plot_controller_comparison(results: List[Dict],
                               save_path: str = "control_comparison.png") -> None:
    """
    绘制控制器对比图：轨迹跟踪 + 误差 + 控制信号 + 误差分布。
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    # --- 子图 1: 轨迹跟踪 ---
    ax = axes[0, 0]
    ax.plot(results[0]["time"], results[0]["q_ref"],
            "k--", linewidth=2, label="Reference", alpha=0.7)
    for res, color in zip(results, colors):
        ax.plot(res["time"], res["q_actual"],
                color=color, linewidth=1.2, label=res["label"], alpha=0.8)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Joint angle (rad)", fontsize=11)
    ax.set_title("Trajectory Tracking", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- 子图 2: 跟踪误差 ---
    ax = axes[0, 1]
    for res, color in zip(results, colors):
        ax.plot(res["time"], res["errors"],
                color=color, linewidth=1.0, label=res["label"], alpha=0.8)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Tracking error (rad)", fontsize=11)
    ax.set_title("Tracking Error", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    # --- 子图 3: 控制信号 ---
    ax = axes[1, 0]
    for res, color in zip(results, colors):
        ax.plot(res["time"], res["ctrl_signal"],
                color=color, linewidth=1.0, label=res["label"], alpha=0.8)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Control torque (N·m)", fontsize=11)
    ax.set_title("Control Signal", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 子图 4: 误差分布 ---
    ax = axes[1, 1]
    for res, color in zip(results, colors):
        ax.hist(res["errors"], bins=60, color=color, alpha=0.5,
                label=f"{res['label']} (RMSE={res['rmse']:.4f})", edgecolor="white")
    ax.set_xlabel("Tracking error (rad)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Error Distribution", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("P vs PD vs PID Controller Comparison\n"
                 "Sine wave tracking with gravity compensation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ 控制器对比图已保存: {save_path}")


# ============================================================
# 主程序
# ============================================================

def main():
    print(DIVIDER)
    print("第 8 章 · 02 - 机器人控制基础")
    print("PD/PID 控制器 · 重力补偿 · 轨迹跟踪")
    print(DIVIDER)

    # --- 加载模型 ---
    model = mujoco.MjModel.from_xml_string(SINGLE_JOINT_ARM_XML)
    data = mujoco.MjData(model)

    print(f"\n模型概要:")
    print(f"  关节数: {model.njnt}, 自由度: {model.nv}")
    print(f"  执行器: {model.nu}")
    print(f"  仿真步长: {model.opt.timestep} s")

    # 第 2 节: 重力补偿
    demo_gravity_compensation(model, data)

    # 第 4 节: 控制器对比（包含了第 3 节的轨迹跟踪）
    results = demo_controller_comparison(model, data)
    plot_controller_comparison(results)

    # --- 数据工程视角总结 ---
    print(f"\n{DIVIDER}")
    print("数据工程视角总结:")
    print(SUB_DIVIDER)
    print("  1. P 控制器数据特征: 稳态误差大 → 数据有系统性偏移")
    print("  2. PD 控制器数据特征: 响应快、振荡少 → 数据更平滑")
    print("  3. PID 控制器数据特征: 无稳态误差 → 但可能有积分饱和")
    print("  4. 重力补偿: 数据中的偏移消失 → 数据质量指标改善")
    print("  5. 控制频率: 步长越小数据越密 → 存储和处理成本增加")
    print(f"\n  关键指标对比:")
    for r in results:
        print(f"    {r['label']:<8}: RMSE = {r['rmse']:.6f}, "
              f"MaxErr = {r['max_error']:.6f}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
