"""
第 8 章 · 03 - Sim-to-Real 概念 (Sim-to-Real for Data Engineers)

目标: 理解仿真与真实之间的差距（sim-to-real gap），
     掌握域随机化技术及其对数据分布的影响。

核心知识点:
  1. Sim-to-Real Gap: 仿真数据与真实数据的分布差异
  2. 域随机化 (Domain Randomization): 随机扰动物理参数
  3. 参数敏感性: 不同参数对数据分布的影响程度
  4. 数据增强: 针对机器人数据的增强技术
  5. DomainRandomizer 类: 可复用的域随机化工具

数据工程视角:
  - 训练数据的分布必须覆盖真实世界的变化范围
  - 域随机化是 sim-to-real 最简单有效的方法
  - 数据工程师需要理解每个参数如何影响数据分布

运行: python 03_sim_to_real.py
输出: sim2real_distributions.png (参数随机化对数据分布的影响)
依赖: pip install numpy matplotlib mujoco
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# ============================================================
# 演示用 MJCF 模型（带参数标注）
# ============================================================

BASE_ARM_XML = """
<mujoco model="sim2real_arm">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <default>
    <joint type="hinge" axis="0 1 0" limited="true" damping="0.5"/>
    <geom type="capsule" size="0.02" rgba="0.5 0.7 0.9 1"
          friction="1.0 0.005 0.0001"/>
  </default>

  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom type="sphere" size="0.04" mass="0"/>

      <body name="link1" pos="0 0 0">
        <joint name="j1" range="-180 180"/>
        <geom name="g1" fromto="0 0 0  0.3 0 0" mass="1.0"/>

        <body name="link2" pos="0.3 0 0">
          <joint name="j2" range="-150 150"/>
          <geom name="g2" fromto="0 0 0  0.25 0 0" mass="0.8"/>

          <site name="end_effector" pos="0.25 0 0" size="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="j1" ctrlrange="-50 50"/>
    <motor joint="j2" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""


# ============================================================
# 第 1 节：Sim-to-Real Gap 解释
# ============================================================

def explain_sim2real_gap() -> None:
    """解释什么是 sim-to-real gap 及其对数据的影响。"""
    print(f"\n{DIVIDER}")
    print("第 1 节：什么是 Sim-to-Real Gap?")
    print(DIVIDER)

    print("""
  仿真 (Simulation)              真实 (Real World)
  ─────────────────              ────────────────
  完美的刚体物理                  柔性、间隙、磨损
  精确的参数 (质量=1.0kg)         参数不确定 (0.9~1.1kg)
  无传感器噪声                    传感器有噪声和延迟
  完美的执行器                    执行器有摩擦、死区
  确定性仿真                      环境随机变化

  这些差异导致:
  ┌──────────────────────────────────────────────┐
  │  仿真数据分布  ≠  真实数据分布                    │
  │  在仿真中训练的模型 → 到真实中性能下降             │
  └──────────────────────────────────────────────┘

  解决方案: 域随机化 (Domain Randomization)
  ─────────
  在仿真中随机改变物理参数 → 生成更多样的数据
  → 覆盖真实世界可能的参数范围 → 模型泛化能力提升

  数据工程师需要关注:
  1. 数据集中的参数分布是否覆盖了真实世界的变化
  2. 不同参数设置下数据的统计特征如何变化
  3. 如何设计数据管道来管理多参数变体的数据
""")


# ============================================================
# 第 2 节：DomainRandomizer 类
# ============================================================

@dataclass
class RandomizationRange:
    """单个参数的随机化范围。"""
    param_name: str
    nominal: float           # 标称值（默认值）
    low: float               # 最小值
    high: float              # 最大值
    distribution: str = "uniform"  # "uniform" 或 "gaussian"
    std: Optional[float] = None    # gaussian 模式的标准差

    def sample(self, rng: np.random.Generator) -> float:
        """从分布中采样一个值。"""
        if self.distribution == "uniform":
            return rng.uniform(self.low, self.high)
        elif self.distribution == "gaussian":
            std = self.std if self.std else (self.high - self.low) / 6
            val = rng.normal(self.nominal, std)
            return np.clip(val, self.low, self.high)
        else:
            raise ValueError(f"未知分布: {self.distribution}")


class DomainRandomizer:
    """
    域随机化器：随机扰动 MuJoCo 模型的物理参数。

    支持随机化的参数:
      - 质量 (body mass)
      - 摩擦系数 (geom friction)
      - 阻尼 (joint damping)
      - 仿真步长 (timestep)
      - 重力 (gravity)

    使用模式:
      randomizer = DomainRandomizer(base_xml)
      randomizer.add_mass_range("link1", 0.8, 1.2)
      randomizer.add_friction_range("g1", 0.5, 1.5)
      model, params = randomizer.sample()  # 返回随机化后的模型
    """

    def __init__(self, base_xml: str, seed: int = 42):
        self.base_xml = base_xml
        self.rng = np.random.default_rng(seed)
        self.ranges: Dict[str, List[RandomizationRange]] = {
            "mass": [],
            "friction": [],
            "damping": [],
            "timestep": [],
            "gravity": [],
        }

    def add_mass_range(self, geom_name: str,
                       low: float, high: float,
                       distribution: str = "uniform") -> "DomainRandomizer":
        """添加质量随机化范围（通过 geom 找到所属 body 的质量）。"""
        base_model = mujoco.MjModel.from_xml_string(self.base_xml)
        geom_id = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        body_id = base_model.geom_bodyid[geom_id]
        nominal = base_model.body_mass[body_id] if base_model.body_mass[body_id] > 0 else 1.0
        self.ranges["mass"].append(RandomizationRange(
            param_name=geom_name, nominal=float(nominal),
            low=low, high=high, distribution=distribution
        ))
        return self

    def add_friction_range(self, geom_name: str,
                           low: float, high: float,
                           distribution: str = "uniform") -> "DomainRandomizer":
        """添加摩擦系数随机化范围。"""
        self.ranges["friction"].append(RandomizationRange(
            param_name=geom_name, nominal=1.0,
            low=low, high=high, distribution=distribution
        ))
        return self

    def add_damping_range(self, joint_name: str,
                          low: float, high: float,
                          distribution: str = "uniform") -> "DomainRandomizer":
        """添加关节阻尼随机化范围。"""
        self.ranges["damping"].append(RandomizationRange(
            param_name=joint_name, nominal=0.5,
            low=low, high=high, distribution=distribution
        ))
        return self

    def add_timestep_range(self, low: float, high: float) -> "DomainRandomizer":
        """添加仿真步长随机化范围。"""
        self.ranges["timestep"].append(RandomizationRange(
            param_name="timestep", nominal=0.002,
            low=low, high=high
        ))
        return self

    def add_gravity_range(self, low: float, high: float) -> "DomainRandomizer":
        """添加重力随机化范围。"""
        self.ranges["gravity"].append(RandomizationRange(
            param_name="gravity_z", nominal=-9.81,
            low=low, high=high
        ))
        return self

    def sample(self) -> Tuple[mujoco.MjModel, Dict[str, Any]]:
        """
        采样一组随机参数，返回新的模型和参数记录。

        返回:
          model: 参数化后的 MjModel
          params: 记录本次采样的所有参数值
        """
        model = mujoco.MjModel.from_xml_string(self.base_xml)
        params = {}

        # 随机化质量
        for r in self.ranges["mass"]:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, r.param_name)
            val = r.sample(self.rng)
            body_id = model.geom_bodyid[geom_id]
            model.body_mass[body_id] = val
            params[f"mass_{r.param_name}"] = val

        # 随机化摩擦
        for r in self.ranges["friction"]:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, r.param_name)
            val = r.sample(self.rng)
            model.geom_friction[geom_id, 0] = val
            params[f"friction_{r.param_name}"] = val

        # 随机化阻尼
        for r in self.ranges["damping"]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, r.param_name)
            val = r.sample(self.rng)
            model.dof_damping[joint_id] = val
            params[f"damping_{r.param_name}"] = val

        # 随机化步长
        for r in self.ranges["timestep"]:
            val = r.sample(self.rng)
            model.opt.timestep = val
            params["timestep"] = val

        # 随机化重力
        for r in self.ranges["gravity"]:
            val = r.sample(self.rng)
            model.opt.gravity[2] = val
            params["gravity_z"] = val

        return model, params


# ============================================================
# 第 3 节：数据采集与分布分析
# ============================================================

def collect_trajectory(model: mujoco.MjModel,
                       duration: float = 2.0,
                       ctrl_fn=None) -> Dict[str, np.ndarray]:
    """
    在给定模型上采集一条轨迹。

    参数:
      model: MuJoCo 模型（可能被随机化过）
      duration: 仿真时长
      ctrl_fn: 控制函数 (data, t) -> ctrl
    """
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    n_steps = int(duration / dt)

    qpos_list = []
    qvel_list = []
    ctrl_list = []
    ee_pos_list = []

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    for step in range(n_steps):
        t = step * dt

        if ctrl_fn is not None:
            data.ctrl[:] = ctrl_fn(data, t)
        else:
            # 默认: 正弦波控制
            data.ctrl[0] = 10 * np.sin(2 * np.pi * 0.5 * t)
            data.ctrl[1] = 5 * np.sin(2 * np.pi * 0.3 * t + 0.5)

        mujoco.mj_step(model, data)

        qpos_list.append(data.qpos.copy())
        qvel_list.append(data.qvel.copy())
        ctrl_list.append(data.ctrl.copy())
        ee_pos_list.append(data.site_xpos[site_id].copy())

    return {
        "qpos": np.array(qpos_list),
        "qvel": np.array(qvel_list),
        "ctrl": np.array(ctrl_list),
        "ee_pos": np.array(ee_pos_list),
    }


def demo_domain_randomization() -> Tuple[List[Dict], List[Dict[str, Any]]]:
    """
    域随机化演示：采集多组不同参数下的数据，对比分布。
    """
    print(f"\n{DIVIDER}")
    print("第 3 节：域随机化数据采集")
    print(DIVIDER)

    # 设置随机化范围
    randomizer = DomainRandomizer(BASE_ARM_XML, seed=42)
    randomizer \
        .add_mass_range("g1", low=0.5, high=2.0) \
        .add_mass_range("g2", low=0.4, high=1.6) \
        .add_friction_range("g1", low=0.3, high=2.0) \
        .add_friction_range("g2", low=0.3, high=2.0) \
        .add_damping_range("j1", low=0.1, high=2.0) \
        .add_damping_range("j2", low=0.1, high=2.0) \
        .add_gravity_range(low=-10.5, high=-9.0)

    # --- 基线: 标称参数 ---
    base_model = mujoco.MjModel.from_xml_string(BASE_ARM_XML)
    base_traj = collect_trajectory(base_model)

    # --- 域随机化: 采集多条轨迹 ---
    n_variants = 50
    all_trajs = []
    all_params = []

    print(f"\n  采集 {n_variants} 条随机化轨迹...")
    for i in range(n_variants):
        model_rand, params = randomizer.sample()
        traj = collect_trajectory(model_rand)
        all_trajs.append(traj)
        all_params.append(params)

        if (i + 1) % 10 == 0:
            print(f"    已完成: {i + 1}/{n_variants}")

    # --- 统计对比 ---
    print(f"\n  数据分布对比:")
    print(f"  {'指标':<25} {'基线':<20} {'随机化 (均值±标准差)'}")
    print(f"  {SUB_DIVIDER}")

    metrics = [
        ("qpos 均值", lambda t: t["qpos"].mean()),
        ("qpos 标准差", lambda t: t["qpos"].std()),
        ("qvel 最大值", lambda t: np.abs(t["qvel"]).max()),
        ("末端 X 范围", lambda t: np.ptp(t["ee_pos"][:, 0])),
        ("末端 Z 范围", lambda t: np.ptp(t["ee_pos"][:, 2])),
    ]

    for name, fn in metrics:
        base_val = fn(base_traj)
        rand_vals = [fn(t) for t in all_trajs]
        mean_val = np.mean(rand_vals)
        std_val = np.std(rand_vals)
        print(f"  {name:<25} {base_val:<20.4f} {mean_val:.4f} ± {std_val:.4f}")

    return all_trajs, all_params


# ============================================================
# 第 4 节：数据增强技术
# ============================================================

class RobotDataAugmentor:
    """
    机器人数据增强器：对已有轨迹数据进行增强。

    支持的增强方式:
      1. 添加高斯噪声（模拟传感器噪声）
      2. 时间拉伸（模拟不同执行速度）
      3. 关节偏移（模拟校准误差）
      4. 信号平滑（模拟低通滤波）
      5. 随机裁剪（数据多样性）
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def add_noise(self, data: np.ndarray,
                  std: float = 0.01) -> np.ndarray:
        """添加高斯噪声，模拟传感器不确定性。"""
        noise = self.rng.normal(0, std, size=data.shape)
        return data + noise

    def time_stretch(self, data: np.ndarray,
                     factor: float = 1.0) -> np.ndarray:
        """时间拉伸/压缩，factor > 1 变慢，< 1 变快。"""
        n = data.shape[0]
        new_n = int(n * factor)
        old_indices = np.linspace(0, n - 1, new_n)
        if data.ndim == 1:
            return np.interp(old_indices, np.arange(n), data)
        else:
            result = np.zeros((new_n, data.shape[1]))
            for col in range(data.shape[1]):
                result[:, col] = np.interp(old_indices, np.arange(n), data[:, col])
            return result

    def joint_offset(self, data: np.ndarray,
                     max_offset: float = 0.05) -> np.ndarray:
        """添加恒定偏移，模拟关节零点校准误差。"""
        offset = self.rng.uniform(-max_offset, max_offset, size=data.shape[1:])
        return data + offset

    def smooth(self, data: np.ndarray,
               window_size: int = 5) -> np.ndarray:
        """滑动平均平滑，模拟低通滤波效果。"""
        kernel = np.ones(window_size) / window_size
        if data.ndim == 1:
            return np.convolve(data, kernel, mode="same")
        else:
            result = np.zeros_like(data)
            for col in range(data.shape[1]):
                result[:, col] = np.convolve(data[:, col], kernel, mode="same")
            return result

    def random_crop(self, data: np.ndarray,
                    crop_ratio: float = 0.8) -> np.ndarray:
        """随机裁剪连续片段。"""
        n = data.shape[0]
        crop_len = int(n * crop_ratio)
        start = self.rng.integers(0, n - crop_len)
        return data[start:start + crop_len]

    def augment(self, qpos: np.ndarray,
                noise_std: float = 0.01,
                time_factor: Optional[float] = None,
                offset: bool = False,
                smooth_window: Optional[int] = None) -> np.ndarray:
        """组合增强。"""
        result = qpos.copy()

        if noise_std > 0:
            result = self.add_noise(result, noise_std)
        if time_factor is not None and time_factor != 1.0:
            result = self.time_stretch(result, time_factor)
        if offset:
            result = self.joint_offset(result)
        if smooth_window is not None:
            result = self.smooth(result, smooth_window)

        return result


def demo_data_augmentation() -> None:
    """数据增强演示。"""
    print(f"\n{DIVIDER}")
    print("第 4 节：数据增强技术")
    print(DIVIDER)

    # 生成基线轨迹
    base_model = mujoco.MjModel.from_xml_string(BASE_ARM_XML)
    base_traj = collect_trajectory(base_model, duration=2.0)
    base_qpos = base_traj["qpos"]

    augmentor = RobotDataAugmentor(seed=42)

    augmentations = {
        "噪声 (σ=0.02)": augmentor.add_noise(base_qpos, std=0.02),
        "时间拉伸 (1.5x)": augmentor.time_stretch(base_qpos, factor=1.5),
        "关节偏移": augmentor.joint_offset(base_qpos, max_offset=0.1),
        "平滑 (窗口=10)": augmentor.smooth(base_qpos, window_size=10),
    }

    print(f"\n  原始数据: shape = {base_qpos.shape}")
    for name, aug_data in augmentations.items():
        diff = np.abs(aug_data[:min(len(aug_data), len(base_qpos))]
                      - base_qpos[:min(len(aug_data), len(base_qpos))]).mean()
        print(f"  {name}: shape = {aug_data.shape}, 平均变化 = {diff:.4f}")


# ============================================================
# 第 5 节：绘制分布对比图
# ============================================================

def plot_distributions(all_trajs: List[Dict],
                       all_params: List[Dict[str, Any]],
                       save_path: str = "sim2real_distributions.png") -> None:
    """
    绘制域随机化对数据分布的影响。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 收集基线数据
    base_model = mujoco.MjModel.from_xml_string(BASE_ARM_XML)
    base_traj = collect_trajectory(base_model)

    # --- 子图 1: qpos 分布对比 ---
    ax = axes[0, 0]
    ax.hist(base_traj["qpos"][:, 0], bins=50, alpha=0.7,
            color="steelblue", label="Nominal", density=True, edgecolor="white")
    all_qpos = np.concatenate([t["qpos"][:, 0] for t in all_trajs])
    ax.hist(all_qpos, bins=50, alpha=0.5,
            color="coral", label="Randomized", density=True, edgecolor="white")
    ax.set_xlabel("Joint 1 position (rad)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Joint Position Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 子图 2: qvel 分布对比 ---
    ax = axes[0, 1]
    ax.hist(base_traj["qvel"][:, 0], bins=50, alpha=0.7,
            color="steelblue", label="Nominal", density=True, edgecolor="white")
    all_qvel = np.concatenate([t["qvel"][:, 0] for t in all_trajs])
    ax.hist(all_qvel, bins=50, alpha=0.5,
            color="coral", label="Randomized", density=True, edgecolor="white")
    ax.set_xlabel("Joint 1 velocity (rad/s)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Joint Velocity Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 子图 3: 末端执行器位置 XZ ---
    ax = axes[0, 2]
    ax.scatter(base_traj["ee_pos"][:, 0], base_traj["ee_pos"][:, 2],
               s=1, alpha=0.5, color="steelblue", label="Nominal")
    for traj in all_trajs[:10]:
        ax.scatter(traj["ee_pos"][:, 0], traj["ee_pos"][:, 2],
                   s=0.5, alpha=0.2, color="coral")
    ax.scatter([], [], s=10, color="coral", label="Randomized (10 trajs)")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Z (m)", fontsize=10)
    ax.set_title("End Effector Position (XZ)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- 子图 4: 质量参数分布 ---
    ax = axes[1, 0]
    masses = [p.get("mass_g1", 1.0) for p in all_params]
    ax.hist(masses, bins=20, color="mediumpurple", alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Nominal = 1.0")
    ax.set_xlabel("Link 1 mass (kg)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Randomized Mass Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 子图 5: 参数 vs 数据指标关联 ---
    ax = axes[1, 1]
    masses = [p.get("mass_g1", 1.0) for p in all_params]
    max_vels = [np.abs(t["qvel"]).max() for t in all_trajs]
    ax.scatter(masses, max_vels, alpha=0.6, color="teal", s=30, edgecolor="white")
    ax.set_xlabel("Link 1 mass (kg)", fontsize=10)
    ax.set_ylabel("Max joint velocity (rad/s)", fontsize=10)
    ax.set_title("Mass vs Peak Velocity", fontsize=12)
    ax.grid(True, alpha=0.3)

    # 拟合趋势线
    z = np.polyfit(masses, max_vels, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(masses), max(masses), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Linear fit")
    ax.legend(fontsize=9)

    # --- 子图 6: 轨迹间方差分布 ---
    ax = axes[1, 2]
    traj_stds = [t["qpos"][:, 0].std() for t in all_trajs]
    ax.hist(traj_stds, bins=20, color="goldenrod", alpha=0.7, edgecolor="white")
    base_std = base_traj["qpos"][:, 0].std()
    ax.axvline(base_std, color="red", linestyle="--", linewidth=2,
               label=f"Nominal std = {base_std:.4f}")
    ax.set_xlabel("qpos standard deviation", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Trajectory Variability", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Sim-to-Real: Domain Randomization Effects on Data Distribution\n"
                 "50 randomized trajectories vs nominal baseline",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ 分布对比图已保存: {save_path}")


# ============================================================
# 主程序
# ============================================================

def main():
    print(DIVIDER)
    print("第 8 章 · 03 - Sim-to-Real 概念")
    print("域随机化 · 数据分布 · 数据增强")
    print(DIVIDER)

    # 第 1 节: 概念解释
    explain_sim2real_gap()

    # 第 3 节: 域随机化实验
    all_trajs, all_params = demo_domain_randomization()

    # 第 4 节: 数据增强
    demo_data_augmentation()

    # 第 5 节: 绘图
    plot_distributions(all_trajs, all_params)

    # --- 总结 ---
    print(f"\n{DIVIDER}")
    print("关键收获:")
    print("  1. Sim-to-real gap 是仿真数据与真实数据的分布差异")
    print("  2. 域随机化通过扰动参数来扩展数据分布")
    print("  3. 不同参数对数据分布的影响程度不同（质量 > 摩擦 > 阻尼）")
    print("  4. 数据增强可以在已有数据上增加多样性")
    print("  5. 数据工程师需要记录每条轨迹对应的参数（参数溯源）")
    print(DIVIDER)


if __name__ == "__main__":
    main()
