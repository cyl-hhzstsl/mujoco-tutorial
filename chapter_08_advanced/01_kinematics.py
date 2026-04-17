"""
第 8 章 · 01 - 正运动学与逆运动学 (Forward & Inverse Kinematics)

目标: 理解关节空间与笛卡尔空间的映射关系，
     掌握 Jacobian 矩阵在 IK 和工作空间分析中的应用。

核心知识点:
  1. 正运动学 (FK): qpos → 末端执行器位置（使用 mj_forward）
  2. Jacobian 计算: mj_jacSite / mj_jacBody → 速度映射矩阵
  3. 逆运动学 (IK): 基于 Jacobian 伪逆的迭代求解器
  4. 工作空间分析: 遍历关节空间，绘制可达区域
  5. 奇异性与条件数: Jacobian 退化时的处理

运行: python 01_kinematics.py
输出: workspace.png (工作空间散点图)
依赖: pip install numpy matplotlib mujoco
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# ============================================================
# 3 连杆平面机械臂的 MJCF 模型
# ============================================================

THREE_LINK_ARM_XML = """
<mujoco model="three_link_arm">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <default>
    <joint type="hinge" axis="0 1 0" limited="true" damping="0.5"/>
    <geom type="capsule" size="0.02" rgba="0.4 0.6 0.8 1"/>
  </default>

  <worldbody>
    <!-- 基座（固定） -->
    <body name="base" pos="0 0 0">
      <geom type="sphere" size="0.04" rgba="0.3 0.3 0.3 1"/>

      <!-- 第一连杆: 肩关节 -->
      <body name="link1" pos="0 0 0">
        <joint name="shoulder" range="-180 180"/>
        <geom fromto="0 0 0  0.3 0 0" size="0.02"/>

        <!-- 第二连杆: 肘关节 -->
        <body name="link2" pos="0.3 0 0">
          <joint name="elbow" range="-150 150"/>
          <geom fromto="0 0 0  0.25 0 0" size="0.018"/>

          <!-- 第三连杆: 腕关节 -->
          <body name="link3" pos="0.25 0 0">
            <joint name="wrist" range="-120 120"/>
            <geom fromto="0 0 0  0.2 0 0" size="0.015"/>

            <!-- 末端执行器（用 site 标记位置） -->
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
# 第 1 节：正运动学 (Forward Kinematics)
# ============================================================

def demo_forward_kinematics(model: mujoco.MjModel,
                            data: mujoco.MjData) -> None:
    """
    正运动学演示：给定关节角度 → 计算末端执行器位置。

    原理:
      1. 设置 data.qpos（关节角度）
      2. 调用 mj_forward() 计算所有物理量
      3. 读取 data.site_xpos 获取末端执行器位置
    """
    print(f"\n{DIVIDER}")
    print("第 1 节：正运动学 (Forward Kinematics)")
    print(DIVIDER)

    # 获取 site 的索引
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    # --- 测试多组关节角度 ---
    test_configs = [
        ("零位", [0.0, 0.0, 0.0]),
        ("肩关节 90°", [np.pi / 2, 0.0, 0.0]),
        ("全部 45°", [np.pi / 4, np.pi / 4, np.pi / 4]),
        ("肘关节弯曲 -90°", [0.0, -np.pi / 2, 0.0]),
        ("复合姿态", [np.pi / 3, -np.pi / 4, np.pi / 6]),
    ]

    print(f"\n{'配置':<20} {'qpos (度)':<30} {'末端位置 (x, y, z)':<30}")
    print(SUB_DIVIDER)

    results = []
    for name, qpos_rad in test_configs:
        # 设置关节角度
        data.qpos[:] = qpos_rad
        # 执行正运动学计算（核心调用）
        mujoco.mj_forward(model, data)

        # 读取末端位置
        ee_pos = data.site_xpos[site_id].copy()
        qpos_deg = np.degrees(qpos_rad)

        print(f"{name:<20} [{qpos_deg[0]:>6.1f}, {qpos_deg[1]:>6.1f}, {qpos_deg[2]:>6.1f}]"
              f"      [{ee_pos[0]:>7.4f}, {ee_pos[1]:>7.4f}, {ee_pos[2]:>7.4f}]")

        results.append((name, qpos_rad, ee_pos))

    # 验证: 零位时，末端应该在 x = 0.3 + 0.25 + 0.2 = 0.75
    data.qpos[:] = [0, 0, 0]
    mujoco.mj_forward(model, data)
    ee_zero = data.site_xpos[site_id].copy()
    expected_x = 0.3 + 0.25 + 0.2  # 三个连杆长度之和
    print(f"\n✓ 零位验证: x = {ee_zero[0]:.4f}, 期望 = {expected_x:.4f}, "
          f"误差 = {abs(ee_zero[0] - expected_x):.6f}")

    return results


# ============================================================
# 第 2 节：Jacobian 矩阵计算
# ============================================================

def compute_jacobian(model: mujoco.MjModel,
                     data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算末端执行器相对于关节角度的 Jacobian 矩阵。

    Jacobian 矩阵 J 的物理含义:
      ẋ = J · q̇
      末端速度 = Jacobian × 关节速度

    返回:
      jacp: 位置 Jacobian (3 × nv)
      jacr: 旋转 Jacobian (3 × nv)
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    nv = model.nv  # 自由度数量

    jacp = np.zeros((3, nv))  # 位置 Jacobian
    jacr = np.zeros((3, nv))  # 旋转 Jacobian

    # MuJoCo 的 Jacobian 计算（核心 API）
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    return jacp, jacr


def demo_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    Jacobian 矩阵演示：理解速度映射和奇异性。
    """
    print(f"\n{DIVIDER}")
    print("第 2 节：Jacobian 矩阵")
    print(DIVIDER)

    # --- 在几个不同的配置下计算 Jacobian ---
    configs = [
        ("零位（完全伸展）", [0.0, 0.0, 0.0]),
        ("45° 弯曲", [np.pi / 4, np.pi / 4, np.pi / 4]),
        ("接近奇异（完全折叠）", [0.0, np.pi, 0.0]),
    ]

    for name, qpos in configs:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        jacp, jacr = compute_jacobian(model, data)

        # 条件数：衡量 Jacobian 的「健康程度」，越大越接近奇异
        cond = np.linalg.cond(jacp)

        print(f"\n配置: {name}")
        print(f"  qpos (度): {np.degrees(qpos)}")
        print(f"  位置 Jacobian (3×{model.nv}):")
        for i, axis in enumerate(["dx", "dy", "dz"]):
            row_str = "  ".join(f"{v:>8.4f}" for v in jacp[i])
            print(f"    {axis}: [{row_str}]")
        print(f"  条件数: {cond:.2f}" +
              (" ⚠️ 接近奇异!" if cond > 100 else " ✓ 状态良好"))

    # --- 数值验证: Jacobian 近似于有限差分 ---
    print(f"\n{SUB_DIVIDER}")
    print("Jacobian 数值验证（与有限差分对比）")

    data.qpos[:] = [0.5, -0.3, 0.2]
    mujoco.mj_forward(model, data)
    jacp_analytic, _ = compute_jacobian(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    eps = 1e-6
    jacp_numerical = np.zeros((3, model.nv))

    for j in range(model.nv):
        # 前向扰动
        data.qpos[j] += eps
        mujoco.mj_forward(model, data)
        pos_plus = data.site_xpos[site_id].copy()

        # 后向扰动
        data.qpos[j] -= 2 * eps
        mujoco.mj_forward(model, data)
        pos_minus = data.site_xpos[site_id].copy()

        # 恢复
        data.qpos[j] += eps
        jacp_numerical[:, j] = (pos_plus - pos_minus) / (2 * eps)

    mujoco.mj_forward(model, data)

    error = np.abs(jacp_analytic - jacp_numerical).max()
    print(f"  解析 Jacobian vs 有限差分: 最大误差 = {error:.2e}")
    print(f"  {'✓ 验证通过' if error < 1e-4 else '✗ 误差过大'}")


# ============================================================
# 第 3 节：逆运动学 (Inverse Kinematics)
# ============================================================

class JacobianIKSolver:
    """
    基于 Jacobian 伪逆的迭代 IK 求解器。

    算法:
      1. 计算当前末端位置与目标的误差 Δx
      2. 通过 Jacobian 伪逆计算关节增量: Δq = J⁺ · Δx
      3. 更新关节角度: q = q + α · Δq
      4. 重复直到收敛或达到最大迭代次数

    阻尼最小二乘法 (Damped Least Squares, DLS):
      Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ Δx
      当 Jacobian 接近奇异时，λ 提供数值稳定性
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 site_name: str = "end_effector",
                 max_iter: int = 200,
                 tol: float = 1e-4,
                 step_size: float = 0.5,
                 damping: float = 1e-4):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.damping = damping

    def solve(self, target_pos: np.ndarray,
              q_init: Optional[np.ndarray] = None) -> Dict:
        """
        求解 IK：找到使末端到达 target_pos 的关节角度。

        返回:
          dict: qpos, final_pos, error, iterations, success, trajectory
        """
        nv = self.model.nv

        if q_init is not None:
            self.data.qpos[:] = q_init
        else:
            self.data.qpos[:] = 0

        trajectory = []  # 记录求解轨迹

        for i in range(self.max_iter):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[self.site_id].copy()
            error_vec = target_pos - current_pos
            error_norm = np.linalg.norm(error_vec)

            trajectory.append({
                "iter": i,
                "qpos": self.data.qpos.copy(),
                "pos": current_pos.copy(),
                "error": error_norm,
            })

            if error_norm < self.tol:
                return {
                    "qpos": self.data.qpos.copy(),
                    "final_pos": current_pos,
                    "error": error_norm,
                    "iterations": i + 1,
                    "success": True,
                    "trajectory": trajectory,
                }

            # 计算 Jacobian
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

            # 阻尼最小二乘 (Damped Least Squares)
            # Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ Δx
            JJT = jacp @ jacp.T + self.damping ** 2 * np.eye(3)
            dq = jacp.T @ np.linalg.solve(JJT, error_vec)

            # 更新关节角度
            self.data.qpos[:] += self.step_size * dq

            # 限制关节角度在范围内
            for j in range(self.model.njnt):
                if self.model.jnt_limited[j]:
                    idx = self.model.jnt_qposadr[j]
                    low, high = self.model.jnt_range[j]
                    self.data.qpos[idx] = np.clip(self.data.qpos[idx], low, high)

        mujoco.mj_forward(self.model, self.data)
        final_pos = self.data.site_xpos[self.site_id].copy()
        return {
            "qpos": self.data.qpos.copy(),
            "final_pos": final_pos,
            "error": np.linalg.norm(target_pos - final_pos),
            "iterations": self.max_iter,
            "success": False,
            "trajectory": trajectory,
        }


def demo_inverse_kinematics(model: mujoco.MjModel,
                            data: mujoco.MjData) -> None:
    """逆运动学演示：给定目标位置 → 求解关节角度。"""
    print(f"\n{DIVIDER}")
    print("第 3 节：逆运动学 (Inverse Kinematics)")
    print(DIVIDER)

    solver = JacobianIKSolver(model, data)

    # 测试多个目标点
    targets = [
        ("正前方", np.array([0.5, 0.0, 0.0])),
        ("右上方", np.array([0.3, 0.0, 0.3])),
        ("近处", np.array([0.15, 0.0, 0.1])),
        ("不可达（太远）", np.array([1.0, 0.0, 0.0])),
    ]

    print(f"\n{'目标':<20} {'目标位置':<25} {'实际位置':<25} "
          f"{'误差':<12} {'迭代':<8} {'状态'}")
    print(SUB_DIVIDER)

    for name, target in targets:
        result = solver.solve(target, q_init=np.array([0.3, -0.3, 0.1]))

        t_str = f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]"
        p = result["final_pos"]
        p_str = f"[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
        status = "✓ 成功" if result["success"] else "✗ 未收敛"

        print(f"{name:<20} {t_str:<25} {p_str:<25} "
              f"{result['error']:<12.6f} {result['iterations']:<8} {status}")

        if result["success"]:
            q_deg = np.degrees(result["qpos"])
            print(f"  → 关节角度: [{q_deg[0]:.1f}°, {q_deg[1]:.1f}°, {q_deg[2]:.1f}°]")


# ============================================================
# 第 4 节：工作空间分析
# ============================================================

def analyze_workspace(model: mujoco.MjModel,
                      data: mujoco.MjData,
                      resolution: int = 20) -> Dict[str, np.ndarray]:
    """
    工作空间分析：遍历关节空间，映射到笛卡尔空间。

    方法: 在每个关节的范围内均匀采样，组合所有可能的关节角度，
         对每个组合执行 FK，记录末端位置。

    参数:
      resolution: 每个关节的采样密度（总点数 = resolution³）
    """
    print(f"\n{DIVIDER}")
    print("第 4 节：工作空间分析")
    print(DIVIDER)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    # 获取关节范围
    joint_ranges = []
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            low, high = model.jnt_range[j]
        else:
            low, high = -np.pi, np.pi
        joint_ranges.append((low, high))

    # 生成网格点
    grids = [np.linspace(low, high, resolution) for low, high in joint_ranges]
    total_points = resolution ** model.njnt
    print(f"  每轴采样: {resolution} 点")
    print(f"  总采样量: {total_points} 点")

    positions = []
    joint_configs = []
    jacobian_conds = []

    for i, q0 in enumerate(grids[0]):
        for q1 in grids[1]:
            for q2 in grids[2]:
                qpos = np.array([q0, q1, q2])
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)

                pos = data.site_xpos[site_id].copy()
                positions.append(pos)
                joint_configs.append(qpos.copy())

                # 计算 Jacobian 条件数（可选，用于标记奇异区域）
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                cond = np.linalg.cond(jacp)
                jacobian_conds.append(cond)

    positions = np.array(positions)
    jacobian_conds = np.array(jacobian_conds)

    # 统计
    x_range = positions[:, 0].max() - positions[:, 0].min()
    z_range = positions[:, 2].max() - positions[:, 2].min()
    reach_max = np.linalg.norm(positions, axis=1).max()

    print(f"\n  工作空间统计:")
    print(f"    X 范围: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] "
          f"(跨度 {x_range:.3f} m)")
    print(f"    Z 范围: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] "
          f"(跨度 {z_range:.3f} m)")
    print(f"    最大可达距离: {reach_max:.3f} m")
    print(f"    奇异点比例: {(jacobian_conds > 100).sum() / len(jacobian_conds) * 100:.1f}%")

    return {
        "positions": positions,
        "joint_configs": np.array(joint_configs),
        "jacobian_conds": jacobian_conds,
    }


def plot_workspace(workspace_data: Dict[str, np.ndarray],
                   save_path: str = "workspace.png") -> None:
    """
    绘制工作空间散点图（XZ 平面 + 条件数着色）。
    """
    positions = workspace_data["positions"]
    conds = workspace_data["jacobian_conds"]

    # 对条件数取 log 以便可视化
    log_conds = np.log10(np.clip(conds, 1, 1e6))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- 子图 1: XZ 平面工作空间 ---
    ax = axes[0]
    scatter = ax.scatter(positions[:, 0], positions[:, 2],
                         c=log_conds, cmap="RdYlGn_r",
                         s=2, alpha=0.6)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Z (m)", fontsize=11)
    ax.set_title("Workspace (XZ plane)\nColor = log₁₀(condition number)", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="log₁₀(cond)")

    # 标记原点
    ax.plot(0, 0, "k^", markersize=10, label="Base")
    ax.legend(fontsize=9)

    # --- 子图 2: 可达距离分布 ---
    ax = axes[1]
    distances = np.linalg.norm(positions, axis=1)
    ax.hist(distances, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Distance from base (m)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Reachable Distance Distribution", fontsize=12)
    ax.axvline(distances.mean(), color="red", linestyle="--",
               label=f"Mean = {distances.mean():.3f} m")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 子图 3: 条件数分布 ---
    ax = axes[2]
    ax.hist(log_conds, bins=50, color="coral", alpha=0.7, edgecolor="white")
    ax.set_xlabel("log₁₀(Condition Number)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Jacobian Condition Number Distribution", fontsize=12)
    ax.axvline(np.log10(100), color="red", linestyle="--",
               label="Singularity threshold (100)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ 工作空间图已保存: {save_path}")


# ============================================================
# 主程序
# ============================================================

def main():
    print(DIVIDER)
    print("第 8 章 · 01 - 正运动学与逆运动学")
    print("3 连杆平面机械臂 FK/IK/工作空间分析")
    print(DIVIDER)

    # --- 加载模型 ---
    model = mujoco.MjModel.from_xml_string(THREE_LINK_ARM_XML)
    data = mujoco.MjData(model)

    print(f"\n模型概要:")
    print(f"  关节数: {model.njnt}")
    print(f"  自由度: {model.nv}")
    print(f"  执行器: {model.nu}")
    print(f"  连杆长度: 0.3 + 0.25 + 0.2 = 0.75 m")

    # 第 1 节: 正运动学
    demo_forward_kinematics(model, data)

    # 第 2 节: Jacobian
    demo_jacobian(model, data)

    # 第 3 节: 逆运动学
    demo_inverse_kinematics(model, data)

    # 第 4 节: 工作空间分析
    workspace = analyze_workspace(model, data, resolution=25)
    plot_workspace(workspace)

    print(f"\n{DIVIDER}")
    print("完成! 关键收获:")
    print("  1. FK 是确定性映射: qpos → 末端位置")
    print("  2. IK 可能有多个解、无解、或接近奇异")
    print("  3. Jacobian 条件数反映了机械臂在该配置下的操作性能")
    print("  4. 工作空间分析帮助数据工程师判断轨迹数据是否合理")
    print(DIVIDER)


if __name__ == "__main__":
    main()
