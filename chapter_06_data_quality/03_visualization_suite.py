"""
第 6 章 · 03 - 数据质量可视化工具箱 (Visualization Suite)

目标: 提供一套完整的可视化工具，帮助快速识别数据质量问题。

核心知识点:
  1. 关节角度时间序列 + 限位标注带
  2. 异常点高亮标注
  3. 关节角度分布直方图 + 期望范围
  4. 关节相关性热图
  5. 帧间速度热图（发现不连续性）
  6. 跨 episode 箱线图
  7. 基座 3D 轨迹图
  8. 质量评分仪表盘（多子图汇总）

运行: python 03_visualization_suite.py
依赖: pip install numpy matplotlib
输出: quality_report/ 目录下的多张图
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("错误: matplotlib 未安装。请执行 pip install matplotlib")


# ============================================================
# 可视化工具类
# ============================================================

class QualityVisualizer:
    """
    数据质量可视化工具箱。

    提供多种可视化方法，每种聚焦于数据质量的不同方面。
    所有图表保存到指定的输出目录。

    使用方法:
        viz = QualityVisualizer(output_dir="quality_report")
        viz.plot_joint_timeseries(qpos, joint_limits=...)
        viz.plot_anomaly_overlay(qpos, anomaly_frames=...)
        viz.plot_all(episodes_data)  # 一键生成所有图表
    """

    def __init__(self, output_dir: str = "quality_report",
                 joint_names: Optional[List[str]] = None,
                 hz: float = 50.0,
                 figsize_default: Tuple[int, int] = (14, 8)):
        self.output_dir = output_dir
        self.joint_names = joint_names
        self.hz = hz
        self.figsize = figsize_default

        os.makedirs(output_dir, exist_ok=True)

    def _joint_label(self, idx: int) -> str:
        if self.joint_names and idx < len(self.joint_names):
            return self.joint_names[idx]
        return f"Joint {idx}"

    def _save(self, fig, filename: str):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    已保存: {path}")

    # ----------------------------------------------------------
    # 1. 关节角度时间序列 + 限位带
    # ----------------------------------------------------------
    def plot_joint_timeseries(self, qpos: np.ndarray,
                              joint_limits_lower: Optional[np.ndarray] = None,
                              joint_limits_upper: Optional[np.ndarray] = None,
                              title: str = "Joint Angle Time Series",
                              filename: str = "01_joint_timeseries.png"):
        """
        绘制每个关节的时间序列曲线，并用半透明色带标注关节限位。

        限位之外的区域用红色填充，一眼可见违规帧。
        """
        if not HAS_MPL:
            return

        T, nq = qpos.shape
        time_axis = np.arange(T) / self.hz

        # 最多显示 7 个关节，每行 1 个
        nq_display = min(nq, 7)
        fig, axes = plt.subplots(nq_display, 1, figsize=(self.figsize[0], 2 * nq_display),
                                 sharex=True)
        if nq_display == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=13, fontweight="bold")

        for j in range(nq_display):
            ax = axes[j]
            ax.plot(time_axis, qpos[:, j], color="steelblue", linewidth=0.8, label=self._joint_label(j))

            # 绘制限位带
            if joint_limits_lower is not None and joint_limits_upper is not None:
                lo = joint_limits_lower[j] if j < len(joint_limits_lower) else None
                hi = joint_limits_upper[j] if j < len(joint_limits_upper) else None
                if lo is not None and hi is not None:
                    ax.axhspan(lo, hi, color="green", alpha=0.08, label="Allowed range")
                    ax.axhline(lo, color="red", linewidth=0.6, linestyle="--", alpha=0.5)
                    ax.axhline(hi, color="red", linewidth=0.6, linestyle="--", alpha=0.5)

                    # 标红违规区域
                    below = qpos[:, j] < lo
                    above = qpos[:, j] > hi
                    if below.any():
                        ax.fill_between(time_axis, qpos[:, j], lo,
                                        where=below, color="red", alpha=0.3)
                    if above.any():
                        ax.fill_between(time_axis, qpos[:, j], hi,
                                        where=above, color="red", alpha=0.3)

            ax.set_ylabel(self._joint_label(j), fontsize=9)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=8)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------
    # 2. 异常高亮叠加
    # ----------------------------------------------------------
    def plot_anomaly_overlay(self, qpos: np.ndarray,
                             anomaly_frames: Dict[int, List[int]],
                             title: str = "Anomaly Overlay",
                             filename: str = "02_anomaly_overlay.png"):
        """
        在时间序列上高亮标注异常帧。

        anomaly_frames: {joint_idx: [frame_idx, ...]}
        """
        if not HAS_MPL:
            return

        T, nq = qpos.shape
        time_axis = np.arange(T) / self.hz
        nq_display = min(nq, 7)

        fig, axes = plt.subplots(nq_display, 1, figsize=(self.figsize[0], 2 * nq_display),
                                 sharex=True)
        if nq_display == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for j in range(nq_display):
            ax = axes[j]
            ax.plot(time_axis, qpos[:, j], color="steelblue", linewidth=0.8)

            # 高亮异常帧
            frames = anomaly_frames.get(j, [])
            if frames:
                frames = np.array(frames)
                frames = frames[frames < T]
                ax.scatter(time_axis[frames], qpos[frames, j],
                           color="red", s=15, zorder=5, label=f"{len(frames)} anomalies")
                ax.legend(fontsize=8, loc="upper right")

            ax.set_ylabel(self._joint_label(j), fontsize=9)
            ax.grid(alpha=0.2)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------
    # 3. 关节角度分布直方图
    # ----------------------------------------------------------
    def plot_distribution_histograms(self, qpos: np.ndarray,
                                     expected_lower: Optional[np.ndarray] = None,
                                     expected_upper: Optional[np.ndarray] = None,
                                     filename: str = "03_distribution_histograms.png"):
        """
        每个关节的值分布直方图。

        用竖线标出期望范围边界，超出范围的部分用红色。
        """
        if not HAS_MPL:
            return

        nq = qpos.shape[1]
        ncols = min(4, nq)
        nrows = (nq + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        fig.suptitle("Joint Value Distribution", fontsize=13, fontweight="bold")
        axes_flat = axes.flatten() if nq > 1 else [axes]

        for j in range(nq):
            ax = axes_flat[j]
            vals = qpos[:, j]

            # 分离范围内外的数据
            if expected_lower is not None and expected_upper is not None:
                lo, hi = expected_lower[j], expected_upper[j]
                in_range = vals[(vals >= lo) & (vals <= hi)]
                out_range = vals[(vals < lo) | (vals > hi)]

                bins = np.linspace(vals.min() - 0.1, vals.max() + 0.1, 40)
                ax.hist(in_range, bins=bins, color="steelblue", alpha=0.7, label="In range")
                if len(out_range) > 0:
                    ax.hist(out_range, bins=bins, color="red", alpha=0.6, label="Out of range")
                ax.axvline(lo, color="darkred", linestyle="--", linewidth=1)
                ax.axvline(hi, color="darkred", linestyle="--", linewidth=1)
                ax.legend(fontsize=7)
            else:
                ax.hist(vals, bins=40, color="steelblue", alpha=0.7)

            ax.set_title(self._joint_label(j), fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.2)

        # 隐藏多余的子图
        for idx in range(nq, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------
    # 4. 关节相关性热图
    # ----------------------------------------------------------
    def plot_correlation_heatmap(self, qpos: np.ndarray,
                                 filename: str = "04_correlation_heatmap.png"):
        """
        关节间 Pearson 相关系数的热图。

        强正相关（红色）= 同步运动
        强负相关（蓝色）= 反向运动
        零相关（白色）= 独立运动
        """
        if not HAS_MPL:
            return

        nq = qpos.shape[1]
        corr = np.corrcoef(qpos.T)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

        labels = [self._joint_label(j) for j in range(nq)]
        ax.set_xticks(range(nq))
        ax.set_yticks(range(nq))
        ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=9)

        # 在每个格子中标注数值
        for i in range(nq):
            for j in range(nq):
                color = "white" if abs(corr[i, j]) > 0.6 else "black"
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
        ax.set_title("Joint Correlation Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------
    # 5. 帧间速度热图
    # ----------------------------------------------------------
    def plot_velocity_heatmap(self, qpos: np.ndarray,
                              filename: str = "05_velocity_heatmap.png"):
        """
        帧间速度 (差分) 的热图。

        横轴 = 时间步，纵轴 = 关节
        亮色 = 大速度变化，暗色 = 平稳
        突然的亮条 = 不连续性（跳变、丢帧）
        """
        if not HAS_MPL:
            return

        velocity = np.abs(np.diff(qpos, axis=0)) * self.hz  # 转换为每秒
        T, nq = velocity.shape

        fig, ax = plt.subplots(figsize=(self.figsize[0], 5))

        # 对数缩放以增强可视性
        velocity_log = np.log1p(velocity)

        im = ax.imshow(velocity_log.T, aspect="auto", cmap="hot",
                        extent=[0, T / self.hz, nq - 0.5, -0.5])

        labels = [self._joint_label(j) for j in range(nq)]
        ax.set_yticks(range(nq))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_title("Frame-to-Frame Velocity Heatmap (log scale)", fontsize=13, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8, label="log(1 + |velocity|)")
        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------
    # 6. 跨 episode 箱线图
    # ----------------------------------------------------------
    def plot_episode_boxplots(self, episodes: List[Tuple[str, np.ndarray]],
                              filename: str = "06_episode_boxplots.png"):
        """
        对每个关节，绘制不同 episode 的值域箱线图。

        可以发现:
        - 某些 episode 的分布异常偏离
        - 不同 episode 之间的一致性
        """
        if not HAS_MPL:
            return
        if not episodes:
            return

        nq = episodes[0][1].shape[1]
        nq_display = min(nq, 7)

        fig, axes = plt.subplots(1, nq_display, figsize=(3 * nq_display, 6))
        if nq_display == 1:
            axes = [axes]
        fig.suptitle("Cross-Episode Box Plots", fontsize=13, fontweight="bold")

        for j in range(nq_display):
            ax = axes[j]
            box_data = [ep_qpos[:, j] for _, ep_qpos in episodes]

            bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                            showfliers=True, flierprops=dict(markersize=2, alpha=0.5))

            colors = plt.cm.Set3(np.linspace(0, 1, len(episodes)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(self._joint_label(j), fontsize=10)
            ax.set_xlabel("Episode", fontsize=8)
            if len(episodes) > 10:
                ax.set_xticks([1, len(episodes) // 2, len(episodes)])
            ax.grid(axis="y", alpha=0.2)

        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------
    # 7. 3D 轨迹图
    # ----------------------------------------------------------
    def plot_3d_trajectory(self, positions: np.ndarray,
                           title: str = "3D Base Position Trajectory",
                           filename: str = "07_3d_trajectory.png"):
        """
        绘制基座（或末端执行器）的 3D 位置轨迹。

        positions: (T, 3) 数组
        颜色渐变表示时间进展。
        """
        if not HAS_MPL:
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        T = positions.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, T))

        # 绘制散点（带颜色渐变）
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=colors, s=3, alpha=0.6)

        # 绘制线段
        for i in range(T - 1):
            ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                    color=colors[i], linewidth=0.5, alpha=0.5)

        # 标注起点和终点
        ax.scatter(*positions[0], color="green", s=100, marker="^", label="Start", zorder=10)
        ax.scatter(*positions[-1], color="red", s=100, marker="v", label="End", zorder=10)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend()

        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------
    # 8. 质量评分仪表盘
    # ----------------------------------------------------------
    def plot_quality_dashboard(self, quality_scores: Dict[str, float],
                               episode_scores: Optional[List[float]] = None,
                               joint_health: Optional[List[float]] = None,
                               filename: str = "08_quality_dashboard.png"):
        """
        综合质量仪表盘，多子图汇总:
        - 各维度质量分数的雷达图
        - Episode 质量分数分布
        - 关节健康度条形图
        - 总体评分仪表盘

        quality_scores: {"completeness": 0.95, "consistency": 0.88, ...}
        """
        if not HAS_MPL:
            return

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle("Data Quality Dashboard", fontsize=15, fontweight="bold")

        # --- 子图 1: 总体评分 (大字显示) ---
        ax = fig.add_subplot(gs[0, 0])
        overall = np.mean(list(quality_scores.values())) * 100
        color = "green" if overall >= 80 else ("orange" if overall >= 60 else "red")
        ax.text(0.5, 0.5, f"{overall:.0f}", transform=ax.transAxes,
                fontsize=64, fontweight="bold", color=color,
                ha="center", va="center")
        ax.text(0.5, 0.15, "Overall Score", transform=ax.transAxes,
                fontsize=14, ha="center", va="center", color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # --- 子图 2: 各维度条形图 ---
        ax = fig.add_subplot(gs[0, 1])
        dims = list(quality_scores.keys())
        vals = [quality_scores[d] * 100 for d in dims]
        colors_bar = ["green" if v >= 80 else ("orange" if v >= 60 else "red") for v in vals]
        y_pos = np.arange(len(dims))
        bars = ax.barh(y_pos, vals, color=colors_bar, alpha=0.8, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(dims, fontsize=9)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score")
        ax.set_title("Quality Dimensions", fontsize=11)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.2)

        # --- 子图 3: episode 评分分布 ---
        ax = fig.add_subplot(gs[0, 2])
        if episode_scores:
            ax.hist(episode_scores, bins=min(20, len(episode_scores)),
                    color="steelblue", alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(episode_scores), color="red", linestyle="--",
                       label=f"Mean={np.mean(episode_scores):.1f}")
            ax.legend(fontsize=9)
        ax.set_title("Episode Score Distribution", fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)

        # --- 子图 4: 关节健康度 ---
        ax = fig.add_subplot(gs[1, 0])
        if joint_health:
            nj = len(joint_health)
            labels = [self._joint_label(j) for j in range(nj)]
            colors_jh = ["green" if h >= 0.8 else ("orange" if h >= 0.5 else "red") for h in joint_health]
            ax.bar(range(nj), [h * 100 for h in joint_health], color=colors_jh, alpha=0.8)
            ax.set_xticks(range(nj))
            ax.set_xticklabels(labels, fontsize=8, rotation=45)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Health %")
        ax.set_title("Joint Health", fontsize=11)
        ax.grid(axis="y", alpha=0.2)

        # --- 子图 5: 检查项通过/失败矩阵 ---
        ax = fig.add_subplot(gs[1, 1:])
        check_names = ["NaN/Inf", "Quaternion", "Joint Limits", "Frame Jumps",
                        "Traj Length", "Dead Joints", "Timestamps", "Action Range"]
        n_eps = 10
        np.random.seed(42)
        pass_matrix = np.random.choice([0, 1], size=(n_eps, len(check_names)), p=[0.15, 0.85])

        cmap = mcolors.ListedColormap(["#ff6b6b", "#51cf66"])
        ax.imshow(pass_matrix, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(check_names)))
        ax.set_xticklabels(check_names, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(n_eps))
        ax.set_yticklabels([f"Ep {i}" for i in range(n_eps)], fontsize=8)
        ax.set_title("Validation Pass/Fail Matrix", fontsize=11)

        for i in range(n_eps):
            for j in range(len(check_names)):
                ax.text(j, i, "✓" if pass_matrix[i, j] else "✗",
                        ha="center", va="center", fontsize=9,
                        color="white" if pass_matrix[i, j] == 0 else "darkgreen")

        plt.tight_layout()
        self._save(fig, filename)

    # ----------------------------------------------------------
    # 一键生成所有图表
    # ----------------------------------------------------------
    def plot_all(self, episodes: List[Tuple[str, dict]],
                 joint_limits_lower: Optional[np.ndarray] = None,
                 joint_limits_upper: Optional[np.ndarray] = None):
        """
        一键生成所有质量报告图表。

        参数:
            episodes: [(ep_id, data_dict), ...]
            joint_limits_lower/upper: 关节限位
        """
        if not HAS_MPL:
            print("  matplotlib 未安装，无法生成图表")
            return
        if not episodes:
            print("  无数据")
            return

        print(f"\n  开始生成质量报告图表...")
        print(f"  输出目录: {self.output_dir}")

        # 取第一个 episode 做时间序列图
        first_qpos = episodes[0][1]["observations"]["qpos"]

        # 1. 时间序列
        print(f"\n  [1/8] 关节角度时间序列...")
        self.plot_joint_timeseries(first_qpos, joint_limits_lower, joint_limits_upper)

        # 2. 异常高亮
        print(f"  [2/8] 异常高亮叠加...")
        anomalies = self._detect_simple_anomalies(first_qpos)
        self.plot_anomaly_overlay(first_qpos, anomalies)

        # 3. 分布直方图
        print(f"  [3/8] 分布直方图...")
        all_qpos = np.concatenate([ep["observations"]["qpos"] for _, ep in episodes], axis=0)
        self.plot_distribution_histograms(all_qpos, joint_limits_lower, joint_limits_upper)

        # 4. 相关性热图
        print(f"  [4/8] 相关性热图...")
        self.plot_correlation_heatmap(all_qpos)

        # 5. 速度热图
        print(f"  [5/8] 帧间速度热图...")
        self.plot_velocity_heatmap(first_qpos)

        # 6. 跨 episode 箱线图
        print(f"  [6/8] 跨 episode 箱线图...")
        ep_qpos_list = [(eid, ep["observations"]["qpos"]) for eid, ep in episodes[:20]]
        self.plot_episode_boxplots(ep_qpos_list)

        # 7. 3D 轨迹
        print(f"  [7/8] 3D 轨迹...")
        # 用前 3 个关节模拟 3D 位置
        positions_3d = first_qpos[:, :3]
        self.plot_3d_trajectory(positions_3d)

        # 8. 质量仪表盘
        print(f"  [8/8] 质量仪表盘...")
        quality_scores = {
            "Completeness": 0.95,
            "Consistency": 0.88,
            "Range Valid": 0.92,
            "Smoothness": 0.85,
            "No Anomaly": 0.78,
        }
        ep_scores = [np.random.uniform(60, 100) for _ in range(len(episodes))]
        joint_health = [np.random.uniform(0.6, 1.0) for _ in range(first_qpos.shape[1])]
        self.plot_quality_dashboard(quality_scores, ep_scores, joint_health)

        print(f"\n  ✅ 所有图表已生成到 {self.output_dir}/")

    def _detect_simple_anomalies(self, qpos: np.ndarray,
                                  z_threshold: float = 3.0) -> Dict[int, List[int]]:
        """简易 z-score 异常检测（仅用于可视化演示）"""
        T, nq = qpos.shape
        anomalies = {}
        for j in range(nq):
            col = qpos[:, j]
            mu, sigma = col.mean(), col.std()
            if sigma < 1e-10:
                continue
            z = np.abs((col - mu) / sigma)
            bad = np.where(z > z_threshold)[0]
            if len(bad) > 0:
                anomalies[j] = bad.tolist()
        return anomalies


# ============================================================
# 测试数据生成
# ============================================================

def generate_demo_episodes(num_episodes=15, nq=7, nu=7, hz=50):
    """生成用于可视化演示的数据集"""
    episodes = []
    for i in range(num_episodes):
        num_frames = np.random.randint(200, 500)
        t = np.linspace(0, 4 * np.pi, num_frames)

        freq_scale = 1.0 + np.random.randn() * 0.15
        amp_scale = 0.3 + np.random.rand() * 0.2
        noise = 0.01

        qpos = np.column_stack([
            np.sin(t * (j + 1) * freq_scale) * amp_scale + np.random.randn(num_frames) * noise
            for j in range(nq)
        ])

        # 在某些 episode 中注入异常
        if i % 5 == 0:
            # 关节 0 超限
            qpos[50:55, 0] = 3.5
        if i % 7 == 0:
            # 帧间跳变
            qpos[100, 2] += 2.0

        action = np.column_stack([
            np.sin(t * (j + 1) + 0.1) * 0.4 for j in range(nu)
        ])

        data = {
            "observations": {"qpos": qpos},
            "action": action,
        }
        episodes.append((f"ep_{i:04d}", data))

    return episodes


# ============================================================
# 演示函数
# ============================================================

def section_1_timeseries():
    """第 1 节: 关节时间序列可视化"""
    print(DIVIDER)
    print("第 1 节：关节角度时间序列")
    print(DIVIDER)

    print("""
    时间序列图是最基本的可视化方式:
    - 绿色半透明带: 关节允许范围
    - 红色虚线: 限位边界
    - 红色填充: 超限区域
    """)

    if not HAS_MPL:
        print("  跳过 (matplotlib 未安装)")
        return

    joint_names = ["J1_base", "J2_shoulder", "J3_elbow", "J4_wrist1",
                   "J5_wrist2", "J6_wrist3", "J7_finger"]
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_report")
    viz = QualityVisualizer(output_dir=output_dir, joint_names=joint_names, hz=50)

    episodes = generate_demo_episodes(num_episodes=15)

    lower = np.array([-np.pi] * 7)
    upper = np.array([np.pi] * 7)

    viz.plot_all(episodes, lower, upper)


def section_2_individual_plots():
    """第 2 节: 单独调用各图表"""
    print("\n" + DIVIDER)
    print("第 2 节：单独调用可视化方法")
    print(DIVIDER)

    if not HAS_MPL:
        print("  跳过 (matplotlib 未安装)")
        return

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_report")
    viz = QualityVisualizer(output_dir=output_dir, hz=50)

    # 生成带明显异常的数据
    T = 300
    t = np.linspace(0, 6 * np.pi, T)
    qpos = np.column_stack([np.sin(t * (j + 1)) * 0.5 for j in range(7)])

    # 注入多种异常
    qpos[80:85, 0] = 4.0    # 超限
    qpos[150, 3] += 3.0     # 跳变
    qpos[200:210, 5] = 0.42  # 死关节片段

    print(f"\n  生成带异常的单独图表...")

    viz.plot_joint_timeseries(
        qpos,
        joint_limits_lower=np.full(7, -np.pi),
        joint_limits_upper=np.full(7, np.pi),
        title="Demo: Joint Time Series with Anomalies",
        filename="demo_timeseries_anomaly.png"
    )

    viz.plot_velocity_heatmap(
        qpos,
        filename="demo_velocity_heatmap.png"
    )

    # 3D 轨迹（圆形轨迹 + 扰动）
    theta = np.linspace(0, 4 * np.pi, T)
    positions = np.column_stack([
        np.cos(theta) * 0.3 + np.random.randn(T) * 0.01,
        np.sin(theta) * 0.3 + np.random.randn(T) * 0.01,
        np.linspace(0.1, 0.4, T) + np.random.randn(T) * 0.005,
    ])
    viz.plot_3d_trajectory(positions, title="Demo: Spiral 3D Trajectory",
                           filename="demo_3d_spiral.png")


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("第 6 章 · 03 - 数据质量可视化工具箱")
    print("用图表快速发现数据质量问题")
    print(DIVIDER)

    section_1_timeseries()
    section_2_individual_plots()

    print("\n" + DIVIDER)
    print("✅ 可视化工具演示完成！")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_report")
    print(f"   所有图表保存在: {output_dir}/")
    print(DIVIDER)


if __name__ == "__main__":
    main()
