"""
第 6 章 · 02 - 数据统计分析 (Data Statistics)

目标: 全面计算数据集的统计特征，为数据质量评估和模型训练提供定量依据。

核心知识点:
  1. 关节级统计: 均值、标准差、最小值、最大值、百分位数
  2. 轨迹级统计: 时长、平滑度、覆盖范围
  3. 数据集级统计: episode 数量、总帧数、关节范围分布
  4. 相关性分析: 关节间的联动关系
  5. 频谱分析: FFT 揭示周期性运动模式
  6. 分布偏移检测: 比较两个数据集的分布差异

运行: python 02_data_statistics.py
依赖: pip install numpy matplotlib scipy
输出: data_statistics.png (统计可视化)
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# 安全导入可视化
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("警告: matplotlib 未安装，将跳过可视化。pip install matplotlib")

try:
    from scipy import signal, stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("提示: scipy 未安装，部分高级分析将使用简化版本。pip install scipy")


# ============================================================
# 关节级统计
# ============================================================

@dataclass
class JointStatistics:
    """
    单个关节的统计量。

    包含描述性统计、百分位数、运动范围等。
    """
    joint_index: int
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    median: float = 0.0
    p5: float = 0.0      # 第 5 百分位
    p25: float = 0.0     # 第 25 百分位 (Q1)
    p75: float = 0.0     # 第 75 百分位 (Q3)
    p95: float = 0.0     # 第 95 百分位
    range: float = 0.0   # max - min
    iqr: float = 0.0     # Q3 - Q1
    skewness: float = 0.0
    kurtosis: float = 0.0

    def to_dict(self) -> dict:
        return {
            "joint_index": self.joint_index,
            "mean": self.mean, "std": self.std,
            "min": self.min_val, "max": self.max_val,
            "median": self.median,
            "percentiles": {"p5": self.p5, "p25": self.p25, "p75": self.p75, "p95": self.p95},
            "range": self.range, "iqr": self.iqr,
            "skewness": self.skewness, "kurtosis": self.kurtosis,
        }


# ============================================================
# 轨迹级统计
# ============================================================

@dataclass
class TrajectoryStatistics:
    """
    单条轨迹的统计量。

    包含时长、平滑度、运动覆盖范围等宏观指标。
    """
    episode_id: str = ""
    num_frames: int = 0
    duration_seconds: float = 0.0
    smoothness: float = 0.0       # 越小越平滑（基于加速度）
    total_path_length: float = 0.0  # 关节空间总路径长度
    joint_coverage: float = 0.0   # 使用了多少关节范围

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "num_frames": self.num_frames,
            "duration_seconds": self.duration_seconds,
            "smoothness": self.smoothness,
            "total_path_length": self.total_path_length,
            "joint_coverage": self.joint_coverage,
        }


# ============================================================
# 数据集统计类
# ============================================================

class DatasetStatistics:
    """
    数据集级别的综合统计分析器。

    使用方法:
        ds_stats = DatasetStatistics(hz=50)
        for ep_id, data in episodes:
            ds_stats.add_episode(ep_id, data)
        ds_stats.compute()
        ds_stats.print_summary()
        ds_stats.export_json("stats.json")
        ds_stats.save_visualization("data_statistics.png")
    """

    def __init__(self, hz: float = 50.0, joint_names: Optional[List[str]] = None):
        self.hz = hz
        self.joint_names = joint_names

        # 原始数据缓存
        self._all_qpos: List[np.ndarray] = []
        self._all_actions: List[np.ndarray] = []
        self._episode_ids: List[str] = []

        # 计算结果
        self.joint_stats: List[JointStatistics] = []
        self.trajectory_stats: List[TrajectoryStatistics] = []
        self.correlation_matrix: Optional[np.ndarray] = None
        self.fft_results: Optional[Dict] = None
        self.dataset_summary: Dict[str, Any] = {}

        self._computed = False

    def add_episode(self, episode_id: str, data: dict):
        """添加一个 episode 的数据"""
        qpos = data.get("observations", {}).get("qpos", None)
        action = data.get("action", None)

        if qpos is not None:
            self._all_qpos.append(qpos.copy())
        if action is not None:
            self._all_actions.append(action.copy())
        self._episode_ids.append(episode_id)

    # ----------------------------------------------------------
    # 关节级统计
    # ----------------------------------------------------------
    def _compute_joint_stats(self, all_qpos_concat: np.ndarray):
        """计算每个关节的描述性统计"""
        nq = all_qpos_concat.shape[1]
        self.joint_stats = []

        for j in range(nq):
            col = all_qpos_concat[:, j]
            js = JointStatistics(joint_index=j)
            js.mean = float(np.mean(col))
            js.std = float(np.std(col))
            js.min_val = float(np.min(col))
            js.max_val = float(np.max(col))
            js.median = float(np.median(col))
            js.p5 = float(np.percentile(col, 5))
            js.p25 = float(np.percentile(col, 25))
            js.p75 = float(np.percentile(col, 75))
            js.p95 = float(np.percentile(col, 95))
            js.range = js.max_val - js.min_val
            js.iqr = js.p75 - js.p25

            # 偏度和峰度
            if HAS_SCIPY:
                js.skewness = float(stats.skew(col))
                js.kurtosis = float(stats.kurtosis(col))
            else:
                n = len(col)
                m = js.mean
                s = js.std if js.std > 0 else 1e-10
                js.skewness = float(np.mean(((col - m) / s) ** 3))
                js.kurtosis = float(np.mean(((col - m) / s) ** 4) - 3)

            self.joint_stats.append(js)

    # ----------------------------------------------------------
    # 轨迹级统计
    # ----------------------------------------------------------
    def _compute_trajectory_stats(self):
        """计算每条轨迹的宏观统计"""
        self.trajectory_stats = []

        for i, qpos in enumerate(self._all_qpos):
            ts = TrajectoryStatistics()
            ts.episode_id = self._episode_ids[i] if i < len(self._episode_ids) else f"ep_{i}"
            ts.num_frames = qpos.shape[0]
            ts.duration_seconds = ts.num_frames / self.hz

            # 平滑度: 基于加速度的 L2 范数均值（越小越平滑）
            if qpos.shape[0] >= 3:
                accel = np.diff(qpos, n=2, axis=0) * (self.hz ** 2)
                ts.smoothness = float(np.mean(np.linalg.norm(accel, axis=1)))
            else:
                ts.smoothness = 0.0

            # 总路径长度: 关节空间中的累积位移
            if qpos.shape[0] >= 2:
                diffs = np.diff(qpos, axis=0)
                ts.total_path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

            # 关节覆盖范围: 实际使用范围占总可用范围的比例
            ranges = qpos.max(axis=0) - qpos.min(axis=0)
            # 假设关节范围 [-π, π]
            max_possible = 2 * np.pi
            ts.joint_coverage = float(np.mean(np.clip(ranges / max_possible, 0, 1)))

            self.trajectory_stats.append(ts)

    # ----------------------------------------------------------
    # 相关性分析
    # ----------------------------------------------------------
    def _compute_correlation(self, all_qpos_concat: np.ndarray):
        """
        计算关节间的 Pearson 相关系数矩阵。

        高相关性意味着两个关节联动，可能有机械耦合或控制策略的协调。
        """
        nq = all_qpos_concat.shape[1]
        self.correlation_matrix = np.corrcoef(all_qpos_concat.T)  # (nq, nq)

    # ----------------------------------------------------------
    # 频谱分析
    # ----------------------------------------------------------
    def _compute_fft(self, all_qpos_concat: np.ndarray, max_freq_display: float = 25.0):
        """
        对每个关节进行 FFT 频谱分析。

        揭示运动中的周期性模式:
          - 步态中的步频
          - 操作中的重复动作频率
          - 异常高频振荡（可能是噪声或不稳定）
        """
        nq = all_qpos_concat.shape[1]
        n = all_qpos_concat.shape[0]

        freqs = np.fft.rfftfreq(n, d=1.0 / self.hz)
        freq_mask = freqs <= max_freq_display

        self.fft_results = {
            "frequencies": freqs[freq_mask].tolist(),
            "joint_spectra": {},
            "dominant_frequencies": {},
        }

        for j in range(nq):
            # 去均值后做 FFT
            sig = all_qpos_concat[:, j] - all_qpos_concat[:, j].mean()

            if HAS_SCIPY:
                # 使用窗函数减少频谱泄漏
                window = signal.windows.hann(n)
                sig_windowed = sig * window
                spectrum = np.abs(np.fft.rfft(sig_windowed))
            else:
                spectrum = np.abs(np.fft.rfft(sig))

            spectrum_masked = spectrum[freq_mask]
            self.fft_results["joint_spectra"][f"joint_{j}"] = spectrum_masked.tolist()

            # 找主频（排除直流分量）
            if len(spectrum_masked) > 1:
                peak_idx = np.argmax(spectrum_masked[1:]) + 1
                dominant_freq = freqs[freq_mask][peak_idx]
                self.fft_results["dominant_frequencies"][f"joint_{j}"] = float(dominant_freq)

    # ----------------------------------------------------------
    # 数据集级汇总
    # ----------------------------------------------------------
    def _compute_dataset_summary(self):
        """汇总数据集的宏观指标"""
        total_frames = sum(ts.num_frames for ts in self.trajectory_stats)
        durations = [ts.duration_seconds for ts in self.trajectory_stats]

        self.dataset_summary = {
            "episode_count": len(self._all_qpos),
            "total_frames": total_frames,
            "total_duration_seconds": sum(durations),
            "avg_episode_frames": total_frames / max(len(self._all_qpos), 1),
            "avg_episode_duration": np.mean(durations) if durations else 0,
            "min_episode_duration": np.min(durations) if durations else 0,
            "max_episode_duration": np.max(durations) if durations else 0,
            "num_joints": self._all_qpos[0].shape[1] if self._all_qpos else 0,
            "hz": self.hz,
        }

    # ----------------------------------------------------------
    # 主计算入口
    # ----------------------------------------------------------
    def compute(self):
        """执行所有统计计算"""
        if not self._all_qpos:
            print("  警告: 无数据，跳过统计计算")
            return

        # 拼接所有 qpos
        all_qpos_concat = np.concatenate(self._all_qpos, axis=0)

        print(f"  正在计算统计... (总帧数: {all_qpos_concat.shape[0]}, "
              f"关节数: {all_qpos_concat.shape[1]})")

        self._compute_joint_stats(all_qpos_concat)
        self._compute_trajectory_stats()
        self._compute_correlation(all_qpos_concat)
        self._compute_fft(all_qpos_concat)
        self._compute_dataset_summary()

        self._computed = True
        print("  统计计算完成！")

    # ----------------------------------------------------------
    # 打印摘要
    # ----------------------------------------------------------
    def print_summary(self):
        if not self._computed:
            print("  请先调用 compute()")
            return

        print(f"\n  📊 数据集概览:")
        print(f"    Episode 数量: {self.dataset_summary['episode_count']}")
        print(f"    总帧数: {self.dataset_summary['total_frames']}")
        print(f"    总时长: {self.dataset_summary['total_duration_seconds']:.1f}s")
        print(f"    平均 episode 帧数: {self.dataset_summary['avg_episode_frames']:.0f}")
        print(f"    关节数: {self.dataset_summary['num_joints']}")

        print(f"\n  📐 关节统计:")
        header = f"    {'关节':>6s} | {'均值':>8s} | {'标准差':>8s} | {'最小值':>8s} | {'最大值':>8s} | {'范围':>8s} | {'偏度':>6s}"
        print(header)
        print("    " + "-" * len(header.strip()))
        for js in self.joint_stats:
            name = self.joint_names[js.joint_index] if self.joint_names and js.joint_index < len(self.joint_names) else f"J{js.joint_index}"
            print(f"    {name:>6s} | {js.mean:>8.4f} | {js.std:>8.4f} | "
                  f"{js.min_val:>8.4f} | {js.max_val:>8.4f} | {js.range:>8.4f} | {js.skewness:>6.2f}")

        if self.fft_results and self.fft_results["dominant_frequencies"]:
            print(f"\n  🎵 主频分析:")
            for joint, freq in self.fft_results["dominant_frequencies"].items():
                print(f"    {joint}: {freq:.2f} Hz")

    # ----------------------------------------------------------
    # 分布偏移检测
    # ----------------------------------------------------------
    @staticmethod
    def compare_distributions(stats_a: "DatasetStatistics",
                              stats_b: "DatasetStatistics") -> Dict[str, Any]:
        """
        比较两个数据集的分布差异。

        使用方法:
          - Kolmogorov-Smirnov 检验（如果有 scipy）
          - 均值/标准差差异
          - Jensen-Shannon 散度（离散化后）

        返回每个关节的偏移检测结果。
        """
        if not stats_a._all_qpos or not stats_b._all_qpos:
            return {"error": "需要两个非空数据集"}

        concat_a = np.concatenate(stats_a._all_qpos, axis=0)
        concat_b = np.concatenate(stats_b._all_qpos, axis=0)

        nq = min(concat_a.shape[1], concat_b.shape[1])
        results = {}

        for j in range(nq):
            col_a = concat_a[:, j]
            col_b = concat_b[:, j]

            comparison = {
                "mean_diff": float(np.mean(col_b) - np.mean(col_a)),
                "std_ratio": float(np.std(col_b) / max(np.std(col_a), 1e-10)),
            }

            if HAS_SCIPY:
                ks_stat, ks_pval = stats.ks_2samp(col_a, col_b)
                comparison["ks_statistic"] = float(ks_stat)
                comparison["ks_pvalue"] = float(ks_pval)
                comparison["distribution_shifted"] = ks_pval < 0.05
            else:
                # 简化版: 用均值差 / 合并标准差 作为效应量
                pooled_std = np.sqrt((np.var(col_a) + np.var(col_b)) / 2)
                effect_size = abs(np.mean(col_b) - np.mean(col_a)) / max(pooled_std, 1e-10)
                comparison["cohens_d"] = float(effect_size)
                comparison["distribution_shifted"] = effect_size > 0.5

            results[f"joint_{j}"] = comparison

        return results

    # ----------------------------------------------------------
    # 导出 JSON
    # ----------------------------------------------------------
    def export_json(self, filepath: str):
        """导出统计结果为 JSON 文件（适合数据平台消费）"""
        if not self._computed:
            print("  请先调用 compute()")
            return

        export_data = {
            "dataset_summary": self.dataset_summary,
            "joint_statistics": [js.to_dict() for js in self.joint_stats],
            "trajectory_statistics": [ts.to_dict() for ts in self.trajectory_stats],
            "dominant_frequencies": self.fft_results.get("dominant_frequencies", {}) if self.fft_results else {},
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"  统计结果已导出: {filepath}")

    # ----------------------------------------------------------
    # 可视化
    # ----------------------------------------------------------
    def save_visualization(self, filepath: str = "data_statistics.png"):
        """生成统计可视化大图并保存"""
        if not HAS_MPL:
            print("  matplotlib 未安装，跳过可视化")
            return
        if not self._computed:
            print("  请先调用 compute()")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("数据集统计分析 (Dataset Statistics)", fontsize=14, fontweight="bold")

        nq = len(self.joint_stats)
        joint_labels = [
            self.joint_names[i] if self.joint_names and i < len(self.joint_names)
            else f"J{i}" for i in range(nq)
        ]

        # --- 图 1: 关节均值和范围 ---
        ax = axes[0, 0]
        means = [js.mean for js in self.joint_stats]
        stds = [js.std for js in self.joint_stats]
        x = np.arange(nq)
        ax.bar(x, means, yerr=stds, capsize=3, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(joint_labels, fontsize=8)
        ax.set_title("Joint Mean ± Std")
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3)

        # --- 图 2: 关节范围箱线图 ---
        ax = axes[0, 1]
        if self._all_qpos:
            concat = np.concatenate(self._all_qpos, axis=0)
            bp = ax.boxplot([concat[:, j] for j in range(nq)], labels=joint_labels, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightcoral")
                patch.set_alpha(0.7)
        ax.set_title("Joint Value Distribution")
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3)

        # --- 图 3: 相关性热图 ---
        ax = axes[0, 2]
        if self.correlation_matrix is not None:
            im = ax.imshow(self.correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(nq))
            ax.set_yticks(range(nq))
            ax.set_xticklabels(joint_labels, fontsize=8)
            ax.set_yticklabels(joint_labels, fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Joint Correlation")

        # --- 图 4: episode 时长分布 ---
        ax = axes[1, 0]
        durations = [ts.duration_seconds for ts in self.trajectory_stats]
        if durations:
            ax.hist(durations, bins=min(20, len(durations)), color="mediumpurple", alpha=0.8, edgecolor="white")
        ax.set_title("Episode Duration Distribution")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

        # --- 图 5: 频谱 ---
        ax = axes[1, 1]
        if self.fft_results and "frequencies" in self.fft_results:
            freqs = self.fft_results["frequencies"]
            for j in range(min(nq, 4)):
                key = f"joint_{j}"
                if key in self.fft_results["joint_spectra"]:
                    spectrum = self.fft_results["joint_spectra"][key]
                    ax.plot(freqs, spectrum, label=joint_labels[j], alpha=0.7)
            ax.set_title("Frequency Spectrum (first 4 joints)")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # --- 图 6: 平滑度 vs 路径长度 ---
        ax = axes[1, 2]
        smoothness = [ts.smoothness for ts in self.trajectory_stats]
        path_lengths = [ts.total_path_length for ts in self.trajectory_stats]
        if smoothness and path_lengths:
            sc = ax.scatter(path_lengths, smoothness, c="darkorange", alpha=0.6, edgecolors="gray", s=50)
            ax.set_xlabel("Total Path Length")
            ax.set_ylabel("Smoothness (lower = smoother)")
            ax.set_title("Smoothness vs Path Length")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  可视化已保存: {filepath}")


# ============================================================
# 测试数据生成
# ============================================================

def generate_dataset(num_episodes=20, nq=7, nu=7, hz=50):
    """生成用于演示的模拟数据集"""
    episodes = []
    for i in range(num_episodes):
        num_frames = np.random.randint(150, 500)
        t = np.linspace(0, 4 * np.pi, num_frames)

        # 每个 episode 的运动模式略有不同
        freq_scale = 1.0 + np.random.randn() * 0.2
        amp_scale = 0.3 + np.random.rand() * 0.2
        noise_level = 0.01 + np.random.rand() * 0.02

        qpos = np.column_stack([
            np.sin(t * (j + 1) * freq_scale) * amp_scale + np.random.randn(num_frames) * noise_level
            for j in range(nq)
        ])
        qvel = np.diff(qpos, axis=0, prepend=qpos[:1]) * hz
        action = np.column_stack([
            np.sin(t * (j + 1) * freq_scale + 0.1) * amp_scale * 0.8
            for j in range(nu)
        ])

        data = {
            "observations": {"qpos": qpos, "qvel": qvel},
            "action": action,
        }
        episodes.append((f"ep_{i:04d}", data))

    return episodes


def generate_shifted_dataset(num_episodes=15, nq=7, nu=7, hz=50):
    """生成分布有偏移的数据集（用于对比检测）"""
    episodes = []
    for i in range(num_episodes):
        num_frames = np.random.randint(100, 300)
        t = np.linspace(0, 4 * np.pi, num_frames)

        # 偏移: 更高频率、不同振幅、偏置
        freq_scale = 1.5 + np.random.randn() * 0.3
        amp_scale = 0.5 + np.random.rand() * 0.3
        offset = 0.2

        qpos = np.column_stack([
            np.sin(t * (j + 1) * freq_scale) * amp_scale + offset
            for j in range(nq)
        ])
        qvel = np.diff(qpos, axis=0, prepend=qpos[:1]) * hz
        action = np.column_stack([
            np.sin(t * (j + 1) * freq_scale + 0.1) * amp_scale * 0.8
            for j in range(nu)
        ])

        data = {
            "observations": {"qpos": qpos, "qvel": qvel},
            "action": action,
        }
        episodes.append((f"shifted_ep_{i:04d}", data))

    return episodes


# ============================================================
# 演示函数
# ============================================================

def section_1_joint_statistics():
    """第 1 节: 关节级统计"""
    print(DIVIDER)
    print("第 1 节：关节级统计 (Joint-Level Statistics)")
    print(DIVIDER)

    print("""
    关节级统计是最基础的数据分析:
    - 均值/标准差: 数据的中心位置和分散程度
    - 百分位数: 对异常值不敏感的分布描述
    - 偏度: 分布的对称性 (0=对称, >0右偏, <0左偏)
    - 峰度: 分布的尖峰程度 (0=正态, >0尖峰, <0扁平)
    """)

    joint_names = ["J1_base", "J2_shoulder", "J3_elbow", "J4_wrist1",
                   "J5_wrist2", "J6_wrist3", "J7_finger"]

    ds = DatasetStatistics(hz=50, joint_names=joint_names)
    episodes = generate_dataset(num_episodes=20)
    for ep_id, data in episodes:
        ds.add_episode(ep_id, data)
    ds.compute()
    ds.print_summary()

    return ds


def section_2_trajectory_statistics():
    """第 2 节: 轨迹级统计"""
    print("\n" + DIVIDER)
    print("第 2 节：轨迹级统计 (Trajectory-Level Statistics)")
    print(DIVIDER)

    print("""
    轨迹级统计从宏观角度描述每个 episode:
    - 时长: 太短可能是中断，太长可能需要分割
    - 平滑度: 基于加速度范数，衡量运动的流畅性
    - 路径长度: 关节空间中走过的总距离
    - 覆盖范围: 使用了多少可用关节空间
    """)

    ds = DatasetStatistics(hz=50)
    episodes = generate_dataset(num_episodes=15)
    for ep_id, data in episodes:
        ds.add_episode(ep_id, data)
    ds.compute()

    print(f"\n  轨迹统计表:")
    print(f"    {'Episode':>16s} | {'帧数':>6s} | {'时长(s)':>8s} | {'平滑度':>10s} | {'路径长':>10s} | {'覆盖率':>8s}")
    print(f"    " + "-" * 70)
    for ts in ds.trajectory_stats[:10]:
        print(f"    {ts.episode_id:>16s} | {ts.num_frames:>6d} | {ts.duration_seconds:>8.2f} | "
              f"{ts.smoothness:>10.1f} | {ts.total_path_length:>10.2f} | {ts.joint_coverage:>8.1%}")
    if len(ds.trajectory_stats) > 10:
        print(f"    ... (共 {len(ds.trajectory_stats)} 条轨迹)")


def section_3_correlation_and_fft():
    """第 3 节: 相关性分析 & 频谱分析"""
    print("\n" + DIVIDER)
    print("第 3 节：相关性分析 & 频谱分析")
    print(DIVIDER)

    print("""
    相关性分析:
    - 发现关节间的联动关系
    - 高相关性可能表明机械耦合或协调控制
    - 低相关性说明关节独立运动

    频谱分析 (FFT):
    - 揭示周期性运动的频率
    - 高频成分可能是噪声
    - 可用于对比不同策略的运动模式
    """)

    ds = DatasetStatistics(hz=50)
    episodes = generate_dataset(num_episodes=20)
    for ep_id, data in episodes:
        ds.add_episode(ep_id, data)
    ds.compute()

    # 显示相关性矩阵
    if ds.correlation_matrix is not None:
        nq = ds.correlation_matrix.shape[0]
        print(f"\n  关节相关性矩阵 ({nq}x{nq}):")
        print("        ", end="")
        for j in range(nq):
            print(f"{'J'+str(j):>7s}", end="")
        print()
        for i in range(nq):
            print(f"    J{i}  ", end="")
            for j in range(nq):
                val = ds.correlation_matrix[i, j]
                print(f"{val:>7.3f}", end="")
            print()

    # 显示主频
    if ds.fft_results:
        print(f"\n  各关节主频:")
        for joint, freq in ds.fft_results.get("dominant_frequencies", {}).items():
            print(f"    {joint}: {freq:.2f} Hz")


def section_4_distribution_shift():
    """第 4 节: 分布偏移检测"""
    print("\n" + DIVIDER)
    print("第 4 节：分布偏移检测 (Distribution Shift Detection)")
    print(DIVIDER)

    print("""
    当训练数据和部署环境不同时，数据分布会发生偏移。
    常见场景:
    - 仿真到真实 (sim-to-real gap)
    - 不同操作员采集的数据
    - 不同机器人硬件

    检测方法:
    - KS 检验 (Kolmogorov-Smirnov test)
    - Cohen's d 效应量
    - 均值差异 / 标准差比率
    """)

    # 构建两个数据集
    ds_a = DatasetStatistics(hz=50)
    for ep_id, data in generate_dataset(num_episodes=20):
        ds_a.add_episode(ep_id, data)
    ds_a.compute()

    ds_b = DatasetStatistics(hz=50)
    for ep_id, data in generate_shifted_dataset(num_episodes=15):
        ds_b.add_episode(ep_id, data)
    ds_b.compute()

    # 比较分布
    comparison = DatasetStatistics.compare_distributions(ds_a, ds_b)

    print(f"\n  分布比较结果 (数据集 A vs 数据集 B):")
    for joint, result in comparison.items():
        shifted = result.get("distribution_shifted", False)
        icon = "🔴" if shifted else "🟢"
        print(f"    {icon} {joint}: 均值差={result['mean_diff']:.4f}, "
              f"标准差比={result['std_ratio']:.3f}", end="")
        if "ks_pvalue" in result:
            print(f", KS p={result['ks_pvalue']:.4f}", end="")
        if "cohens_d" in result:
            print(f", Cohen's d={result['cohens_d']:.3f}", end="")
        print()


def section_5_export_and_visualize():
    """第 5 节: 导出与可视化"""
    print("\n" + DIVIDER)
    print("第 5 节：导出 JSON & 可视化")
    print(DIVIDER)

    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]
    ds = DatasetStatistics(hz=50, joint_names=joint_names)
    episodes = generate_dataset(num_episodes=25)
    for ep_id, data in episodes:
        ds.add_episode(ep_id, data)
    ds.compute()

    # 导出 JSON
    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(output_dir, "data_statistics.json")
    ds.export_json(json_path)

    # 生成可视化
    png_path = os.path.join(output_dir, "data_statistics.png")
    ds.save_visualization(png_path)


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("第 6 章 · 02 - 数据统计分析 (Data Statistics)")
    print("全面计算数据集的统计特征")
    print(DIVIDER)

    section_1_joint_statistics()
    section_2_trajectory_statistics()
    section_3_correlation_and_fft()
    section_4_distribution_shift()
    section_5_export_and_visualize()

    print("\n" + DIVIDER)
    print("✅ 数据统计分析演示完成！")
    print(DIVIDER)


if __name__ == "__main__":
    main()
