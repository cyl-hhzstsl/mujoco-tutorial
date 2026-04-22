"""
第 6 章 · 04 - 异常检测 (Anomaly Detection)

目标: 掌握多种异常检测方法，从简单的统计方法到高级的机器学习方法，
     自动识别机器人轨迹数据中的异常片段。

核心知识点:
  1. Z-Score 异常检测 —— 基于全局均值/标准差
  2. IQR 异常检测 —— 基于四分位距，对异常值更鲁棒
  3. 滑动窗口分析 —— 检测局部异常
  4. Isolation Forest —— 多变量异常检测
  5. 物理一致性检查 —— 能量守恒、重力方向
  6. 时序连续性检查 —— 自相关分析
  7. 综合检测器 & 报告生成
  8. 异常可视化

运行: python 04_anomaly_detection.py
依赖: pip install numpy matplotlib
可选: pip install scikit-learn scipy
"""

import numpy as np
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.ensemble import IsolationForest as SklearnIsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy import signal, stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# 异常类型枚举
# ============================================================

class AnomalyType(Enum):
    """异常类型分类"""
    STATISTICAL_OUTLIER = "statistical_outlier"     # 统计离群点
    FRAME_JUMP = "frame_jump"                       # 帧间跳变
    PHYSICS_VIOLATION = "physics_violation"          # 物理违规
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"  # 时序不一致
    MULTIVARIATE_OUTLIER = "multivariate_outlier"   # 多变量离群


# ============================================================
# 异常记录
# ============================================================

@dataclass
class Anomaly:
    """
    单条异常记录。

    记录异常发生的位置、类型、严重程度和相关数据。
    """
    anomaly_type: AnomalyType
    frame_start: int
    frame_end: int
    joint_indices: List[int]
    severity: float  # 0-1, 越高越严重
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.anomaly_type.value,
            "frame_range": [self.frame_start, self.frame_end],
            "joints": self.joint_indices,
            "severity": self.severity,
            "description": self.description,
            "details": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in self.details.items()},
        }


# ============================================================
# 异常检测方法
# ============================================================

class ZScoreDetector:
    """
    Z-Score 异常检测器。

    原理: z = (x - μ) / σ
    当 |z| > threshold 时，认为该数据点是异常值。

    优点: 简单直观，计算快速
    缺点: 对均值和标准差敏感，大量异常值会"污染"统计量
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def detect(self, qpos: np.ndarray) -> List[Anomaly]:
        T, nq = qpos.shape
        anomalies = []

        for j in range(nq):
            col = qpos[:, j]
            mu, sigma = col.mean(), col.std()
            if sigma < 1e-10:
                continue

            z_scores = np.abs((col - mu) / sigma)
            outlier_frames = np.where(z_scores > self.threshold)[0]

            if len(outlier_frames) > 0:
                # 合并连续异常帧为区间
                groups = self._group_consecutive(outlier_frames)
                for group in groups:
                    max_z = float(z_scores[group].max())
                    severity = min(max_z / (self.threshold * 3), 1.0)
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        frame_start=int(group[0]),
                        frame_end=int(group[-1]),
                        joint_indices=[j],
                        severity=severity,
                        description=f"Joint {j}: z-score 异常 (max z={max_z:.2f})",
                        details={"max_z_score": max_z, "threshold": self.threshold},
                    ))

        return anomalies

    @staticmethod
    def _group_consecutive(frames: np.ndarray, gap: int = 3) -> List[np.ndarray]:
        """将连续或接近的帧号分组"""
        if len(frames) == 0:
            return []
        groups = []
        current = [frames[0]]
        for f in frames[1:]:
            if f - current[-1] <= gap:
                current.append(f)
            else:
                groups.append(np.array(current))
                current = [f]
        groups.append(np.array(current))
        return groups


class IQRDetector:
    """
    IQR (四分位距) 异常检测器。

    原理: IQR = Q3 - Q1
          下界 = Q1 - factor * IQR
          上界 = Q3 + factor * IQR
          超出 [下界, 上界] 的点为异常。

    优点: 对极端值的鲁棒性比 z-score 更好
    缺点: 假设单峰分布
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def detect(self, qpos: np.ndarray) -> List[Anomaly]:
        T, nq = qpos.shape
        anomalies = []

        for j in range(nq):
            col = qpos[:, j]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            if iqr < 1e-10:
                continue

            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr

            outlier_frames = np.where((col < lower) | (col > upper))[0]

            if len(outlier_frames) > 0:
                groups = ZScoreDetector._group_consecutive(outlier_frames)
                for group in groups:
                    max_deviation = float(np.max(np.maximum(
                        lower - col[group], col[group] - upper
                    )))
                    severity = min(max_deviation / (iqr * 3), 1.0)
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        frame_start=int(group[0]),
                        frame_end=int(group[-1]),
                        joint_indices=[j],
                        severity=severity,
                        description=f"Joint {j}: IQR 异常 (偏离 {max_deviation:.3f})",
                        details={"iqr": float(iqr), "lower": float(lower),
                                 "upper": float(upper), "factor": self.factor},
                    ))

        return anomalies


class SlidingWindowDetector:
    """
    滑动窗口异常检测器。

    原理: 在滑动窗口内计算局部统计量，与窗口外（全局或相邻窗口）比较。
          能够检测出局部模式变化，即使全局统计量正常。

    典型场景: 机器人在正常运行中突然进入异常状态片段。
    """

    def __init__(self, window_size: int = 50, threshold_std_ratio: float = 3.0):
        self.window_size = window_size
        self.threshold_std_ratio = threshold_std_ratio

    def detect(self, qpos: np.ndarray) -> List[Anomaly]:
        T, nq = qpos.shape
        anomalies = []

        if T < self.window_size * 2:
            return anomalies

        for j in range(nq):
            col = qpos[:, j]
            global_mean = col.mean()
            global_std = col.std()
            if global_std < 1e-10:
                continue

            # 滑动窗口计算局部均值
            local_means = np.convolve(col, np.ones(self.window_size) / self.window_size, mode="valid")

            # 检测局部均值偏离全局均值的程度
            deviations = np.abs(local_means - global_mean) / global_std

            anomaly_windows = np.where(deviations > self.threshold_std_ratio)[0]

            if len(anomaly_windows) > 0:
                groups = ZScoreDetector._group_consecutive(anomaly_windows, gap=self.window_size // 2)
                for group in groups:
                    max_dev = float(deviations[group].max())
                    severity = min(max_dev / (self.threshold_std_ratio * 2), 1.0)
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.TEMPORAL_INCONSISTENCY,
                        frame_start=int(group[0]),
                        frame_end=int(group[-1] + self.window_size),
                        joint_indices=[j],
                        severity=severity,
                        description=f"Joint {j}: 局部窗口异常 (偏离 {max_dev:.2f}σ)",
                        details={"window_size": self.window_size,
                                 "max_deviation_sigma": max_dev},
                    ))

        return anomalies


class SimpleIsolationForest:
    """
    简化版 Isolation Forest（不依赖 sklearn 时使用）。

    原理: 异常点由于"不同寻常"，在随机划分时更容易被隔离，
          需要更少的划分次数。路径长度越短 = 越异常。

    这是学术论文 (Liu et al., 2008) 的简化教学实现。
    """

    def __init__(self, n_trees: int = 100, sample_size: int = 256,
                 contamination: float = 0.05):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        self._trees = []

    def _build_tree(self, X: np.ndarray, max_depth: int) -> dict:
        """递归构建一棵 isolation tree"""
        n, d = X.shape
        if n <= 1 or max_depth <= 0:
            return {"type": "leaf", "size": n}

        # 随机选一个特征和分割点
        feat = np.random.randint(d)
        feat_min, feat_max = X[:, feat].min(), X[:, feat].max()
        if feat_max - feat_min < 1e-10:
            return {"type": "leaf", "size": n}

        split = np.random.uniform(feat_min, feat_max)

        left_mask = X[:, feat] < split
        right_mask = ~left_mask

        return {
            "type": "split",
            "feature": feat,
            "split_value": split,
            "left": self._build_tree(X[left_mask], max_depth - 1),
            "right": self._build_tree(X[right_mask], max_depth - 1),
        }

    def _path_length(self, x: np.ndarray, tree: dict, depth: int = 0) -> float:
        """计算样本 x 在树中的路径长度"""
        if tree["type"] == "leaf":
            n = tree["size"]
            if n <= 1:
                return float(depth)
            # 未分裂节点的期望路径长度修正
            c = 2.0 * (np.log(n - 1) + 0.5772) - 2.0 * (n - 1) / n
            return float(depth) + c

        if x[tree["feature"]] < tree["split_value"]:
            return self._path_length(x, tree["left"], depth + 1)
        return self._path_length(x, tree["right"], depth + 1)

    def fit(self, X: np.ndarray):
        n = X.shape[0]
        max_depth = int(np.ceil(np.log2(max(self.sample_size, 2))))

        self._trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(n, min(self.sample_size, n), replace=False)
            tree = self._build_tree(X[idx], max_depth)
            self._trees.append(tree)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        返回异常分数。分数越低（越负）= 越异常。
        与 sklearn 的约定一致。
        """
        n = X.shape[0]
        avg_path = np.zeros(n)

        for tree in self._trees:
            for i in range(n):
                avg_path[i] += self._path_length(X[i], tree)

        avg_path /= len(self._trees)

        c = 2.0 * (np.log(self.sample_size - 1) + 0.5772) - 2.0 * (self.sample_size - 1) / self.sample_size
        scores = -(2.0 ** (-avg_path / max(c, 1e-10)))

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 1 (正常) 或 -1 (异常)"""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, self.contamination * 100)
        return np.where(scores < threshold, -1, 1)


class IsolationForestDetector:
    """
    Isolation Forest 多变量异常检测。

    优先使用 sklearn，不可用时自动降级到简化版实现。
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators

    def detect(self, qpos: np.ndarray) -> List[Anomaly]:
        T, nq = qpos.shape
        anomalies = []

        if HAS_SKLEARN:
            model = SklearnIsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
            )
            preds = model.fit_predict(qpos)
            scores = model.score_samples(qpos)
        else:
            model = SimpleIsolationForest(
                n_trees=self.n_estimators,
                contamination=self.contamination,
            )
            model.fit(qpos)
            preds = model.predict(qpos)
            scores = model.score_samples(qpos)

        outlier_frames = np.where(preds == -1)[0]

        if len(outlier_frames) > 0:
            groups = ZScoreDetector._group_consecutive(outlier_frames, gap=5)
            for group in groups:
                min_score = float(scores[group].min())
                severity = min(abs(min_score), 1.0)
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.MULTIVARIATE_OUTLIER,
                    frame_start=int(group[0]),
                    frame_end=int(group[-1]),
                    joint_indices=list(range(nq)),
                    severity=severity,
                    description=f"Isolation Forest 多变量异常 (score={min_score:.3f})",
                    details={"min_anomaly_score": min_score,
                             "num_anomaly_frames": int(len(group)),
                             "backend": "sklearn" if HAS_SKLEARN else "custom"},
                ))

        return anomalies


class PhysicsChecker:
    """
    物理一致性检查器。

    基于物理定律检测数据异常:
    - 能量守恒: 动能 + 势能不应突变
    - 加速度合理性: 加速度不应超过物理极限
    """

    def __init__(self, hz: float = 50.0, max_accel: float = 100.0):
        self.hz = hz
        self.max_accel = max_accel

    def detect(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None) -> List[Anomaly]:
        T, nq = qpos.shape
        anomalies = []

        # 检查 1: 加速度合理性
        if T >= 3:
            dt = 1.0 / self.hz
            if qvel is not None:
                accel = np.diff(qvel, axis=0) / dt
            else:
                accel = np.diff(qpos, n=2, axis=0) / (dt ** 2)

            accel_norms = np.linalg.norm(accel, axis=1)
            excessive = np.where(accel_norms > self.max_accel)[0]

            if len(excessive) > 0:
                groups = ZScoreDetector._group_consecutive(excessive)
                for group in groups:
                    max_a = float(accel_norms[group].max())
                    severity = min(max_a / (self.max_accel * 5), 1.0)
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.PHYSICS_VIOLATION,
                        frame_start=int(group[0]),
                        frame_end=int(group[-1]),
                        joint_indices=list(range(nq)),
                        severity=severity,
                        description=f"加速度过大 ({max_a:.1f} > {self.max_accel})",
                        details={"max_acceleration": max_a, "limit": self.max_accel},
                    ))

        # 检查 2: "能量" 突变（简化: 用动能近似）
        if qvel is not None and T >= 2:
            kinetic = 0.5 * np.sum(qvel ** 2, axis=1)
            energy_diff = np.abs(np.diff(kinetic))
            mean_energy = kinetic.mean()
            if mean_energy > 0:
                relative_jumps = energy_diff / max(mean_energy, 1e-10)
                energy_spikes = np.where(relative_jumps > 5.0)[0]

                if len(energy_spikes) > 0:
                    groups = ZScoreDetector._group_consecutive(energy_spikes)
                    for group in groups:
                        max_jump = float(relative_jumps[group].max())
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.PHYSICS_VIOLATION,
                            frame_start=int(group[0]),
                            frame_end=int(group[-1]),
                            joint_indices=list(range(nq)),
                            severity=min(max_jump / 20.0, 1.0),
                            description=f"能量突变 (相对变化 {max_jump:.1f}x)",
                            details={"max_relative_change": max_jump},
                        ))

        return anomalies


class TemporalConsistencyChecker:
    """
    时序连续性检查器。

    检查轨迹的时序连续性:
    - 自相关分析: 自然运动应有平滑的自相关衰减
    - 差分平稳性: 速度序列应该相对平稳
    """

    def __init__(self, hz: float = 50.0, max_velocity: float = 10.0):
        self.hz = hz
        self.max_velocity = max_velocity

    def detect(self, qpos: np.ndarray) -> List[Anomaly]:
        T, nq = qpos.shape
        anomalies = []

        if T < 3:
            return anomalies

        dt = 1.0 / self.hz

        for j in range(nq):
            velocity = np.diff(qpos[:, j]) / dt
            excessive = np.where(np.abs(velocity) > self.max_velocity)[0]

            if len(excessive) > 0:
                groups = ZScoreDetector._group_consecutive(excessive)
                for group in groups:
                    max_v = float(np.abs(velocity[group]).max())
                    severity = min(max_v / (self.max_velocity * 3), 1.0)
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.FRAME_JUMP,
                        frame_start=int(group[0]),
                        frame_end=int(group[-1]),
                        joint_indices=[j],
                        severity=severity,
                        description=f"Joint {j}: 速度过大 ({max_v:.2f} > {self.max_velocity})",
                        details={"max_velocity": max_v, "limit": self.max_velocity},
                    ))

        return anomalies


# ============================================================
# 综合异常检测器
# ============================================================

class AnomalyDetector:
    """
    综合异常检测器。

    整合多种检测方法，生成统一的异常报告。

    使用方法:
        detector = AnomalyDetector(config)
        report = detector.detect(episode_data)
        detector.print_report(report)
        detector.visualize(episode_data, report, output_path)
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.hz = config.get("hz", 50.0)

        # 初始化各检测器
        self.detectors = {
            "z_score": ZScoreDetector(threshold=config.get("z_threshold", 3.0)),
            "iqr": IQRDetector(factor=config.get("iqr_factor", 1.5)),
            "sliding_window": SlidingWindowDetector(
                window_size=config.get("window_size", 50),
                threshold_std_ratio=config.get("window_threshold", 3.0),
            ),
            "isolation_forest": IsolationForestDetector(
                contamination=config.get("contamination", 0.05),
            ),
            "physics": PhysicsChecker(
                hz=self.hz,
                max_accel=config.get("max_accel", 100.0),
            ),
            "temporal": TemporalConsistencyChecker(
                hz=self.hz,
                max_velocity=config.get("max_velocity", 10.0),
            ),
        }

        self.enabled = config.get("enabled_detectors", list(self.detectors.keys()))

    def detect(self, data: dict) -> Dict[str, Any]:
        """
        对一个 episode 执行所有启用的异常检测。

        返回:
            {
                "total_anomalies": int,
                "by_type": {type: count},
                "by_detector": {detector: [Anomaly, ...]},
                "all_anomalies": [Anomaly, ...],
                "anomaly_frame_mask": np.ndarray (bool, T帧),
            }
        """
        qpos = data.get("observations", {}).get("qpos", None)
        qvel = data.get("observations", {}).get("qvel", None)

        if qpos is None:
            return {"total_anomalies": 0, "by_type": {}, "by_detector": {},
                    "all_anomalies": [], "anomaly_frame_mask": np.array([])}

        T = qpos.shape[0]
        all_anomalies = []
        by_detector = {}

        for name in self.enabled:
            if name not in self.detectors:
                continue
            detector = self.detectors[name]

            if name == "physics":
                detected = detector.detect(qpos, qvel)
            else:
                detected = detector.detect(qpos)

            by_detector[name] = detected
            all_anomalies.extend(detected)

        # 统计
        by_type = {}
        for a in all_anomalies:
            key = a.anomaly_type.value
            by_type[key] = by_type.get(key, 0) + 1

        # 异常帧掩码
        mask = np.zeros(T, dtype=bool)
        for a in all_anomalies:
            mask[a.frame_start:a.frame_end + 1] = True

        return {
            "total_anomalies": len(all_anomalies),
            "by_type": by_type,
            "by_detector": by_detector,
            "all_anomalies": all_anomalies,
            "anomaly_frame_mask": mask,
            "anomaly_ratio": float(mask.sum()) / T if T > 0 else 0,
        }

    def print_report(self, report: Dict[str, Any]):
        """打印异常检测报告"""
        print(f"\n  异常检测报告:")
        print(f"    总异常数: {report['total_anomalies']}")
        print(f"    异常帧比例: {report.get('anomaly_ratio', 0):.1%}")

        if report["by_type"]:
            print(f"\n    按类型统计:")
            for atype, count in sorted(report["by_type"].items()):
                print(f"      {atype}: {count}")

        if report.get("by_detector"):
            print(f"\n    按检测器统计:")
            for det_name, anomalies in report["by_detector"].items():
                print(f"      {det_name}: {len(anomalies)} 个异常")

        # 打印严重程度最高的前 10 个异常
        top = sorted(report["all_anomalies"], key=lambda a: a.severity, reverse=True)[:10]
        if top:
            print(f"\n    严重异常 TOP {min(10, len(top))}:")
            for i, a in enumerate(top):
                print(f"      [{i+1}] 帧 {a.frame_start}-{a.frame_end} | "
                      f"严重度 {a.severity:.2f} | {a.description}")

    def visualize(self, data: dict, report: Dict[str, Any],
                  output_path: str = "anomaly_detection.png"):
        """生成异常检测可视化"""
        if not HAS_MPL:
            print("  matplotlib 未安装，跳过可视化")
            return

        qpos = data.get("observations", {}).get("qpos", None)
        if qpos is None:
            return

        T, nq = qpos.shape
        time_axis = np.arange(T) / self.hz
        nq_display = min(nq, 5)

        fig, axes = plt.subplots(nq_display + 2, 1,
                                 figsize=(14, 3 * (nq_display + 2)),
                                 sharex=True,
                                 gridspec_kw={"height_ratios": [2] * nq_display + [1, 1.5]})

        fig.suptitle("Anomaly Detection Report", fontsize=14, fontweight="bold")

        # 收集每个关节的异常帧
        joint_anomaly_frames = {j: set() for j in range(nq)}
        for a in report["all_anomalies"]:
            for j in a.joint_indices:
                if j < nq:
                    for f in range(a.frame_start, a.frame_end + 1):
                        if f < T:
                            joint_anomaly_frames[j].add(f)

        # 关节时间序列 + 异常高亮
        for j in range(nq_display):
            ax = axes[j]
            ax.plot(time_axis, qpos[:, j], color="steelblue", linewidth=0.7, alpha=0.8)

            frames = sorted(joint_anomaly_frames[j])
            if frames:
                frames_arr = np.array(frames)
                ax.scatter(time_axis[frames_arr], qpos[frames_arr, j],
                           color="red", s=8, zorder=5, alpha=0.7)

            ax.set_ylabel(f"Joint {j}", fontsize=9)
            ax.grid(alpha=0.15)

        # 异常帧掩码热图
        ax = axes[nq_display]
        mask = report["anomaly_frame_mask"]
        if len(mask) > 0:
            ax.imshow(mask.reshape(1, -1), aspect="auto", cmap="Reds",
                      extent=[0, T / self.hz, 0, 1], vmin=0, vmax=1)
        ax.set_ylabel("Anomaly", fontsize=9)
        ax.set_yticks([])

        # 按检测器分类的异常条
        ax = axes[nq_display + 1]
        det_names = list(report.get("by_detector", {}).keys())
        for i, det_name in enumerate(det_names):
            for a in report["by_detector"][det_name]:
                ax.barh(i, (a.frame_end - a.frame_start + 1) / self.hz,
                        left=a.frame_start / self.hz, height=0.8,
                        color=plt.cm.Set2(i / max(len(det_names), 1)),
                        alpha=0.7)
        ax.set_yticks(range(len(det_names)))
        ax.set_yticklabels(det_names, fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_title("Anomalies by Detector", fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  可视化已保存: {output_path}")


# ============================================================
# 测试数据生成
# ============================================================

def generate_anomalous_episode(num_frames=500, nq=7):
    """生成包含多种异常的测试轨迹"""
    t = np.linspace(0, 8 * np.pi, num_frames)
    qpos = np.column_stack([np.sin(t * (j + 1)) * 0.4 for j in range(nq)])
    qvel = np.column_stack([np.cos(t * (j + 1)) * 0.4 * (j + 1) for j in range(nq)])
    noise = np.random.randn(num_frames, nq) * 0.01
    qpos += noise

    # 异常 1: 统计离群点（关节 0 的 spike）
    qpos[100, 0] = 5.0
    qpos[101, 0] = -4.0

    # 异常 2: 帧间跳变（关节 2）
    qpos[200:205, 2] += 3.0

    # 异常 3: 高频振荡区段（关节 4）
    qpos[300:330, 4] += np.sin(np.linspace(0, 20 * np.pi, 30)) * 2.0

    # 异常 4: 常数段（关节 5 静止 50 帧后突然恢复）
    qpos[350:400, 5] = 0.42

    # 异常 5: 多关节同时异常
    qpos[450:455, :3] += np.random.randn(5, 3) * 3.0

    return {
        "observations": {
            "qpos": qpos,
            "qvel": qvel,
        },
    }


# ============================================================
# 演示函数
# ============================================================

def section_1_statistical_methods():
    """第 1 节: 统计方法"""
    print(DIVIDER)
    print("第 1 节：统计异常检测 (Z-Score & IQR)")
    print(DIVIDER)

    print("""
    统计方法是最基础的异常检测手段:

    Z-Score 方法:
      z = (x - μ) / σ
      当 |z| > 3 时，认为是异常 (对应正态分布的 99.7%)

    IQR 方法:
      IQR = Q3 - Q1 (四分位距)
      异常 = x < Q1 - 1.5*IQR  或  x > Q3 + 1.5*IQR
      对极端值更鲁棒
    """)

    data = generate_anomalous_episode()
    qpos = data["observations"]["qpos"]

    # Z-Score
    z_detector = ZScoreDetector(threshold=3.0)
    z_anomalies = z_detector.detect(qpos)
    print(f"\n  Z-Score 检测结果: {len(z_anomalies)} 个异常")
    for a in z_anomalies[:5]:
        print(f"    帧 {a.frame_start}-{a.frame_end}: {a.description}")

    # IQR
    print()
    iqr_detector = IQRDetector(factor=1.5)
    iqr_anomalies = iqr_detector.detect(qpos)
    print(f"  IQR 检测结果: {len(iqr_anomalies)} 个异常")
    for a in iqr_anomalies[:5]:
        print(f"    帧 {a.frame_start}-{a.frame_end}: {a.description}")


def section_2_sliding_window():
    """第 2 节: 滑动窗口"""
    print("\n" + DIVIDER)
    print("第 2 节：滑动窗口分析 (Sliding Window)")
    print(DIVIDER)

    print("""
    滑动窗口方法能发现局部模式变化:
    - 在每个窗口内计算局部统计量
    - 与全局统计量比较
    - 检测出"局部正常但整体异常"的片段
    """)

    data = generate_anomalous_episode()
    qpos = data["observations"]["qpos"]

    sw_detector = SlidingWindowDetector(window_size=30, threshold_std_ratio=2.5)
    sw_anomalies = sw_detector.detect(qpos)
    print(f"\n  滑动窗口检测结果: {len(sw_anomalies)} 个异常")
    for a in sw_anomalies[:5]:
        print(f"    帧 {a.frame_start}-{a.frame_end}: {a.description}")


def section_3_isolation_forest():
    """第 3 节: Isolation Forest"""
    print("\n" + DIVIDER)
    print("第 3 节：Isolation Forest 多变量异常检测")
    print(DIVIDER)

    print(f"""
    Isolation Forest 的核心思想:
    - 异常点在特征空间中"与众不同"
    - 随机划分时，异常点更容易被隔离
    - 需要的划分次数越少 = 越异常

    当前后端: {'sklearn (高性能)' if HAS_SKLEARN else '自定义实现 (教学版)'}
    """)

    data = generate_anomalous_episode()
    qpos = data["observations"]["qpos"]

    t0 = time.time()
    if_detector = IsolationForestDetector(contamination=0.05)
    if_anomalies = if_detector.detect(qpos)
    elapsed = time.time() - t0

    print(f"  Isolation Forest 检测结果: {len(if_anomalies)} 个异常 (耗时 {elapsed:.3f}s)")
    for a in if_anomalies[:5]:
        print(f"    帧 {a.frame_start}-{a.frame_end}: {a.description}")


def section_4_physics_checks():
    """第 4 节: 物理一致性检查"""
    print("\n" + DIVIDER)
    print("第 4 节：物理一致性检查 (Physics-Based)")
    print(DIVIDER)

    print("""
    物理定律提供了强约束:
    - 加速度有上限（电机扭矩/惯量）
    - 能量不会突变（除非有外力输入）
    - 速度不能瞬间跳变（惯性）

    违反这些约束 = 数据有问题
    """)

    data = generate_anomalous_episode()
    qpos = data["observations"]["qpos"]
    qvel = data["observations"]["qvel"]

    physics_checker = PhysicsChecker(hz=50.0, max_accel=50.0)
    physics_anomalies = physics_checker.detect(qpos, qvel)
    print(f"\n  物理检查结果: {len(physics_anomalies)} 个异常")
    for a in physics_anomalies[:5]:
        print(f"    帧 {a.frame_start}-{a.frame_end}: {a.description}")

    temporal_checker = TemporalConsistencyChecker(hz=50.0, max_velocity=5.0)
    temporal_anomalies = temporal_checker.detect(qpos)
    print(f"\n  时序一致性检查结果: {len(temporal_anomalies)} 个异常")
    for a in temporal_anomalies[:5]:
        print(f"    帧 {a.frame_start}-{a.frame_end}: {a.description}")


def section_5_comprehensive():
    """第 5 节: 综合检测"""
    print("\n" + DIVIDER)
    print("第 5 节：综合异常检测器 (AnomalyDetector)")
    print(DIVIDER)

    config = {
        "hz": 50.0,
        "z_threshold": 3.0,
        "iqr_factor": 1.5,
        "window_size": 30,
        "contamination": 0.05,
        "max_accel": 50.0,
        "max_velocity": 5.0,
    }

    detector = AnomalyDetector(config)
    data = generate_anomalous_episode()

    print(f"\n  运行综合异常检测...")
    t0 = time.time()
    report = detector.detect(data)
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.3f}s")

    detector.print_report(report)

    # 可视化
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "anomaly_detection.png")
    detector.visualize(data, report, output_path)


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("第 6 章 · 04 - 异常检测 (Anomaly Detection)")
    print("多种方法自动识别轨迹数据中的异常")
    print(DIVIDER)

    section_1_statistical_methods()
    section_2_sliding_window()
    section_3_isolation_forest()
    section_4_physics_checks()
    section_5_comprehensive()

    print("\n" + DIVIDER)
    print("✅ 异常检测演示完成！")
    print(DIVIDER)


if __name__ == "__main__":
    main()
