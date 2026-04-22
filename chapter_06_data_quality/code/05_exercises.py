"""
第 6 章 · 05 - 练习 (Exercises)

目标: 通过动手实践，巩固数据质量与校验的知识。

练习内容:
  1. 编写自定义校验规则
  2. 构建数据清洗流水线（检测 + 修复异常）
  3. 创建质量评分指标 (0-100)
  4. 构建批量质量报告

运行: python 05_exercises.py
依赖: pip install numpy matplotlib
"""

import numpy as np
import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
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


# ============================================================
# 共用工具
# ============================================================

class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    severity: Severity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
        }


def generate_episode(num_frames=300, nq=7, nu=7, inject_anomalies=False):
    """生成测试用 episode"""
    t = np.linspace(0, 4 * np.pi, num_frames)

    qpos = np.column_stack([
        np.sin(t * (j + 1)) * (0.3 + np.random.rand() * 0.2)
        + np.random.randn(num_frames) * 0.01
        for j in range(nq)
    ])
    qvel = np.diff(qpos, axis=0, prepend=qpos[:1]) * 50
    action = np.column_stack([
        np.sin(t * (j + 1) + 0.1) * 0.4 for j in range(nu)
    ])
    timestamps = np.arange(num_frames) / 50.0

    data = {
        "observations": {"qpos": qpos, "qvel": qvel},
        "action": action,
        "timestamps": timestamps,
    }

    if inject_anomalies:
        anomaly_type = np.random.choice(["nan", "spike", "jump", "dead", "ts"])

        if anomaly_type == "nan":
            idx = np.random.randint(10, num_frames - 10)
            qpos[idx, np.random.randint(nq)] = np.nan

        elif anomaly_type == "spike":
            idx = np.random.randint(10, num_frames - 10)
            joint = np.random.randint(nq)
            qpos[idx, joint] = qpos[idx, joint] + np.random.choice([-1, 1]) * 5.0

        elif anomaly_type == "jump":
            idx = np.random.randint(10, num_frames - 10)
            qpos[idx:, :] += np.random.randn(nq) * 2.0

        elif anomaly_type == "dead":
            joint = np.random.randint(nq)
            qpos[:, joint] = 0.42

        elif anomaly_type == "ts":
            idx = np.random.randint(5, num_frames - 5)
            timestamps[idx] = timestamps[idx - 1] - 0.01

        data["observations"]["qpos"] = qpos
        data["timestamps"] = timestamps

    return data


# ============================================================
# 练习 1: 自定义校验规则
# ============================================================

def exercise_1_custom_validation_rule():
    """
    练习 1: 编写自定义校验规则

    任务: 实现一个灵活的规则引擎，允许用户用简单的 Python 函数定义校验规则，
         然后批量执行。

    学习要点:
    - 校验规则的抽象（将规则表示为可调用对象）
    - 规则引擎的设计模式
    - 如何让非程序员也能定义简单规则
    """
    print(DIVIDER)
    print("练习 1: 自定义校验规则引擎")
    print(DIVIDER)

    # ---- 规则引擎实现 ----

    class ValidationRule:
        """
        一条校验规则。

        用法:
            rule = ValidationRule(
                name="no_nan",
                severity=Severity.ERROR,
                check_fn=lambda data: not np.isnan(data["observations"]["qpos"]).any(),
                description="qpos 中不应有 NaN"
            )
        """
        def __init__(self, name: str, check_fn: Callable[[dict], bool],
                     severity: Severity = Severity.ERROR,
                     description: str = ""):
            self.name = name
            self.check_fn = check_fn
            self.severity = severity
            self.description = description

        def execute(self, data: dict) -> ValidationResult:
            try:
                passed = self.check_fn(data)
                return ValidationResult(
                    check_name=self.name,
                    passed=bool(passed),
                    severity=self.severity,
                    message=self.description if not passed else f"{self.name}: 通过",
                )
            except Exception as e:
                return ValidationResult(
                    check_name=self.name,
                    passed=False,
                    severity=Severity.ERROR,
                    message=f"规则执行异常: {str(e)}",
                    details={"error": str(e)},
                )

    class RuleEngine:
        """
        校验规则引擎。

        可以注册多条规则，然后对数据批量执行。
        """
        def __init__(self):
            self.rules: List[ValidationRule] = []

        def add_rule(self, rule: ValidationRule):
            self.rules.append(rule)

        def add_lambda_rule(self, name: str, check_fn: Callable,
                            severity: Severity = Severity.ERROR, description: str = ""):
            """快捷方法: 直接用 lambda 添加规则"""
            self.rules.append(ValidationRule(name, check_fn, severity, description))

        def validate(self, data: dict) -> List[ValidationResult]:
            return [rule.execute(data) for rule in self.rules]

        def print_results(self, results: List[ValidationResult]):
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            print(f"\n  校验结果: {passed}/{total} 通过")
            for r in results:
                icon = "✅" if r.passed else "❌"
                print(f"    {icon} [{r.severity.value}] {r.check_name}: {r.message}")

    # ---- 定义规则 ----
    engine = RuleEngine()

    # 规则 1: 无 NaN
    engine.add_lambda_rule(
        name="no_nan",
        check_fn=lambda d: not np.isnan(d["observations"]["qpos"]).any(),
        severity=Severity.ERROR,
        description="qpos 中发现 NaN 值",
    )

    # 规则 2: 无 Inf
    engine.add_lambda_rule(
        name="no_inf",
        check_fn=lambda d: not np.isinf(d["observations"]["qpos"]).any(),
        severity=Severity.ERROR,
        description="qpos 中发现 Inf 值",
    )

    # 规则 3: 轨迹长度检查
    engine.add_lambda_rule(
        name="min_length",
        check_fn=lambda d: d["observations"]["qpos"].shape[0] >= 50,
        severity=Severity.ERROR,
        description="轨迹长度不足 50 帧",
    )

    # 规则 4: 关节范围检查
    engine.add_lambda_rule(
        name="joint_range",
        check_fn=lambda d: np.all(np.abs(d["observations"]["qpos"]) < np.pi * 1.1),
        severity=Severity.WARNING,
        description="关节角度超出 [-π*1.1, π*1.1] 范围",
    )

    # 规则 5: 无死关节
    engine.add_lambda_rule(
        name="no_dead_joints",
        check_fn=lambda d: np.all(d["observations"]["qpos"].std(axis=0) > 1e-6),
        severity=Severity.WARNING,
        description="检测到死关节 (标准差 < 1e-6)",
    )

    # 规则 6: 时间戳单调递增
    engine.add_lambda_rule(
        name="monotonic_timestamps",
        check_fn=lambda d: np.all(np.diff(d.get("timestamps", [0, 1])) > 0),
        severity=Severity.ERROR,
        description="时间戳非单调递增",
    )

    # 规则 7: 帧间连续性
    engine.add_lambda_rule(
        name="continuity",
        check_fn=lambda d: np.max(np.abs(np.diff(d["observations"]["qpos"], axis=0))) < 2.0,
        severity=Severity.WARNING,
        description="帧间跳变超过 2.0",
    )

    # ---- 测试 ----
    print("\n  [测试 1] 正常数据:")
    good_data = generate_episode(inject_anomalies=False)
    results = engine.validate(good_data)
    engine.print_results(results)

    print(f"\n  [测试 2] 异常数据 (NaN):")
    bad_data = generate_episode(inject_anomalies=False)
    bad_data["observations"]["qpos"][50, 2] = np.nan
    results = engine.validate(bad_data)
    engine.print_results(results)

    print(f"\n  [测试 3] 异常数据 (死关节 + 跳变):")
    bad_data2 = generate_episode(inject_anomalies=False)
    bad_data2["observations"]["qpos"][:, 5] = 0.42
    bad_data2["observations"]["qpos"][100, 3] += 5.0
    results = engine.validate(bad_data2)
    engine.print_results(results)

    print(f"\n  💡 设计理念: 用 lambda 函数定义规则，让规则定义尽量简洁。")
    print(f"     在企业环境中，规则可以从配置文件（YAML/JSON）加载。")


# ============================================================
# 练习 2: 数据清洗流水线
# ============================================================

def exercise_2_data_cleaning_pipeline():
    """
    练习 2: 构建数据清洗流水线

    任务: 先检测异常，再自动修复。
    修复策略:
    - NaN/Inf: 线性插值填充
    - Spike (突刺): 用相邻帧的均值替代
    - 帧间跳变: 平滑过渡
    - 死关节: 标记但不修复（无法恢复丢失的信息）
    """
    print("\n" + DIVIDER)
    print("练习 2: 数据清洗流水线")
    print(DIVIDER)

    class DataCleaner:
        """
        数据清洗器。

        执行"检测 → 修复"流程，并记录所有修改操作。
        """

        def __init__(self, hz: float = 50.0):
            self.hz = hz
            self.log: List[str] = []

        def _log(self, msg: str):
            self.log.append(msg)

        def fix_nan_inf(self, qpos: np.ndarray) -> np.ndarray:
            """用线性插值替换 NaN/Inf"""
            result = qpos.copy()
            T, nq = result.shape

            for j in range(nq):
                col = result[:, j]
                bad = np.isnan(col) | np.isinf(col)
                bad_count = bad.sum()
                if bad_count == 0:
                    continue

                self._log(f"  关节 {j}: 修复 {bad_count} 个 NaN/Inf (线性插值)")

                good_idx = np.where(~bad)[0]
                bad_idx = np.where(bad)[0]

                if len(good_idx) >= 2:
                    result[bad_idx, j] = np.interp(bad_idx, good_idx, col[good_idx])
                elif len(good_idx) == 1:
                    result[bad_idx, j] = col[good_idx[0]]
                else:
                    result[bad_idx, j] = 0.0

            return result

        def fix_spikes(self, qpos: np.ndarray, z_threshold: float = 4.0) -> np.ndarray:
            """
            修复突刺: 如果某帧的 z-score 超阈值，用相邻帧均值替代。
            """
            result = qpos.copy()
            T, nq = result.shape

            for j in range(nq):
                col = result[:, j]
                mu, sigma = col.mean(), col.std()
                if sigma < 1e-10:
                    continue

                z = np.abs((col - mu) / sigma)
                spikes = np.where(z > z_threshold)[0]

                if len(spikes) > 0:
                    self._log(f"  关节 {j}: 修复 {len(spikes)} 个突刺 (邻帧均值)")
                    for idx in spikes:
                        neighbors = []
                        if idx > 0:
                            neighbors.append(result[idx - 1, j])
                        if idx < T - 1:
                            neighbors.append(result[idx + 1, j])
                        if neighbors:
                            result[idx, j] = np.mean(neighbors)

            return result

        def fix_jumps(self, qpos: np.ndarray, max_jump: float = 1.0,
                      smooth_window: int = 5) -> np.ndarray:
            """
            修复帧间跳变: 在跳变处用移动平均平滑过渡。
            """
            result = qpos.copy()
            T, nq = result.shape

            for j in range(nq):
                diffs = np.abs(np.diff(result[:, j]))
                jumps = np.where(diffs > max_jump)[0]

                if len(jumps) > 0:
                    self._log(f"  关节 {j}: 修复 {len(jumps)} 处跳变 (移动平均)")
                    for jump_idx in jumps:
                        start = max(0, jump_idx - smooth_window)
                        end = min(T, jump_idx + smooth_window + 1)
                        window = result[start:end, j]
                        smoothed = np.convolve(window, np.ones(3) / 3, mode="same")
                        result[start:end, j] = smoothed

            return result

        def detect_dead_joints(self, qpos: np.ndarray,
                               threshold: float = 1e-6) -> List[int]:
            """标记死关节（不修复，因为无法恢复信息）"""
            stds = qpos.std(axis=0)
            dead = np.where(stds < threshold)[0].tolist()
            if dead:
                self._log(f"  死关节 (无法修复): {dead}")
            return dead

        def clean(self, data: dict) -> dict:
            """
            执行完整清洗流程。

            返回清洗后的数据（副本，不修改原始数据）。
            """
            self.log = []
            result = {
                "observations": {
                    "qpos": data["observations"]["qpos"].copy(),
                    "qvel": data["observations"].get("qvel", np.array([])).copy()
                         if "qvel" in data.get("observations", {}) else None,
                },
                "action": data.get("action", np.array([])).copy()
                         if "action" in data else None,
                "timestamps": data.get("timestamps", np.array([])).copy()
                             if "timestamps" in data else None,
            }

            qpos = result["observations"]["qpos"]

            # 步骤 1: 修复 NaN/Inf
            self._log("步骤 1: 修复 NaN/Inf")
            qpos = self.fix_nan_inf(qpos)

            # 步骤 2: 修复突刺
            self._log("步骤 2: 修复突刺")
            qpos = self.fix_spikes(qpos)

            # 步骤 3: 修复跳变
            self._log("步骤 3: 修复帧间跳变")
            qpos = self.fix_jumps(qpos)

            # 步骤 4: 标记死关节
            self._log("步骤 4: 标记死关节")
            self.detect_dead_joints(qpos)

            result["observations"]["qpos"] = qpos
            return result

        def print_log(self):
            print("\n  清洗日志:")
            for entry in self.log:
                print(f"    {entry}")

    # ---- 演示 ----
    print("""
    清洗流水线的核心原则:
    1. 不修改原始数据（返回副本）
    2. 记录所有修改操作（可追溯）
    3. 保守修复：只修复有把握的问题
    4. 不可恢复的问题只标记不修复
    """)

    # 生成有问题的数据
    data = generate_episode(num_frames=300)
    qpos = data["observations"]["qpos"]
    qpos[50, 2] = np.nan
    qpos[51, 2] = np.inf
    qpos[100, 0] = 8.0   # spike
    qpos[200:, 3] += 3.0  # jump
    qpos[:, 6] = 0.42     # dead joint

    print("  原始数据问题:")
    print(f"    NaN 数量: {np.isnan(qpos).sum()}")
    print(f"    Inf 数量: {np.isinf(qpos).sum()}")
    print(f"    最大值: {np.nanmax(np.where(np.isinf(qpos), np.nan, qpos)):.2f}")
    print(f"    死关节: {np.where(qpos.std(axis=0) < 1e-6)[0].tolist()}")

    cleaner = DataCleaner(hz=50)
    cleaned = cleaner.clean(data)
    cleaner.print_log()

    cleaned_qpos = cleaned["observations"]["qpos"]
    print(f"\n  清洗后:")
    print(f"    NaN 数量: {np.isnan(cleaned_qpos).sum()}")
    print(f"    Inf 数量: {np.isinf(cleaned_qpos).sum()}")
    print(f"    最大值: {cleaned_qpos.max():.2f}")
    print(f"    死关节: {np.where(cleaned_qpos.std(axis=0) < 1e-6)[0].tolist()}")

    # 可视化对比
    if HAS_MPL:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        t = np.arange(300) / 50.0
        for j in range(min(7, qpos.shape[1])):
            original = np.where(np.isinf(qpos[:, j]), np.nan, qpos[:, j])
            axes[0].plot(t, original, alpha=0.7, label=f"J{j}")
            axes[1].plot(t, cleaned_qpos[:, j], alpha=0.7, label=f"J{j}")
        axes[0].set_title("Before Cleaning")
        axes[0].legend(fontsize=7, ncol=7)
        axes[0].grid(alpha=0.2)
        axes[1].set_title("After Cleaning")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(fontsize=7, ncol=7)
        axes[1].grid(alpha=0.2)
        plt.tight_layout()
        output_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(output_dir, "cleaning_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  对比图已保存: {path}")


# ============================================================
# 练习 3: 质量评分指标
# ============================================================

def exercise_3_quality_score():
    """
    练习 3: 创建质量评分指标 (0-100)

    任务: 设计一个综合质量评分系统，将多个维度的检查结果映射到 0-100 分。

    评分维度:
    - 完整性 (20分): 无 NaN/Inf，无缺失帧
    - 范围合规 (20分): 关节值在合理范围内
    - 平滑度 (20分): 运动平滑，无异常跳变
    - 关节活性 (20分): 所有关节都有有效运动
    - 时序一致 (20分): 时间戳正确，速度合理
    """
    print("\n" + DIVIDER)
    print("练习 3: 质量评分指标 (0-100)")
    print(DIVIDER)

    class QualityScorer:
        """
        数据质量评分器。

        将多个维度的检查结果映射到 0-100 的总分。
        每个维度 20 分，总分 = 各维度得分之和。
        """

        def __init__(self, joint_limits_lower=None, joint_limits_upper=None,
                     hz: float = 50.0):
            self.joint_lower = np.asarray(joint_limits_lower) if joint_limits_lower is not None else None
            self.joint_upper = np.asarray(joint_limits_upper) if joint_limits_upper is not None else None
            self.hz = hz

        def score_completeness(self, data: dict) -> Tuple[float, str]:
            """
            完整性评分 (满分 20):
            - 无 NaN: +10 分，按比例扣分
            - 无 Inf: +5 分
            - 数据不为空: +5 分
            """
            qpos = data.get("observations", {}).get("qpos", None)
            if qpos is None:
                return 0.0, "无 qpos 数据"

            total = qpos.size
            nan_ratio = np.isnan(qpos).sum() / max(total, 1)
            inf_ratio = np.isinf(qpos).sum() / max(total, 1)

            score = 0.0
            score += max(0, 10 * (1 - nan_ratio * 100))  # NaN 扣分
            score += 5.0 if inf_ratio == 0 else max(0, 5 * (1 - inf_ratio * 100))
            score += 5.0 if total > 0 else 0.0

            msg = f"NaN率={nan_ratio:.4%}, Inf率={inf_ratio:.4%}"
            return min(score, 20.0), msg

        def score_range_compliance(self, data: dict) -> Tuple[float, str]:
            """
            范围合规评分 (满分 20):
            - 超出限位的比例越低，得分越高
            """
            qpos = data.get("observations", {}).get("qpos", None)
            if qpos is None or self.joint_lower is None:
                return 20.0, "未配置限位或无数据"

            nq = min(qpos.shape[1], len(self.joint_lower))
            violations = 0
            total_checks = qpos.shape[0] * nq

            for j in range(nq):
                violations += (qpos[:, j] < self.joint_lower[j]).sum()
                violations += (qpos[:, j] > self.joint_upper[j]).sum()

            violation_ratio = violations / max(total_checks, 1)
            score = max(0, 20 * (1 - violation_ratio * 10))

            return min(score, 20.0), f"违规比例={violation_ratio:.4%}"

        def score_smoothness(self, data: dict) -> Tuple[float, str]:
            """
            平滑度评分 (满分 20):
            - 基于帧间最大跳变
            - 跳变越小，得分越高
            """
            qpos = data.get("observations", {}).get("qpos", None)
            if qpos is None or qpos.shape[0] < 2:
                return 0.0, "数据不足"

            # 排除 NaN
            clean = np.nan_to_num(qpos, nan=0.0, posinf=0.0, neginf=0.0)
            diffs = np.abs(np.diff(clean, axis=0))
            max_jump = diffs.max()
            mean_jump = diffs.mean()

            # 超大跳变扣分
            if max_jump > 5.0:
                score = 5.0
            elif max_jump > 2.0:
                score = 10.0
            elif max_jump > 1.0:
                score = 15.0
            else:
                score = 20.0

            # 均值跳变微调
            if mean_jump > 0.5:
                score -= 3.0
            elif mean_jump > 0.2:
                score -= 1.0

            return max(0, min(score, 20.0)), f"最大跳变={max_jump:.3f}, 平均={mean_jump:.4f}"

        def score_joint_activity(self, data: dict) -> Tuple[float, str]:
            """
            关节活性评分 (满分 20):
            - 每个关节的标准差 > 阈值 = "活跃"
            - 活跃关节越多，得分越高
            """
            qpos = data.get("observations", {}).get("qpos", None)
            if qpos is None:
                return 0.0, "无数据"

            nq = qpos.shape[1]
            stds = np.nan_to_num(qpos, nan=0.0).std(axis=0)
            active = (stds > 1e-4).sum()
            ratio = active / max(nq, 1)

            score = 20 * ratio
            return min(score, 20.0), f"活跃关节 {active}/{nq}"

        def score_temporal_consistency(self, data: dict) -> Tuple[float, str]:
            """
            时序一致性评分 (满分 20):
            - 时间戳单调递增: +10 分
            - 速度在合理范围: +10 分
            """
            score = 0.0
            msgs = []

            timestamps = data.get("timestamps", None)
            if timestamps is not None:
                timestamps = np.asarray(timestamps)
                if len(timestamps) >= 2:
                    monotonic = np.all(np.diff(timestamps) > 0)
                    score += 10.0 if monotonic else 0.0
                    msgs.append(f"单调性={'✓' if monotonic else '✗'}")
                else:
                    score += 10.0
            else:
                score += 10.0
                msgs.append("无时间戳")

            qpos = data.get("observations", {}).get("qpos", None)
            if qpos is not None and qpos.shape[0] >= 2:
                velocity = np.abs(np.diff(np.nan_to_num(qpos), axis=0)) * self.hz
                max_v = velocity.max()
                if max_v < 10:
                    score += 10.0
                elif max_v < 50:
                    score += 5.0
                msgs.append(f"最大速度={max_v:.1f}")

            return min(score, 20.0), ", ".join(msgs)

        def score(self, data: dict) -> Dict[str, Any]:
            """计算综合质量评分"""
            dimensions = {}

            for name, scorer_fn in [
                ("completeness", self.score_completeness),
                ("range_compliance", self.score_range_compliance),
                ("smoothness", self.score_smoothness),
                ("joint_activity", self.score_joint_activity),
                ("temporal_consistency", self.score_temporal_consistency),
            ]:
                s, msg = scorer_fn(data)
                dimensions[name] = {"score": round(s, 1), "max": 20, "detail": msg}

            total = sum(d["score"] for d in dimensions.values())

            return {
                "total_score": round(total, 1),
                "max_score": 100,
                "grade": self._grade(total),
                "dimensions": dimensions,
            }

        @staticmethod
        def _grade(score: float) -> str:
            if score >= 90: return "A"
            if score >= 80: return "B"
            if score >= 70: return "C"
            if score >= 60: return "D"
            return "F"

    # ---- 演示 ----
    scorer = QualityScorer(
        joint_limits_lower=[-np.pi] * 7,
        joint_limits_upper=[np.pi] * 7,
    )

    print("\n  === 评分测试 ===")

    # 好数据
    good = generate_episode(num_frames=300, inject_anomalies=False)
    result = scorer.score(good)
    print(f"\n  [正常数据] 总分: {result['total_score']}/100 ({result['grade']})")
    for dim, info in result["dimensions"].items():
        bar = "█" * int(info["score"]) + "░" * (20 - int(info["score"]))
        print(f"    {dim:25s} [{bar}] {info['score']:5.1f}/{info['max']}  {info['detail']}")

    # 中等数据
    medium = generate_episode(num_frames=300, inject_anomalies=False)
    medium["observations"]["qpos"][:, 6] = 0.42
    medium["observations"]["qpos"][100, 0] += 3.0
    result = scorer.score(medium)
    print(f"\n  [中等数据] 总分: {result['total_score']}/100 ({result['grade']})")
    for dim, info in result["dimensions"].items():
        bar = "█" * int(info["score"]) + "░" * (20 - int(info["score"]))
        print(f"    {dim:25s} [{bar}] {info['score']:5.1f}/{info['max']}  {info['detail']}")

    # 差数据
    bad = generate_episode(num_frames=300, inject_anomalies=False)
    bad["observations"]["qpos"][50, 2] = np.nan
    bad["observations"]["qpos"][100:, 0] += 5.0
    bad["observations"]["qpos"][:, 5] = 0.42
    bad["observations"]["qpos"][:, 6] = 0.42
    bad["timestamps"][80] = bad["timestamps"][79] - 0.1
    result = scorer.score(bad)
    print(f"\n  [差数据]   总分: {result['total_score']}/100 ({result['grade']})")
    for dim, info in result["dimensions"].items():
        bar = "█" * int(info["score"]) + "░" * (20 - int(info["score"]))
        print(f"    {dim:25s} [{bar}] {info['score']:5.1f}/{info['max']}  {info['detail']}")


# ============================================================
# 练习 4: 批量质量报告
# ============================================================

def exercise_4_batch_quality_report():
    """
    练习 4: 构建批量质量报告

    任务: 对一个模拟数据集的所有 episode 进行质量评估，
         生成汇总报告（JSON + 可视化）。
    """
    print("\n" + DIVIDER)
    print("练习 4: 批量质量报告")
    print(DIVIDER)

    class QualityScorer:
        def __init__(self):
            pass

        def score(self, data: dict) -> dict:
            qpos = data.get("observations", {}).get("qpos", None)
            if qpos is None:
                return {"total_score": 0, "grade": "F"}

            score = 100.0

            # NaN/Inf 扣分
            nan_count = np.isnan(qpos).sum() + np.isinf(qpos).sum()
            if nan_count > 0:
                score -= min(30, nan_count * 5)

            # 死关节扣分
            dead = (qpos.std(axis=0) < 1e-6).sum()
            score -= dead * 5

            # 跳变扣分
            diffs = np.abs(np.diff(np.nan_to_num(qpos), axis=0))
            large_jumps = (diffs.max(axis=1) > 1.0).sum()
            score -= min(20, large_jumps * 2)

            # 范围扣分
            out_of_range = (np.abs(np.nan_to_num(qpos)) > np.pi).sum()
            if out_of_range > 0:
                score -= min(20, out_of_range)

            score = max(0, min(100, score))
            grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
            return {"total_score": round(score, 1), "grade": grade}

    # 生成模拟数据集
    np.random.seed(42)
    num_episodes = 50
    episodes = []
    for i in range(num_episodes):
        inject = np.random.rand() < 0.3  # 30% 的 episode 有异常
        data = generate_episode(
            num_frames=np.random.randint(100, 500),
            inject_anomalies=inject,
        )
        episodes.append((f"ep_{i:04d}", data))

    # 批量评分
    scorer = QualityScorer()
    scores = []
    grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}

    print(f"\n  正在评估 {num_episodes} 个 episodes...")
    t0 = time.time()

    results_list = []
    for ep_id, data in episodes:
        result = scorer.score(data)
        result["episode_id"] = ep_id
        results_list.append(result)
        scores.append(result["total_score"])
        grades[result["grade"]] = grades.get(result["grade"], 0) + 1

    elapsed = time.time() - t0
    scores_arr = np.array(scores)

    # 打印汇总
    print(f"  耗时: {elapsed:.3f}s ({elapsed/num_episodes*1000:.1f}ms/episode)")

    print(f"\n  📊 汇总统计:")
    print(f"    平均分: {scores_arr.mean():.1f}")
    print(f"    中位数: {np.median(scores_arr):.1f}")
    print(f"    最高分: {scores_arr.max():.1f}")
    print(f"    最低分: {scores_arr.min():.1f}")
    print(f"    标准差: {scores_arr.std():.1f}")

    print(f"\n  📋 等级分布:")
    for grade in ["A", "B", "C", "D", "F"]:
        count = grades[grade]
        bar = "█" * count + "░" * (num_episodes - count)
        pct = count / num_episodes * 100
        print(f"    {grade}: {count:3d} ({pct:5.1f}%) [{bar}]")

    # 最差的 5 个
    sorted_results = sorted(results_list, key=lambda x: x["total_score"])
    print(f"\n  🔴 质量最差的 5 个 episodes:")
    for r in sorted_results[:5]:
        print(f"    {r['episode_id']}: {r['total_score']:.1f} ({r['grade']})")

    # 导出 JSON
    output_dir = os.path.dirname(os.path.abspath(__file__))
    report = {
        "summary": {
            "total_episodes": num_episodes,
            "mean_score": float(scores_arr.mean()),
            "median_score": float(np.median(scores_arr)),
            "std_score": float(scores_arr.std()),
            "grade_distribution": grades,
        },
        "episodes": results_list,
    }
    json_path = os.path.join(output_dir, "batch_quality_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  报告已导出: {json_path}")

    # 可视化
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Batch Quality Report", fontsize=14, fontweight="bold")

        # 分数分布
        ax = axes[0]
        ax.hist(scores, bins=20, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(scores_arr.mean(), color="red", linestyle="--", label=f"Mean={scores_arr.mean():.1f}")
        ax.set_title("Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(alpha=0.2)

        # 等级饼图
        ax = axes[1]
        grade_counts = [grades.get(g, 0) for g in ["A", "B", "C", "D", "F"]]
        grade_colors = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22", "#e74c3c"]
        non_zero = [(g, c, col) for g, c, col in zip(["A","B","C","D","F"], grade_counts, grade_colors) if c > 0]
        if non_zero:
            labels, counts, cols = zip(*non_zero)
            ax.pie(counts, labels=labels, colors=cols, autopct="%1.0f%%", startangle=90)
        ax.set_title("Grade Distribution")

        # 分数时间线
        ax = axes[2]
        ax.plot(range(len(scores)), scores, "o-", markersize=3, color="steelblue", alpha=0.7)
        ax.axhline(80, color="green", linestyle="--", alpha=0.5, label="B threshold")
        ax.axhline(60, color="orange", linestyle="--", alpha=0.5, label="D threshold")
        ax.set_xlabel("Episode Index")
        ax.set_ylabel("Score")
        ax.set_title("Score Timeline")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        png_path = os.path.join(output_dir, "batch_quality_report.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  可视化已保存: {png_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("第 6 章 · 05 - 练习 (Exercises)")
    print("通过实践巩固数据质量知识")
    print(DIVIDER)

    exercise_1_custom_validation_rule()
    exercise_2_data_cleaning_pipeline()
    exercise_3_quality_score()
    exercise_4_batch_quality_report()

    print("\n" + DIVIDER)
    print("✅ 所有练习完成！")
    print()
    print("  📝 挑战题 (Challenge):")
    print("    1. 给 QualityScorer 添加一个 '数据多样性' 维度")
    print("    2. 实现一个增量式清洗器（流式处理，不需要全量加载）")
    print("    3. 将质量报告集成到数据平台的 CI/CD 流水线中")
    print("    4. 实现一个'数据质量监控仪表盘'（实时更新）")
    print(DIVIDER)


if __name__ == "__main__":
    main()
