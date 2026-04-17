"""
第 6 章 · 01 - 数据校验器 (Data Validator)

目标: 构建企业级数据校验流水线，在数据入库前拦截所有常见问题。

核心知识点:
  1. NaN / Inf 检测 —— 数值合法性的第一道防线
  2. 四元数归一化检查 —— 自由关节 / 球关节的旋转表示
  3. 关节限位违规检测 —— qpos 是否超出物理范围
  4. 帧间跳变检测 —— 相邻帧之间的突变
  5. 轨迹长度校验 —— 太短或太长的轨迹
  6. 死关节检测 —— 始终不变的关节值
  7. 时间戳单调性检查 —— 时间必须递增
  8. 动作范围检查 —— 动作值是否在合理区间
  9. 图像维度校验 —— 图像分辨率是否一致
  10. 批量校验 & 汇总报告

运行: python 01_data_validator.py
依赖: pip install numpy
"""

import numpy as np
import os
import json
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


# ============================================================
# 严重级别枚举
# ============================================================

class Severity(Enum):
    """
    校验结果的严重程度:
      ERROR   - 致命错误，数据不可用，必须修复
      WARNING - 警告，数据可能有问题，建议人工审查
      INFO    - 信息，仅作记录，不影响使用
    """
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


# ============================================================
# 校验结果数据类
# ============================================================

@dataclass
class ValidationResult:
    """
    单条校验结果。

    属性:
        check_name: 校验项名称（如 "nan_inf_check"）
        passed: 是否通过
        severity: 严重级别
        message: 人类可读的说明
        details: 具体细节（如哪些帧、哪些关节出了问题）
    """
    check_name: str
    passed: bool
    severity: Severity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
        }


# ============================================================
# 校验报告
# ============================================================

@dataclass
class ValidationReport:
    """
    一个 episode 的完整校验报告。

    包含多条 ValidationResult，并提供汇总方法。
    """
    episode_id: str
    results: List[ValidationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: ValidationResult):
        self.results.append(result)

    @property
    def passed(self) -> bool:
        """只有所有 ERROR 级别都通过，才算整体通过"""
        return all(
            r.passed for r in self.results if r.severity == Severity.ERROR
        )

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == Severity.WARNING)

    def summary(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "overall_passed": self.passed,
            "total_checks": len(self.results),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "checks": [r.to_dict() for r in self.results],
        }

    def print_summary(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        print(f"\n  Episode: {self.episode_id}  [{status}]")
        print(f"  总检查数: {len(self.results)} | 错误: {self.error_count} | 警告: {self.warning_count}")
        for r in self.results:
            icon = "✅" if r.passed else ("❌" if r.severity == Severity.ERROR else "⚠️")
            print(f"    {icon} [{r.severity.value:7s}] {r.check_name}: {r.message}")


# ============================================================
# 数据校验器
# ============================================================

class DataValidator:
    """
    企业级数据校验器。

    支持对单个 episode（dict 格式）执行一系列校验项，
    生成详细的校验报告。

    使用方法:
        validator = DataValidator(config)
        report = validator.validate(episode_data, episode_id="ep_001")
        report.print_summary()

    配置项 (config dict):
        joint_limits_lower: 关节下限数组
        joint_limits_upper: 关节上限数组
        action_limits_lower: 动作下限数组
        action_limits_upper: 动作上限数组
        max_frame_jump: 帧间最大允许跳变 (默认 0.5)
        min_trajectory_length: 最短轨迹帧数 (默认 10)
        max_trajectory_length: 最长轨迹帧数 (默认 10000)
        constant_threshold: 判定"死关节"的标准差阈值 (默认 1e-8)
        expected_image_shape: 期望的图像形状 (H, W, C)
        quaternion_joint_indices: 四元数关节的起始索引列表
        hz: 数据采集频率 (默认 50)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    # ----------------------------------------------------------
    # 1. NaN / Inf 检查
    # ----------------------------------------------------------
    def check_nan_inf(self, data: dict) -> ValidationResult:
        """
        检查所有数值数组中是否存在 NaN 或 Inf。

        原理: NaN/Inf 通常来自除零、数值溢出、或传感器故障。
              它们会导致后续计算（如归一化、损失函数）崩溃。
        """
        bad_fields = {}

        def _scan(obj, prefix=""):
            if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.floating):
                nan_count = int(np.isnan(obj).sum())
                inf_count = int(np.isinf(obj).sum())
                if nan_count > 0 or inf_count > 0:
                    bad_fields[prefix] = {"nan": nan_count, "inf": inf_count}
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _scan(v, f"{prefix}/{k}" if prefix else k)

        _scan(data)

        if bad_fields:
            return ValidationResult(
                check_name="nan_inf_check",
                passed=False,
                severity=Severity.ERROR,
                message=f"发现 NaN/Inf，涉及 {len(bad_fields)} 个字段",
                details={"bad_fields": bad_fields},
            )
        return ValidationResult(
            check_name="nan_inf_check",
            passed=True,
            severity=Severity.ERROR,
            message="所有数值字段均无 NaN/Inf",
        )

    # ----------------------------------------------------------
    # 2. 四元数归一化检查
    # ----------------------------------------------------------
    def check_quaternion_normalization(self, data: dict) -> ValidationResult:
        """
        检查四元数是否归一化（模为 1）。

        原理: MuJoCo 中自由关节 (freejoint) 的 qpos 前 3 个是位置，
              后 4 个是四元数 (w, x, y, z)。球关节 (ball) 也用四元数。
              四元数必须满足 |q| = 1，否则旋转矩阵将不正交。
        """
        qpos = data.get("observations", {}).get("qpos", None)
        if qpos is None:
            return ValidationResult(
                check_name="quaternion_normalization",
                passed=True,
                severity=Severity.WARNING,
                message="未找到 qpos 数据，跳过四元数检查",
            )

        # 从配置中获取四元数关节索引
        quat_indices = self.config.get("quaternion_joint_indices", [])
        if not quat_indices:
            return ValidationResult(
                check_name="quaternion_normalization",
                passed=True,
                severity=Severity.INFO,
                message="未配置四元数关节索引，跳过检查",
            )

        violations = []
        tolerance = 1e-3  # 归一化容差

        for start_idx in quat_indices:
            if start_idx + 4 > qpos.shape[1]:
                continue
            # 提取四元数列 (T, 4)
            quats = qpos[:, start_idx:start_idx + 4]
            norms = np.linalg.norm(quats, axis=1)
            bad_frames = np.where(np.abs(norms - 1.0) > tolerance)[0]
            if len(bad_frames) > 0:
                violations.append({
                    "quat_start_idx": int(start_idx),
                    "bad_frame_count": int(len(bad_frames)),
                    "first_bad_frames": bad_frames[:5].tolist(),
                    "norm_range": [float(norms.min()), float(norms.max())],
                })

        if violations:
            return ValidationResult(
                check_name="quaternion_normalization",
                passed=False,
                severity=Severity.ERROR,
                message=f"四元数归一化异常，涉及 {len(violations)} 组四元数",
                details={"violations": violations},
            )
        return ValidationResult(
            check_name="quaternion_normalization",
            passed=True,
            severity=Severity.ERROR,
            message="所有四元数归一化正常 (|q| ≈ 1)",
        )

    # ----------------------------------------------------------
    # 3. 关节限位违规检测
    # ----------------------------------------------------------
    def check_joint_limits(self, data: dict) -> ValidationResult:
        """
        检查 qpos 是否超出关节物理限位。

        原理: 每个旋转关节都有机械限位。超出限位的数据通常来自
              标定错误、传感器漂移、或仿真不稳定。
        """
        qpos = data.get("observations", {}).get("qpos", None)
        if qpos is None:
            return ValidationResult(
                check_name="joint_limits",
                passed=True,
                severity=Severity.WARNING,
                message="未找到 qpos 数据，跳过限位检查",
            )

        lower = self.config.get("joint_limits_lower")
        upper = self.config.get("joint_limits_upper")
        if lower is None or upper is None:
            return ValidationResult(
                check_name="joint_limits",
                passed=True,
                severity=Severity.INFO,
                message="未配置关节限位，跳过检查",
            )

        lower = np.asarray(lower)
        upper = np.asarray(upper)
        nj = min(qpos.shape[1], len(lower), len(upper))

        violations = {}
        for j in range(nj):
            below = np.where(qpos[:, j] < lower[j])[0]
            above = np.where(qpos[:, j] > upper[j])[0]
            if len(below) > 0 or len(above) > 0:
                violations[f"joint_{j}"] = {
                    "below_limit_frames": int(len(below)),
                    "above_limit_frames": int(len(above)),
                    "actual_range": [float(qpos[:, j].min()), float(qpos[:, j].max())],
                    "allowed_range": [float(lower[j]), float(upper[j])],
                }

        if violations:
            return ValidationResult(
                check_name="joint_limits",
                passed=False,
                severity=Severity.ERROR,
                message=f"关节限位违规，涉及 {len(violations)} 个关节",
                details={"violations": violations},
            )
        return ValidationResult(
            check_name="joint_limits",
            passed=True,
            severity=Severity.ERROR,
            message="所有关节值均在限位范围内",
        )

    # ----------------------------------------------------------
    # 4. 帧间跳变检测
    # ----------------------------------------------------------
    def check_frame_jumps(self, data: dict) -> ValidationResult:
        """
        检测相邻帧之间是否存在异常大的跳变。

        原理: 物理系统具有惯性，关节位置不可能在一帧内突变。
              大跳变通常意味着数据丢帧、拼接错误、或仿真重置。
        """
        qpos = data.get("observations", {}).get("qpos", None)
        if qpos is None:
            return ValidationResult(
                check_name="frame_jumps",
                passed=True,
                severity=Severity.WARNING,
                message="未找到 qpos 数据，跳过跳变检查",
            )

        max_jump = self.config.get("max_frame_jump", 0.5)

        # 计算帧间差分的绝对值
        diffs = np.abs(np.diff(qpos, axis=0))  # (T-1, nq)
        max_diffs = diffs.max(axis=1)  # (T-1,) 每帧的最大跳变

        jump_frames = np.where(max_diffs > max_jump)[0]

        if len(jump_frames) > 0:
            # 记录前 10 个最严重的跳变
            worst_indices = np.argsort(max_diffs)[-10:][::-1]
            worst_details = []
            for idx in worst_indices:
                if max_diffs[idx] > max_jump:
                    joint_idx = int(np.argmax(diffs[idx]))
                    worst_details.append({
                        "frame": int(idx),
                        "joint": int(joint_idx),
                        "jump_magnitude": float(max_diffs[idx]),
                    })

            return ValidationResult(
                check_name="frame_jumps",
                passed=False,
                severity=Severity.WARNING,
                message=f"检测到 {len(jump_frames)} 处帧间跳变 (阈值={max_jump})",
                details={
                    "jump_count": int(len(jump_frames)),
                    "threshold": max_jump,
                    "worst_jumps": worst_details,
                },
            )
        return ValidationResult(
            check_name="frame_jumps",
            passed=True,
            severity=Severity.WARNING,
            message=f"无帧间跳变 (阈值={max_jump})",
        )

    # ----------------------------------------------------------
    # 5. 轨迹长度校验
    # ----------------------------------------------------------
    def check_trajectory_length(self, data: dict) -> ValidationResult:
        """
        校验轨迹帧数是否在合理范围内。

        原理: 太短的轨迹（如 < 10 帧）可能是采集中断；
              太长的轨迹可能包含多次尝试未被正确分割。
        """
        qpos = data.get("observations", {}).get("qpos", None)
        action = data.get("action", None)

        length = 0
        if qpos is not None:
            length = qpos.shape[0]
        elif action is not None:
            length = action.shape[0]
        else:
            return ValidationResult(
                check_name="trajectory_length",
                passed=True,
                severity=Severity.WARNING,
                message="未找到 qpos 或 action 数据",
            )

        min_len = self.config.get("min_trajectory_length", 10)
        max_len = self.config.get("max_trajectory_length", 10000)

        if length < min_len:
            return ValidationResult(
                check_name="trajectory_length",
                passed=False,
                severity=Severity.ERROR,
                message=f"轨迹过短: {length} 帧 (最小要求 {min_len})",
                details={"length": length, "min": min_len, "max": max_len},
            )
        if length > max_len:
            return ValidationResult(
                check_name="trajectory_length",
                passed=False,
                severity=Severity.WARNING,
                message=f"轨迹过长: {length} 帧 (最大建议 {max_len})",
                details={"length": length, "min": min_len, "max": max_len},
            )
        return ValidationResult(
            check_name="trajectory_length",
            passed=True,
            severity=Severity.ERROR,
            message=f"轨迹长度 {length} 帧，在合理范围 [{min_len}, {max_len}]",
            details={"length": length},
        )

    # ----------------------------------------------------------
    # 6. 死关节检测
    # ----------------------------------------------------------
    def check_constant_joints(self, data: dict) -> ValidationResult:
        """
        检测始终不变（"死掉"）的关节。

        原理: 如果某个关节在整条轨迹中完全不动，可能是:
              - 传感器失效（读数锁定在某值）
              - 控制信号未送达该关节
              - 关节被机械锁定
              这些数据对学习该关节的控制策略没有价值。
        """
        qpos = data.get("observations", {}).get("qpos", None)
        if qpos is None:
            return ValidationResult(
                check_name="constant_joints",
                passed=True,
                severity=Severity.WARNING,
                message="未找到 qpos 数据，跳过死关节检查",
            )

        threshold = self.config.get("constant_threshold", 1e-8)
        stds = qpos.std(axis=0)
        dead_joints = np.where(stds < threshold)[0]

        if len(dead_joints) > 0:
            return ValidationResult(
                check_name="constant_joints",
                passed=False,
                severity=Severity.WARNING,
                message=f"发现 {len(dead_joints)} 个死关节 (std < {threshold})",
                details={
                    "dead_joint_indices": dead_joints.tolist(),
                    "stds": {f"joint_{j}": float(stds[j]) for j in dead_joints},
                },
            )
        return ValidationResult(
            check_name="constant_joints",
            passed=True,
            severity=Severity.WARNING,
            message="所有关节均有变化",
        )

    # ----------------------------------------------------------
    # 7. 时间戳单调性检查
    # ----------------------------------------------------------
    def check_timestamp_monotonicity(self, data: dict) -> ValidationResult:
        """
        检查时间戳是否严格递增。

        原理: 物理时间不可回退。如果时间戳乱序，说明数据拼接有问题，
              或者系统时钟被修改过。
        """
        timestamps = data.get("timestamps", None)
        if timestamps is None:
            return ValidationResult(
                check_name="timestamp_monotonicity",
                passed=True,
                severity=Severity.INFO,
                message="未找到 timestamps 字段，跳过检查",
            )

        timestamps = np.asarray(timestamps)
        diffs = np.diff(timestamps)

        # 检查严格递增
        non_increasing = np.where(diffs <= 0)[0]
        if len(non_increasing) > 0:
            return ValidationResult(
                check_name="timestamp_monotonicity",
                passed=False,
                severity=Severity.ERROR,
                message=f"时间戳非单调递增，{len(non_increasing)} 处异常",
                details={
                    "violation_count": int(len(non_increasing)),
                    "first_violations": non_increasing[:10].tolist(),
                    "first_diffs": diffs[non_increasing[:10]].tolist(),
                },
            )

        # 检查间隔是否均匀（可选）
        hz = self.config.get("hz", 50)
        expected_dt = 1.0 / hz
        dt_std = float(diffs.std())
        dt_mean = float(diffs.mean())

        return ValidationResult(
            check_name="timestamp_monotonicity",
            passed=True,
            severity=Severity.ERROR,
            message=f"时间戳单调递增，平均间隔 {dt_mean:.4f}s (期望 {expected_dt:.4f}s)",
            details={"dt_mean": dt_mean, "dt_std": dt_std, "expected_dt": expected_dt},
        )

    # ----------------------------------------------------------
    # 8. 动作范围检查
    # ----------------------------------------------------------
    def check_action_range(self, data: dict) -> ValidationResult:
        """
        检查动作值是否在指定范围内。

        原理: 电机有扭矩限制，执行器有行程限制。
              超出范围的动作在真实机器人上会被截断或引发保护停机。
        """
        action = data.get("action", None)
        if action is None:
            return ValidationResult(
                check_name="action_range",
                passed=True,
                severity=Severity.WARNING,
                message="未找到 action 数据，跳过检查",
            )

        lower = self.config.get("action_limits_lower")
        upper = self.config.get("action_limits_upper")
        if lower is None or upper is None:
            # 默认检查 [-1, 1] 范围（许多策略使用归一化动作）
            lower = np.full(action.shape[1], -1.0)
            upper = np.full(action.shape[1], 1.0)

        lower = np.asarray(lower)
        upper = np.asarray(upper)

        violations = {}
        na = min(action.shape[1], len(lower), len(upper))
        for a in range(na):
            below = int((action[:, a] < lower[a]).sum())
            above = int((action[:, a] > upper[a]).sum())
            if below > 0 or above > 0:
                violations[f"action_{a}"] = {
                    "below_count": below,
                    "above_count": above,
                    "actual_range": [float(action[:, a].min()), float(action[:, a].max())],
                    "allowed_range": [float(lower[a]), float(upper[a])],
                }

        if violations:
            return ValidationResult(
                check_name="action_range",
                passed=False,
                severity=Severity.WARNING,
                message=f"动作超出范围，涉及 {len(violations)} 个维度",
                details={"violations": violations},
            )
        return ValidationResult(
            check_name="action_range",
            passed=True,
            severity=Severity.WARNING,
            message="所有动作值均在允许范围内",
        )

    # ----------------------------------------------------------
    # 9. 图像维度校验
    # ----------------------------------------------------------
    def check_image_dimensions(self, data: dict) -> ValidationResult:
        """
        检查图像数据的维度是否一致且符合预期。

        原理: 如果图像分辨率不一致，批量处理时会报错。
              错误的通道数（如 RGBA vs RGB）也会导致模型输入不匹配。
        """
        images_dict = data.get("observations", {}).get("images", None)
        if images_dict is None:
            return ValidationResult(
                check_name="image_dimensions",
                passed=True,
                severity=Severity.INFO,
                message="未找到图像数据，跳过检查",
            )

        expected_shape = self.config.get("expected_image_shape", None)
        issues = {}

        for cam_name, imgs in images_dict.items():
            if not isinstance(imgs, np.ndarray):
                continue
            if imgs.ndim != 4:  # 期望 (T, H, W, C)
                issues[cam_name] = f"维度异常: 期望 4D (T,H,W,C)，实际 {imgs.ndim}D"
                continue
            if expected_shape is not None:
                actual = imgs.shape[1:]  # (H, W, C)
                if actual != tuple(expected_shape):
                    issues[cam_name] = f"分辨率不匹配: 期望 {expected_shape}，实际 {list(actual)}"

        if issues:
            return ValidationResult(
                check_name="image_dimensions",
                passed=False,
                severity=Severity.ERROR,
                message=f"图像维度异常: {len(issues)} 个相机",
                details={"issues": issues},
            )
        return ValidationResult(
            check_name="image_dimensions",
            passed=True,
            severity=Severity.ERROR,
            message="所有图像维度正确",
        )

    # ----------------------------------------------------------
    # 执行所有校验
    # ----------------------------------------------------------
    def validate(self, data: dict, episode_id: str = "unknown") -> ValidationReport:
        """
        对一个 episode 执行全部校验，返回 ValidationReport。
        """
        report = ValidationReport(episode_id=episode_id)

        # 记录元数据
        qpos = data.get("observations", {}).get("qpos", None)
        if qpos is not None:
            report.metadata["num_frames"] = int(qpos.shape[0])
            report.metadata["num_joints"] = int(qpos.shape[1])

        # 依次执行所有检查
        checks = [
            self.check_nan_inf,
            self.check_quaternion_normalization,
            self.check_joint_limits,
            self.check_frame_jumps,
            self.check_trajectory_length,
            self.check_constant_joints,
            self.check_timestamp_monotonicity,
            self.check_action_range,
            self.check_image_dimensions,
        ]
        for check_fn in checks:
            result = check_fn(data)
            report.add(result)

        return report


# ============================================================
# 批量校验器
# ============================================================

class BatchValidator:
    """
    批量校验：扫描一个目录下的所有 episode 文件，
    逐个校验并生成汇总统计。

    为了演示方便，这里用内存中的 episode 列表代替文件 I/O。
    """

    def __init__(self, validator: DataValidator):
        self.validator = validator

    def validate_episodes(
        self, episodes: List[Tuple[str, dict]]
    ) -> Dict[str, Any]:
        """
        批量校验多个 episode。

        参数:
            episodes: [(episode_id, data_dict), ...]

        返回: 汇总报告 dict
        """
        reports = []
        for ep_id, data in episodes:
            report = self.validator.validate(data, episode_id=ep_id)
            reports.append(report)

        # 汇总统计
        total = len(reports)
        passed = sum(1 for r in reports if r.passed)
        failed = total - passed

        # 每种检查的通过率
        check_pass_rates = {}
        if reports:
            for result in reports[0].results:
                name = result.check_name
                pass_count = sum(
                    1 for r in reports
                    for res in r.results
                    if res.check_name == name and res.passed
                )
                check_pass_rates[name] = {
                    "pass_count": pass_count,
                    "total": total,
                    "pass_rate": pass_count / total if total > 0 else 0,
                }

        summary = {
            "total_episodes": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "check_pass_rates": check_pass_rates,
            "episode_reports": [r.summary() for r in reports],
        }

        return summary, reports


# ============================================================
# 测试数据生成
# ============================================================

def generate_good_episode(num_frames=200, nq=7, nu=7):
    """生成一个"正常"的 episode 用于测试"""
    t = np.linspace(0, 4 * np.pi, num_frames)
    qpos = np.column_stack([np.sin(t * (i + 1)) * 0.3 for i in range(nq)])
    qvel = np.column_stack([np.cos(t * (i + 1)) * 0.2 for i in range(nq)])
    action = np.column_stack([np.sin(t * (i + 1) + 0.1) * 0.4 for i in range(nu)])
    timestamps = np.arange(num_frames) / 50.0

    return {
        "observations": {
            "qpos": qpos,
            "qvel": qvel,
        },
        "action": action,
        "timestamps": timestamps,
    }


def generate_bad_episode(num_frames=200, nq=7, nu=7):
    """
    生成一个包含各种问题的 episode:
      - 注入 NaN
      - 关节超限
      - 帧间跳变
      - 死关节
      - 时间戳乱序
      - 动作超限
    """
    t = np.linspace(0, 4 * np.pi, num_frames)
    qpos = np.column_stack([np.sin(t * (i + 1)) * 0.3 for i in range(nq)])
    qvel = np.column_stack([np.cos(t * (i + 1)) * 0.2 for i in range(nq)])
    action = np.column_stack([np.sin(t * (i + 1) + 0.1) * 0.4 for i in range(nu)])
    timestamps = np.arange(num_frames) / 50.0

    # 注入 NaN
    qpos[50, 2] = np.nan
    qpos[51, 2] = np.nan

    # 关节 0 超限
    qpos[100:110, 0] = 5.0  # 超出 [-π, π]

    # 帧间跳变
    qpos[150, 3] = qpos[149, 3] + 3.0

    # 死关节 (关节 6 设为常数)
    qpos[:, 6] = 0.42

    # 时间戳乱序
    timestamps[80] = timestamps[79] - 0.1

    # 动作超限
    action[30:40, 1] = 2.5

    return {
        "observations": {
            "qpos": qpos,
            "qvel": qvel,
        },
        "action": action,
        "timestamps": timestamps,
    }


# ============================================================
# 演示函数
# ============================================================

def section_1_basic_validation():
    """第 1 节: 单个 episode 校验"""
    print(DIVIDER)
    print("第 1 节：单个 Episode 校验")
    print(DIVIDER)

    # 配置校验器
    config = {
        "joint_limits_lower": [-np.pi] * 7,
        "joint_limits_upper": [np.pi] * 7,
        "action_limits_lower": [-1.0] * 7,
        "action_limits_upper": [1.0] * 7,
        "max_frame_jump": 0.5,
        "min_trajectory_length": 10,
        "max_trajectory_length": 5000,
        "constant_threshold": 1e-8,
        "hz": 50,
    }
    validator = DataValidator(config)

    # --- 测试正常数据 ---
    print("\n[测试 1] 正常 Episode:")
    good_data = generate_good_episode()
    report_good = validator.validate(good_data, episode_id="good_ep_001")
    report_good.print_summary()

    # --- 测试异常数据 ---
    print("\n" + SUB_DIVIDER)
    print("\n[测试 2] 异常 Episode (注入了多种问题):")
    bad_data = generate_bad_episode()
    report_bad = validator.validate(bad_data, episode_id="bad_ep_001")
    report_bad.print_summary()

    return validator


def section_2_detailed_report():
    """第 2 节: 查看详细报告"""
    print("\n" + DIVIDER)
    print("第 2 节：详细校验报告 (JSON 格式)")
    print(DIVIDER)

    config = {
        "joint_limits_lower": [-np.pi] * 7,
        "joint_limits_upper": [np.pi] * 7,
        "max_frame_jump": 0.5,
        "constant_threshold": 1e-8,
    }
    validator = DataValidator(config)
    bad_data = generate_bad_episode()
    report = validator.validate(bad_data, episode_id="bad_ep_detail")

    summary = report.summary()

    # 只打印失败的检查项的详情
    print("\n  失败项详情:")
    for check in summary["checks"]:
        if not check["passed"]:
            print(f"\n  📋 {check['check_name']} [{check['severity']}]")
            print(f"     {check['message']}")
            if check["details"]:
                detail_str = json.dumps(check["details"], indent=6, ensure_ascii=False)
                # 截断过长的输出
                if len(detail_str) > 500:
                    detail_str = detail_str[:500] + "\n      ... (截断)"
                print(f"     详情: {detail_str}")


def section_3_batch_validation():
    """第 3 节: 批量校验"""
    print("\n" + DIVIDER)
    print("第 3 节：批量校验 & 汇总统计")
    print(DIVIDER)

    config = {
        "joint_limits_lower": [-np.pi] * 7,
        "joint_limits_upper": [np.pi] * 7,
        "action_limits_lower": [-1.0] * 7,
        "action_limits_upper": [1.0] * 7,
        "max_frame_jump": 0.5,
        "constant_threshold": 1e-8,
    }
    validator = DataValidator(config)
    batch_validator = BatchValidator(validator)

    # 生成混合数据集: 7 好 + 3 坏
    episodes = []
    for i in range(7):
        episodes.append((f"good_ep_{i:03d}", generate_good_episode(
            num_frames=np.random.randint(100, 400)
        )))
    for i in range(3):
        episodes.append((f"bad_ep_{i:03d}", generate_bad_episode(
            num_frames=np.random.randint(100, 400)
        )))

    # 打乱顺序
    np.random.shuffle(episodes)

    print(f"\n  批量校验 {len(episodes)} 个 episodes...")
    t0 = time.time()
    summary, reports = batch_validator.validate_episodes(episodes)
    elapsed = time.time() - t0

    print(f"  耗时: {elapsed:.3f}s")
    print(f"\n  汇总结果:")
    print(f"    总 episode 数: {summary['total_episodes']}")
    print(f"    通过: {summary['passed']}  失败: {summary['failed']}")
    print(f"    通过率: {summary['pass_rate']:.1%}")

    print(f"\n  各检查项通过率:")
    for name, stats in summary["check_pass_rates"].items():
        bar_len = int(stats["pass_rate"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {name:30s} [{bar}] {stats['pass_rate']:.0%}")

    # 每个 episode 的状态
    print(f"\n  各 Episode 状态:")
    for r in reports:
        status = "✅" if r.passed else "❌"
        print(f"    {status} {r.episode_id:20s} (错误: {r.error_count}, 警告: {r.warning_count})")


def section_4_custom_checks():
    """第 4 节: 展示如何扩展自定义检查"""
    print("\n" + DIVIDER)
    print("第 4 节：扩展自定义校验")
    print(DIVIDER)

    print("""
    DataValidator 设计为可扩展的:

    1. 继承 DataValidator
    2. 添加新的 check_xxx 方法
    3. 重写 validate() 加入新检查

    示例: 添加速度平滑度检查
    """)

    class ExtendedValidator(DataValidator):
        """扩展校验器: 增加速度平滑度检查"""

        def check_velocity_smoothness(self, data: dict) -> ValidationResult:
            """检查速度是否过度振荡（频率异常高）"""
            qvel = data.get("observations", {}).get("qvel", None)
            if qvel is None:
                return ValidationResult(
                    check_name="velocity_smoothness",
                    passed=True,
                    severity=Severity.INFO,
                    message="未找到 qvel，跳过平滑度检查",
                )

            # 计算加速度（速度的差分）
            accel = np.diff(qvel, axis=0)
            # 统计符号变化次数（振荡检测）
            sign_changes = np.sum(np.diff(np.sign(accel), axis=0) != 0, axis=0)
            max_ratio = sign_changes.max() / max(qvel.shape[0] - 2, 1)

            if max_ratio > 0.8:
                return ValidationResult(
                    check_name="velocity_smoothness",
                    passed=False,
                    severity=Severity.WARNING,
                    message=f"速度振荡严重 (符号变化率 {max_ratio:.2f})",
                    details={"sign_change_ratio": float(max_ratio)},
                )
            return ValidationResult(
                check_name="velocity_smoothness",
                passed=True,
                severity=Severity.WARNING,
                message=f"速度平滑度正常 (符号变化率 {max_ratio:.2f})",
            )

        def validate(self, data, episode_id="unknown"):
            report = super().validate(data, episode_id)
            report.add(self.check_velocity_smoothness(data))
            return report

    # 演示
    ext_validator = ExtendedValidator({
        "joint_limits_lower": [-np.pi] * 7,
        "joint_limits_upper": [np.pi] * 7,
    })
    data = generate_good_episode()
    report = ext_validator.validate(data, "extended_ep_001")
    report.print_summary()


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("第 6 章 · 01 - 数据校验器 (Data Validator)")
    print("构建企业级数据校验流水线")
    print(DIVIDER)

    section_1_basic_validation()
    section_2_detailed_report()
    section_3_batch_validation()
    section_4_custom_checks()

    print("\n" + DIVIDER)
    print("✅ 数据校验器演示完成！")
    print(DIVIDER)


if __name__ == "__main__":
    main()
