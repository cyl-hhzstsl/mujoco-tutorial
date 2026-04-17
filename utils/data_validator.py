"""
数据质量校验器 (Data Validator)

功能: 对机器人轨迹数据进行自动化质量校验，
     包括 NaN 检测、四元数归一化、关节限位、跳变检测等。

使用:
  validator = DataValidator()
  report = validator.validate_episode(data_dict)
  report.summary()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


@dataclass
class ValidationIssue:
    """单个校验问题。"""
    severity: str           # "error", "warning", "info"
    check_name: str         # 校验项名称
    message: str            # 详细描述
    indices: Optional[List[int]] = None  # 问题数据的索引
    value: Optional[Any] = None          # 问题值

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "severity": self.severity,
            "check": self.check_name,
            "message": self.message,
        }
        if self.indices is not None:
            d["indices"] = self.indices[:10]  # 最多保留 10 个
            d["total_count"] = len(self.indices)
        if self.value is not None:
            d["value"] = self.value
        return d


@dataclass
class ValidationReport:
    """校验报告。"""
    source: str = ""
    issues: List[ValidationIssue] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warned: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """没有 error 级别的问题则视为有效。"""
        return all(i.severity != "error" for i in self.issues)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def add_issue(self, severity: str, check_name: str, message: str,
                  indices: Optional[List[int]] = None,
                  value: Optional[Any] = None) -> None:
        self.issues.append(ValidationIssue(
            severity=severity, check_name=check_name,
            message=message, indices=indices, value=value,
        ))
        if severity == "error":
            self.checks_failed += 1
        elif severity == "warning":
            self.checks_warned += 1

    def pass_check(self, check_name: str) -> None:
        self.checks_passed += 1

    def summary(self) -> str:
        """生成可读的校验摘要。"""
        lines = [
            f"校验报告: {self.source}",
            f"  通过: {self.checks_passed}, "
            f"警告: {self.checks_warned}, "
            f"错误: {self.checks_failed}",
            f"  状态: {'✓ 有效' if self.is_valid else '✗ 无效'}",
        ]
        if self.issues:
            lines.append("  问题列表:")
            for issue in self.issues:
                icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}[issue.severity]
                lines.append(f"    {icon} [{issue.check_name}] {issue.message}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "is_valid": self.is_valid,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_warned": self.checks_warned,
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata,
        }


@dataclass
class ValidatorConfig:
    """校验器配置参数。"""
    check_nan: bool = True
    check_quaternion: bool = True
    check_joint_limits: bool = True
    check_jumps: bool = True
    check_length: bool = True
    check_velocity: bool = True

    quat_norm_tol: float = 1e-3        # 四元数归一化容差
    jump_threshold: float = 0.5         # 帧间跳变阈值 (rad)
    min_length: int = 10                # 最小帧数
    max_velocity: float = 50.0          # 最大关节速度 (rad/s)
    joint_limit_margin: float = 0.01    # 关节限位容差 (rad)


class DataValidator:
    """
    机器人轨迹数据校验器。

    校验项目:
      1. NaN 检测: 任何字段包含 NaN 都是错误
      2. 四元数归一化: 四元数范数应接近 1
      3. 关节限位: qpos 不应超出关节范围
      4. 帧间跳变: 相邻帧的变化不应过大
      5. 长度检查: 轨迹不应太短
      6. 速度检查: 关节速度不应超过物理极限

    使用:
      validator = DataValidator(config)
      report = validator.validate_episode(data)
    """

    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()

    def validate_episode(self,
                         data: Union[str, Path, Dict[str, np.ndarray]],
                         joint_ranges: Optional[np.ndarray] = None,
                         dt: Optional[float] = None) -> ValidationReport:
        """
        校验一条轨迹。

        参数:
          data: 文件路径或数据字典 {"qpos": array, "qvel": array, ...}
          joint_ranges: (n_joints, 2) 的关节范围数组
          dt: 仿真步长（用于速度校验）

        返回:
          ValidationReport
        """
        if isinstance(data, (str, Path)):
            data = self._load_data(data)

        report = ValidationReport(source=str(type(data)))

        # 提取常用字段
        qpos = data.get("qpos")
        qvel = data.get("qvel")
        quat = data.get("quat")

        if qpos is None:
            report.add_issue("error", "data_format", "数据中没有 qpos 字段")
            return report

        report.metadata["n_frames"] = len(qpos)
        report.metadata["n_joints"] = qpos.shape[1] if qpos.ndim > 1 else 1

        # 逐项校验
        if self.config.check_nan:
            self._check_nan(report, data)

        if self.config.check_length:
            self._check_length(report, qpos)

        if self.config.check_quaternion and quat is not None:
            self._check_quaternion(report, quat)

        if self.config.check_joint_limits and joint_ranges is not None:
            self._check_joint_limits(report, qpos, joint_ranges)

        if self.config.check_jumps:
            self._check_jumps(report, qpos)

        if self.config.check_velocity and qvel is not None:
            self._check_velocity(report, qvel)
        elif self.config.check_velocity and dt is not None:
            self._check_velocity_from_qpos(report, qpos, dt)

        return report

    def _load_data(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """从文件加载数据。"""
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix == ".npy":
            arr = np.load(filepath)
            return {"qpos": arr}
        elif suffix == ".npz":
            npz = np.load(filepath)
            return dict(npz)
        elif suffix == ".pkl":
            import pickle
            with open(filepath, "rb") as f:
                return pickle.load(f)
        elif suffix in (".h5", ".hdf5"):
            import h5py
            data = {}
            with h5py.File(filepath, "r") as f:
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data[key] = f[key][:]
                    elif isinstance(f[key], h5py.Group):
                        for sub_key in f[key].keys():
                            if isinstance(f[key][sub_key], h5py.Dataset):
                                data[f"{key}/{sub_key}"] = f[key][sub_key][:]
            return data
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    def _check_nan(self, report: ValidationReport,
                   data: Dict[str, np.ndarray]) -> None:
        """检测 NaN 值。"""
        found_nan = False
        for key, arr in data.items():
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
                nan_mask = np.isnan(arr)
                if nan_mask.any():
                    nan_count = int(nan_mask.sum())
                    if arr.ndim >= 1:
                        nan_frames = list(np.where(nan_mask.any(axis=tuple(range(1, arr.ndim))))[0])
                    else:
                        nan_frames = [0]
                    report.add_issue(
                        "error", "nan_check",
                        f"字段 '{key}' 包含 {nan_count} 个 NaN 值",
                        indices=nan_frames,
                    )
                    found_nan = True
        if not found_nan:
            report.pass_check("nan_check")

    def _check_length(self, report: ValidationReport,
                      qpos: np.ndarray) -> None:
        """检查轨迹长度。"""
        if len(qpos) < self.config.min_length:
            report.add_issue(
                "error", "length_check",
                f"轨迹过短: {len(qpos)} 帧 (最少 {self.config.min_length})",
                value=len(qpos),
            )
        else:
            report.pass_check("length_check")

    def _check_quaternion(self, report: ValidationReport,
                          quat: np.ndarray) -> None:
        """检查四元数归一化。"""
        if quat.ndim == 1:
            quat = quat.reshape(-1, 4)

        norms = np.linalg.norm(quat, axis=-1)
        bad_mask = np.abs(norms - 1.0) > self.config.quat_norm_tol
        if bad_mask.any():
            bad_indices = list(np.where(bad_mask)[0])
            worst_norm = norms[bad_mask][np.argmax(np.abs(norms[bad_mask] - 1.0))]
            report.add_issue(
                "warning", "quaternion_check",
                f"{bad_mask.sum()} 个四元数未归一化 (最大偏差: {abs(worst_norm - 1.0):.6f})",
                indices=bad_indices,
                value=float(worst_norm),
            )
        else:
            report.pass_check("quaternion_check")

    def _check_joint_limits(self, report: ValidationReport,
                            qpos: np.ndarray,
                            joint_ranges: np.ndarray) -> None:
        """检查关节限位。"""
        n_joints = min(qpos.shape[1] if qpos.ndim > 1 else 1, len(joint_ranges))
        violations = []

        for j in range(n_joints):
            low, high = joint_ranges[j]
            if qpos.ndim > 1:
                col = qpos[:, j]
            else:
                col = qpos

            margin = self.config.joint_limit_margin
            below = col < (low - margin)
            above = col > (high + margin)
            bad = below | above

            if bad.any():
                frames = list(np.where(bad)[0])
                violations.append((j, bad.sum(), frames))

        if violations:
            for j, count, frames in violations:
                report.add_issue(
                    "warning", "joint_limit_check",
                    f"关节 {j}: {count} 帧超出限位范围",
                    indices=frames,
                )
        else:
            report.pass_check("joint_limit_check")

    def _check_jumps(self, report: ValidationReport,
                     qpos: np.ndarray) -> None:
        """检测帧间跳变。"""
        if len(qpos) < 2:
            report.pass_check("jump_check")
            return

        diffs = np.abs(np.diff(qpos, axis=0))
        max_diffs = diffs.max(axis=-1) if diffs.ndim > 1 else diffs
        jump_mask = max_diffs > self.config.jump_threshold

        if jump_mask.any():
            jump_frames = list(np.where(jump_mask)[0])
            max_jump = float(max_diffs[jump_mask].max())
            report.add_issue(
                "warning", "jump_check",
                f"{jump_mask.sum()} 帧存在跳变 (最大跳变: {max_jump:.4f} rad, "
                f"阈值: {self.config.jump_threshold})",
                indices=jump_frames,
                value=max_jump,
            )
        else:
            report.pass_check("jump_check")

    def _check_velocity(self, report: ValidationReport,
                        qvel: np.ndarray) -> None:
        """检查关节速度。"""
        max_vel = np.abs(qvel).max()
        if max_vel > self.config.max_velocity:
            high_frames = list(np.where(np.abs(qvel).max(axis=-1) > self.config.max_velocity)[0])
            report.add_issue(
                "warning", "velocity_check",
                f"最大关节速度 {max_vel:.2f} rad/s 超过阈值 {self.config.max_velocity}",
                indices=high_frames,
                value=float(max_vel),
            )
        else:
            report.pass_check("velocity_check")

    def _check_velocity_from_qpos(self, report: ValidationReport,
                                  qpos: np.ndarray, dt: float) -> None:
        """从 qpos 差分估算速度并检查。"""
        if len(qpos) < 2:
            return
        estimated_vel = np.diff(qpos, axis=0) / dt
        self._check_velocity(report, estimated_vel)


# ============================================================
# 独立运行时的演示
# ============================================================

def main():
    print("=" * 60)
    print("DataValidator 演示")
    print("=" * 60)

    np.random.seed(42)

    # --- 测试 1: 正常数据 ---
    print("\n--- 测试 1: 正常数据 ---")
    good_data = {
        "qpos": np.random.uniform(-1, 1, size=(100, 3)),
        "qvel": np.random.uniform(-5, 5, size=(100, 3)),
    }
    good_data["qpos"] = np.cumsum(good_data["qpos"] * 0.01, axis=0)

    validator = DataValidator()
    report = validator.validate_episode(good_data)
    print(report.summary())

    # --- 测试 2: 含 NaN 的数据 ---
    print("\n--- 测试 2: 含 NaN 数据 ---")
    bad_data = {
        "qpos": np.random.uniform(-1, 1, size=(100, 3)),
        "qvel": np.random.uniform(-5, 5, size=(100, 3)),
    }
    bad_data["qpos"][10, 1] = np.nan
    bad_data["qpos"][50, 0] = np.nan

    report = validator.validate_episode(bad_data)
    print(report.summary())

    # --- 测试 3: 帧间跳变 ---
    print("\n--- 测试 3: 帧间跳变 ---")
    jump_data = {
        "qpos": np.cumsum(np.random.uniform(-0.01, 0.01, size=(100, 2)), axis=0),
    }
    jump_data["qpos"][30] += 2.0  # 人为制造跳变

    report = validator.validate_episode(jump_data)
    print(report.summary())

    # --- 测试 4: 数据过短 ---
    print("\n--- 测试 4: 数据过短 ---")
    short_data = {"qpos": np.array([[0.1, 0.2], [0.3, 0.4]])}
    report = validator.validate_episode(short_data)
    print(report.summary())

    # --- 测试 5: 关节限位超出 ---
    print("\n--- 测试 5: 关节限位 ---")
    limit_data = {
        "qpos": np.random.uniform(-2, 2, size=(100, 2)),
    }
    joint_ranges = np.array([[-1.5, 1.5], [-1.0, 1.0]])

    report = validator.validate_episode(limit_data, joint_ranges=joint_ranges)
    print(report.summary())

    # --- 输出 to_dict ---
    print("\n--- to_dict() ---")
    d = report.to_dict()
    print(f"  is_valid: {d['is_valid']}")
    print(f"  passed: {d['checks_passed']}, failed: {d['checks_failed']}")
    for issue in d["issues"][:3]:
        print(f"  {issue['severity']}: {issue['message']}")


if __name__ == "__main__":
    main()
