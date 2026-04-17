# 第 6 章 · 01 — 数据校验器 (Data Validator)

> **目标**: 构建企业级数据校验流水线，在数据入库前拦截所有常见问题。

## 核心知识点

1. NaN / Inf 检测
2. 四元数归一化检查
3. 关节限位违规检测
4. 帧间跳变检测
5. 轨迹长度校验
6. 死关节检测
7. 时间戳单调性检查
8. 动作范围检查
9. 图像维度校验
10. 批量校验 & 汇总报告

---

## 1. 架构设计

### 严重级别

```python
class Severity(Enum):
    ERROR   = "ERROR"     # 致命错误，数据不可用
    WARNING = "WARNING"   # 可能有问题，建议人工审查
    INFO    = "INFO"      # 仅记录，不影响使用
```

### 校验结果

```python
@dataclass
class ValidationResult:
    check_name: str           # 校验项名称
    passed: bool              # 是否通过
    severity: Severity        # 严重级别
    message: str              # 人类可读说明
    details: Dict[str, Any]   # 具体细节
```

### 校验报告

```python
@dataclass
class ValidationReport:
    episode_id: str
    results: List[ValidationResult]

    @property
    def passed(self) -> bool:
        """只有所有 ERROR 级别都通过，才算整体通过"""
        return all(r.passed for r in self.results if r.severity == Severity.ERROR)
```

---

## 2. 九项校验详解

### 2.1 NaN / Inf 检测

递归扫描所有浮点数组，检查是否存在 `NaN` 或 `Inf`。

| 来源 | 后果 |
| :--- | :--- |
| 除零、数值溢出 | 归一化崩溃 |
| 传感器故障 | 损失函数变 NaN |

级别: **ERROR**

### 2.2 四元数归一化检查

检查四元数 `(w, x, y, z)` 的模是否为 1：\(\|q\| \approx 1\)

```python
norms = np.linalg.norm(quats, axis=1)
bad_frames = np.where(np.abs(norms - 1.0) > 1e-3)[0]
```

级别: **ERROR**（非归一化四元数 → 旋转矩阵不正交）

### 2.3 关节限位违规检测

检查 `qpos` 是否超出关节物理限位 `[lower, upper]`。

级别: **ERROR**（超限数据 = 标定错误 / 传感器漂移 / 仿真不稳定）

### 2.4 帧间跳变检测

计算相邻帧差分 `|qpos[t+1] - qpos[t]|`，检查是否超过阈值。

```python
diffs = np.abs(np.diff(qpos, axis=0))
jump_frames = np.where(diffs.max(axis=1) > max_jump)[0]
```

级别: **WARNING**（物理系统有惯性，关节位置不可能在一帧内突变）

### 2.5 轨迹长度校验

```
太短 (< min_len) → ERROR: 可能是采集中断
太长 (> max_len) → WARNING: 可能包含多次尝试未被正确分割
```

### 2.6 死关节检测

```python
stds = qpos.std(axis=0)
dead_joints = np.where(stds < 1e-8)[0]
```

级别: **WARNING**（传感器失效 / 控制信号未送达 / 关节被机械锁定）

### 2.7 时间戳单调性检查

物理时间不可回退：

```python
diffs = np.diff(timestamps)
non_increasing = np.where(diffs <= 0)[0]
```

级别: **ERROR**（时间戳乱序 = 数据拼接错误）

### 2.8 动作范围检查

检查 action 是否在 `[lower, upper]` 内。默认检查 `[-1, 1]` 范围（归一化动作空间）。

级别: **WARNING**（超限动作在真实机器人上会被截断）

### 2.9 图像维度校验

检查图像数据的 `(T, H, W, C)` 维度是否一致且符合预期。

级别: **ERROR**（分辨率不一致 → 批处理报错）

---

## 3. DataValidator 使用方式

```python
config = {
    "joint_limits_lower": [-np.pi] * 7,
    "joint_limits_upper": [np.pi] * 7,
    "action_limits_lower": [-1.0] * 7,
    "action_limits_upper": [1.0] * 7,
    "max_frame_jump": 0.5,
    "min_trajectory_length": 10,
    "max_trajectory_length": 5000,
    "constant_threshold": 1e-8,
    "quaternion_joint_indices": [3],  # free joint 的四元数起始索引
    "expected_image_shape": (480, 640, 3),
    "hz": 50,
}

validator = DataValidator(config)
report = validator.validate(episode_data, episode_id="ep_001")
report.print_summary()
```

---

## 4. 批量校验

```python
batch_validator = BatchValidator(validator)
summary, reports = batch_validator.validate_episodes([
    ("ep_001", data1), ("ep_002", data2), ...
])
```

汇总报告包含：
- 总通过率
- 每种检查项的通过率
- 每个 episode 的错误数和警告数

---

## 5. 自定义扩展

通过继承 `DataValidator` 添加新检查：

```python
class ExtendedValidator(DataValidator):
    def check_velocity_smoothness(self, data):
        # 自定义检查逻辑
        ...

    def validate(self, data, episode_id="unknown"):
        report = super().validate(data, episode_id)
        report.add(self.check_velocity_smoothness(data))
        return report
```

---

## 6. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              数据校验器 — 核心要点                              │
│                                                              │
│  9 项标准检查:                                                │
│    ERROR 级: NaN/Inf、四元数归一化、关节限位、轨迹长度、        │
│              时间戳单调性、图像维度                             │
│    WARNING 级: 帧间跳变、死关节、动作范围                      │
│                                                              │
│  设计原则:                                                    │
│    • 只有 ERROR 全部通过才算合格                               │
│    • WARNING 建议人工审查但不阻断入库                          │
│    • 每条结果包含 details，可用于追查问题根因                   │
│    • 批量校验生成汇总通过率报告                                │
│    • 可继承扩展自定义检查                                     │
└──────────────────────────────────────────────────────────────┘
```
