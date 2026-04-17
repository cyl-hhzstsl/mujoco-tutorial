# 第 7 章 · 02 — 元数据提取器 (Metadata Extractor)

> **目标**: 从 HDF5 / PKL / XML 等原始文件中自动提取结构化元数据，填充到数据库表中。

## 核心知识点

1. HDF5 文件元数据提取
2. PKL 文件元数据提取
3. MuJoCo 模型文件 (XML) 解析 → JointSchema
4. 文件校验和 (SHA-256)
5. 质量指标快速计算
6. 批量提取

---

## 1. 数据模型

### JointInfo → joint_schemas 表的一行

```python
@dataclass
class JointInfo:
    joint_index: int
    joint_name: str
    joint_type: str       # "free", "ball", "slide", "hinge"
    qpos_start: int
    qpos_dim: int
    range_low: Optional[float]
    range_high: Optional[float]
```

### EpisodeMetadata → episodes 表的一行

```python
@dataclass
class EpisodeMetadata:
    file_path: str
    file_format: str       # "hdf5", "pkl"
    nq, nv, nu: int
    num_steps: int
    has_nan, has_jumps: bool
    quality_score: float
    qpos_range: List[Dict]
    file_size: int
    checksum: str
```

### QualityMetrics → quality_reports 表

```python
@dataclass
class QualityMetrics:
    overall_score: float
    nan_count: int
    jump_count: int
    limit_violations: int
    dead_joints: int
```

---

## 2. Strategy 模式 — 格式提取器

不同文件格式使用不同的提取策略：

```
FormatExtractor (ABC)
  ├── HDF5Extractor    → .hdf5, .h5
  └── PKLExtractor     → .pkl, .pickle
```

每个提取器实现两个方法：
- `can_handle(file_path)` — 判断是否能处理
- `extract(file_path)` → `EpisodeMetadata`

### HDF5 提取的典型字段

```
/qpos      → shape=(num_steps, nq)
/qvel      → shape=(num_steps, nv)
/action    → shape=(num_steps, nu)
attrs["timestep"] → 仿真步长
```

### PKL 提取

尝试多种常见键名：`qpos`, `observations/qpos`, `joint_positions`

---

## 3. ModelExtractor — 从 XML 构建 JointSchema

从 MJCF XML 文件中递归查找 `<joint>` 元素，构建关节结构：

```python
JOINT_TYPE_TO_DIM = {
    "free": 7,   # 位置(3) + 四元数(4)
    "ball": 4,   # 四元数(4)
    "slide": 1,
    "hinge": 1,
}
```

输出 `JointSchema` 可直接通过 `to_db_rows()` 插入数据库。

---

## 4. MetadataExtractor 主类

整个模块的入口，整合所有功能：

```python
extractor = MetadataExtractor()

# 单文件提取
meta = extractor.extract("episode_001.pkl")

# 提取 + 质量评估
meta, quality = extractor.extract_with_quality("episode_001.pkl")

# 模型解析
schema = extractor.extract_model("robot.xml", robot_type="aloha")

# 批量提取
results = extractor.batch_extract("/data/raw/", extensions=[".hdf5", ".pkl"])
```

### 提取流程

```
检测格式 → 选择提取器 → 执行提取 → 计算校验和 → 计算文件大小
```

---

## 5. 工具函数

### 文件校验和

```python
def compute_checksum(file_path, algorithm="sha256") -> str
```

用途：检测文件是否被修改、去重、数据完整性验证。

### 质量快速评估

```python
def compute_quality_metrics(qpos, joint_limits, jump_threshold) -> QualityMetrics
```

第 6 章校验逻辑的精简版，用于入库时的快速评估。

评分规则：
- 有 NaN → -0.3
- 跳变 > 10 次 → -0.2
- 限位违规 → -0.2
- 死关节 > 50% → -0.2

---

## 6. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              元数据提取器 — 核心要点                            │
│                                                              │
│  Strategy 模式:                                               │
│    HDF5Extractor / PKLExtractor 处理不同格式                  │
│    新增格式只需实现 FormatExtractor 接口                      │
│                                                              │
│  三种提取目标:                                                │
│    EpisodeMetadata → episodes 表                             │
│    JointSchema → joint_schemas 表                            │
│    QualityMetrics → quality_reports 表                       │
│                                                              │
│  关键能力:                                                    │
│    • SHA-256 校验和 — 数据完整性最后防线                      │
│    • 批量提取 — 扫描目录，批量处理                            │
│    • 所有输出对齐 SQL Schema — 一步到位插入数据库             │
└──────────────────────────────────────────────────────────────┘
```
