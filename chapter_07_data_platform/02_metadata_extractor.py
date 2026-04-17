"""
第 7 章 · 02 - 元数据提取器 (Metadata Extractor)

目标: 从 HDF5 / PKL / XML 等原始文件中自动提取结构化元数据，
     填充到 01_schema_design.sql 定义的数据库表中。

核心知识点:
  1. HDF5 文件元数据提取 —— 读取数据集结构、属性、数据形状
  2. PKL 文件元数据提取 —— 反序列化并提取关键字段
  3. MuJoCo 模型文件解析 —— 从 XML/URDF 中提取关节结构
  4. JointSchema 自动构建 —— 对应 joint_schemas 表
  5. 文件校验和计算 —— SHA-256 指纹
  6. 质量指标快速提取 —— NaN、跳变、范围
  7. 批量提取 —— 扫描目录，批量处理

设计模式（Java 后端工程师注意）:
  - Strategy 模式: 不同文件格式使用不同的提取策略
  - Template Method: 提取流程固定，具体步骤可覆写
  - Builder 模式: JointSchema 的逐步构建

运行: python 02_metadata_extractor.py
依赖: pip install numpy (h5py 可选)
"""

import numpy as np
import os
import json
import hashlib
import pickle
import time
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from xml.etree import ElementTree as ET

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


# ============================================================
# 数据模型（对应 SQL Schema 中的表结构）
# ============================================================

@dataclass
class JointInfo:
    """
    单个关节的信息，对应 joint_schemas 表的一行。

    Java 等价:
        @Data
        public class JointInfo {
            private int jointIndex;
            private String jointName;
            private String jointType;  // "free", "ball", "slide", "hinge"
            private int qposStart;
            private int qposDim;
            private Double rangeLow;
            private Double rangeHigh;
        }
    """
    joint_index: int
    joint_name: str
    joint_type: str          # "free", "ball", "slide", "hinge"
    qpos_start: int
    qpos_dim: int
    range_low: Optional[float] = None
    range_high: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JointSchema:
    """
    一种机器人的完整关节结构。
    等价于 joint_schemas 表中 robot_type 相同的所有行。
    """
    robot_type: str
    joints: List[JointInfo] = field(default_factory=list)

    @property
    def nq(self) -> int:
        """广义坐标总维度"""
        if not self.joints:
            return 0
        last = self.joints[-1]
        return last.qpos_start + last.qpos_dim

    @property
    def joint_names(self) -> List[str]:
        return [j.joint_name for j in self.joints]

    def to_db_rows(self) -> List[Dict]:
        """转换为可直接插入数据库的行列表"""
        rows = []
        for j in self.joints:
            row = asdict(j)
            row["robot_type"] = self.robot_type
            rows.append(row)
        return rows


@dataclass
class EpisodeMetadata:
    """
    单条轨迹的元数据，对应 episodes 表的一行。
    这是元数据提取器的核心输出。
    """
    file_path: str
    file_format: str = "hdf5"
    nq: Optional[int] = None
    nv: Optional[int] = None
    nu: Optional[int] = None
    num_steps: Optional[int] = None
    timestep: Optional[float] = None
    duration: Optional[float] = None
    has_nan: bool = False
    has_jumps: bool = False
    quality_score: Optional[float] = None
    qpos_range: List[Dict] = field(default_factory=list)
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_db_row(self) -> Dict:
        """转换为可直接插入数据库的字典"""
        row = asdict(self)
        row["qpos_range"] = json.dumps(row["qpos_range"])
        row.pop("extra", None)
        return row


@dataclass
class QualityMetrics:
    """
    质量指标，对应 quality_reports 表。
    """
    overall_score: float = 1.0
    nan_count: int = 0
    jump_count: int = 0
    limit_violations: int = 0
    dead_joints: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 文件格式提取策略（Strategy 模式）
# ============================================================

class FormatExtractor(ABC):
    """
    文件格式提取器的抽象基类。

    Java 等价:
        public interface FormatExtractor {
            boolean canHandle(String filePath);
            EpisodeMetadata extract(String filePath);
        }
    """

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """判断是否能处理该文件格式"""
        ...

    @abstractmethod
    def extract(self, file_path: str) -> EpisodeMetadata:
        """提取元数据"""
        ...


class HDF5Extractor(FormatExtractor):
    """
    HDF5 文件元数据提取器。

    HDF5 是机器人数据最常用的存储格式（第 5 章详解）。
    文件结构通常为:
        /qpos      shape=(num_steps, nq)
        /qvel      shape=(num_steps, nv)
        /action    shape=(num_steps, nu)
        /images/   可选的图像数据组
    """

    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.hdf5', '.h5'))

    def extract(self, file_path: str) -> EpisodeMetadata:
        meta = EpisodeMetadata(
            file_path=file_path,
            file_format="hdf5",
        )

        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                meta = self._extract_from_h5(f, meta)
        except ImportError:
            # h5py 未安装，尝试从文件名推断基本信息
            meta.extra["note"] = "h5py not installed, metadata is limited"
        except Exception as e:
            meta.extra["error"] = str(e)

        return meta

    def _extract_from_h5(self, f, meta: EpisodeMetadata) -> EpisodeMetadata:
        """从打开的 HDF5 文件中提取元数据"""
        # 提取 qpos
        if "qpos" in f:
            qpos = f["qpos"]
            meta.num_steps = qpos.shape[0]
            meta.nq = qpos.shape[1] if len(qpos.shape) > 1 else 1
            data = qpos[:]
            meta.has_nan = bool(np.isnan(data).any())
            meta.qpos_range = self._compute_qpos_range(data)

        # 提取 qvel
        if "qvel" in f:
            meta.nv = f["qvel"].shape[1] if len(f["qvel"].shape) > 1 else 1

        # 提取 action
        if "action" in f:
            meta.nu = f["action"].shape[1] if len(f["action"].shape) > 1 else 1

        # 提取时间步长（从属性中读取）
        if "timestep" in f.attrs:
            meta.timestep = float(f.attrs["timestep"])
        elif meta.num_steps:
            meta.timestep = 0.02  # 默认 50Hz

        if meta.num_steps and meta.timestep:
            meta.duration = meta.num_steps * meta.timestep

        # 列出所有数据集
        meta.extra["datasets"] = list(f.keys())

        return meta

    @staticmethod
    def _compute_qpos_range(data: np.ndarray) -> List[Dict]:
        """计算每个关节维度的 qpos 范围"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        ranges = []
        for i in range(data.shape[1]):
            col = data[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                ranges.append({
                    "dim": i,
                    "min": float(np.min(valid)),
                    "max": float(np.max(valid)),
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                })
        return ranges


class PKLExtractor(FormatExtractor):
    """
    PKL (Pickle) 文件元数据提取器。

    PKL 文件通常存储为字典:
        {"qpos": np.array, "qvel": np.array, "action": np.array, ...}
    """

    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.pkl', '.pickle'))

    def extract(self, file_path: str) -> EpisodeMetadata:
        meta = EpisodeMetadata(
            file_path=file_path,
            file_format="pkl",
        )

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                meta = self._extract_from_dict(data, meta)
            elif isinstance(data, np.ndarray):
                meta.num_steps = data.shape[0]
                meta.nq = data.shape[1] if data.ndim > 1 else 1
                meta.has_nan = bool(np.isnan(data).any())
            else:
                meta.extra["data_type"] = type(data).__name__

        except Exception as e:
            meta.extra["error"] = str(e)

        return meta

    def _extract_from_dict(self, data: dict, meta: EpisodeMetadata) -> EpisodeMetadata:
        """从字典中提取元数据"""
        # 尝试多种常见的键名
        qpos_keys = ["qpos", "observations/qpos", "joint_positions"]
        for key in qpos_keys:
            if key in data:
                qpos = np.asarray(data[key])
                meta.num_steps = qpos.shape[0]
                meta.nq = qpos.shape[1] if qpos.ndim > 1 else 1
                meta.has_nan = bool(np.isnan(qpos).any())
                meta.qpos_range = HDF5Extractor._compute_qpos_range(qpos)
                break

        qvel_keys = ["qvel", "observations/qvel", "joint_velocities"]
        for key in qvel_keys:
            if key in data:
                meta.nv = np.asarray(data[key]).shape[1] if np.asarray(data[key]).ndim > 1 else 1
                break

        action_keys = ["action", "actions", "ctrl"]
        for key in action_keys:
            if key in data:
                meta.nu = np.asarray(data[key]).shape[1] if np.asarray(data[key]).ndim > 1 else 1
                break

        if "timestep" in data:
            meta.timestep = float(data["timestep"])
        elif meta.num_steps:
            meta.timestep = 0.02

        if meta.num_steps and meta.timestep:
            meta.duration = meta.num_steps * meta.timestep

        meta.extra["keys"] = list(data.keys())
        return meta


class ModelExtractor:
    """
    MuJoCo 模型文件 (XML/URDF) 元数据提取器。

    从模型文件中提取关节结构，构建 JointSchema。
    这不是轨迹文件的提取器，而是模型级别的解析工具。
    """

    # MuJoCo 关节类型到 qpos 维度的映射
    JOINT_TYPE_TO_DIM = {
        "free": 7,     # 位置(3) + 四元数(4)
        "ball": 4,     # 四元数(4)
        "slide": 1,    # 标量位移
        "hinge": 1,    # 标量角度
    }

    def extract_from_xml(self, xml_path: str, robot_type: str = "unknown") -> JointSchema:
        """
        从 MJCF XML 文件中提取关节结构。

        MJCF 中关节定义在 <body> 的 <joint> 子元素里:
            <body name="link1">
                <joint name="joint1" type="hinge" range="-3.14 3.14"/>
                ...
            </body>
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        joints = []
        qpos_offset = 0

        # 递归查找所有 <joint> 元素
        for joint_elem in root.iter("joint"):
            name = joint_elem.get("name", f"joint_{len(joints)}")
            jtype = joint_elem.get("type", "hinge")  # MuJoCo 默认类型是 hinge
            dim = self.JOINT_TYPE_TO_DIM.get(jtype, 1)

            range_str = joint_elem.get("range", "")
            range_low, range_high = None, None
            if range_str:
                parts = range_str.split()
                if len(parts) == 2:
                    range_low = float(parts[0])
                    range_high = float(parts[1])

            joint_info = JointInfo(
                joint_index=len(joints),
                joint_name=name,
                joint_type=jtype,
                qpos_start=qpos_offset,
                qpos_dim=dim,
                range_low=range_low,
                range_high=range_high,
                metadata={
                    "damping": joint_elem.get("damping", ""),
                    "armature": joint_elem.get("armature", ""),
                },
            )
            joints.append(joint_info)
            qpos_offset += dim

        return JointSchema(robot_type=robot_type, joints=joints)

    def extract_from_xml_string(self, xml_string: str, robot_type: str = "unknown") -> JointSchema:
        """从 XML 字符串中提取（无需文件）"""
        root = ET.fromstring(xml_string)
        joints = []
        qpos_offset = 0

        for joint_elem in root.iter("joint"):
            name = joint_elem.get("name", f"joint_{len(joints)}")
            jtype = joint_elem.get("type", "hinge")
            dim = self.JOINT_TYPE_TO_DIM.get(jtype, 1)

            range_str = joint_elem.get("range", "")
            range_low, range_high = None, None
            if range_str:
                parts = range_str.split()
                if len(parts) == 2:
                    range_low = float(parts[0])
                    range_high = float(parts[1])

            joints.append(JointInfo(
                joint_index=len(joints),
                joint_name=name,
                joint_type=jtype,
                qpos_start=qpos_offset,
                qpos_dim=dim,
                range_low=range_low,
                range_high=range_high,
            ))
            qpos_offset += dim

        return JointSchema(robot_type=robot_type, joints=joints)


# ============================================================
# 工具函数
# ============================================================

def compute_checksum(file_path: str, algorithm: str = "sha256") -> str:
    """
    计算文件的校验和（SHA-256）。

    用途: 检测文件是否被修改、去重、数据完整性验证。
    Java 等价: MessageDigest.getInstance("SHA-256")
    """
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_format(file_path: str) -> str:
    """
    根据文件扩展名自动检测格式。

    支持的格式: hdf5, pkl, xml, urdf, npz
    """
    ext = Path(file_path).suffix.lower()
    format_map = {
        ".hdf5": "hdf5",
        ".h5": "hdf5",
        ".pkl": "pkl",
        ".pickle": "pkl",
        ".xml": "xml",
        ".urdf": "urdf",
        ".npz": "npz",
    }
    return format_map.get(ext, "unknown")


def compute_quality_metrics(
    qpos: np.ndarray,
    joint_limits: Optional[List[Tuple[float, float]]] = None,
    jump_threshold: float = 1.0,
) -> QualityMetrics:
    """
    从 qpos 数据中快速计算质量指标。

    这是第 6 章校验逻辑的精简版，用于元数据入库时的快速评估。
    """
    metrics = QualityMetrics()

    if qpos.ndim == 1:
        qpos = qpos.reshape(-1, 1)

    # NaN 检测
    nan_mask = np.isnan(qpos)
    metrics.nan_count = int(np.sum(nan_mask))

    # 帧间跳变检测
    if qpos.shape[0] > 1:
        diffs = np.abs(np.diff(qpos, axis=0))
        valid_diffs = diffs[~np.isnan(diffs)]
        if len(valid_diffs) > 0:
            metrics.jump_count = int(np.sum(valid_diffs > jump_threshold))

    # 关节限位违规检测
    if joint_limits:
        for i, (low, high) in enumerate(joint_limits):
            if i < qpos.shape[1]:
                col = qpos[:, i]
                valid = col[~np.isnan(col)]
                violations = np.sum((valid < low) | (valid > high))
                metrics.limit_violations += int(violations)

    # 死关节检测（标准差接近 0）
    for i in range(qpos.shape[1]):
        col = qpos[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) > 1 and np.std(valid) < 1e-8:
            metrics.dead_joints += 1

    # 综合评分（简化版）
    score = 1.0
    if metrics.nan_count > 0:
        score -= 0.3
    if metrics.jump_count > 10:
        score -= 0.2
    elif metrics.jump_count > 0:
        score -= 0.1
    if metrics.limit_violations > 0:
        score -= 0.2
    if metrics.dead_joints > qpos.shape[1] * 0.5:
        score -= 0.2

    metrics.overall_score = max(0.0, score)
    return metrics


# ============================================================
# MetadataExtractor 主类
# ============================================================

class MetadataExtractor:
    """
    元数据提取器 —— 整个模块的入口。

    职责:
      1. 自动检测文件格式
      2. 委派给对应的 FormatExtractor
      3. 计算文件校验和
      4. 提取质量指标
      5. 输出结构化元数据

    Java 等价:
        @Service
        public class MetadataExtractorService {
            @Autowired
            private List<FormatExtractor> extractors;

            public EpisodeMetadata extract(String filePath) { ... }
        }
    """

    def __init__(self):
        # 注册所有格式提取器（类似 Spring 的自动注入列表）
        self._extractors: List[FormatExtractor] = [
            HDF5Extractor(),
            PKLExtractor(),
        ]
        self._model_extractor = ModelExtractor()

    def register_extractor(self, extractor: FormatExtractor):
        """注册自定义的格式提取器（开放扩展）"""
        self._extractors.append(extractor)

    def extract(self, file_path: str, compute_checksum_flag: bool = True) -> EpisodeMetadata:
        """
        从单个文件中提取元数据。

        流程:
          1. 检测文件格式
          2. 选择对应的提取器
          3. 执行提取
          4. 计算文件校验和
          5. 计算文件大小
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 查找能处理此格式的提取器
        extractor = self._find_extractor(file_path)
        if extractor is None:
            # 无法识别的格式，返回基本文件信息
            meta = EpisodeMetadata(
                file_path=file_path,
                file_format=detect_format(file_path),
            )
        else:
            meta = extractor.extract(file_path)

        # 补充文件级元数据
        meta.file_size = os.path.getsize(file_path)
        if compute_checksum_flag:
            meta.checksum = compute_checksum(file_path)

        return meta

    def extract_with_quality(
        self,
        file_path: str,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[EpisodeMetadata, QualityMetrics]:
        """
        提取元数据并计算质量指标（一步到位）。
        """
        meta = self.extract(file_path)

        # 尝试加载 qpos 数据来计算质量
        quality = QualityMetrics()
        try:
            qpos = self._load_qpos(file_path)
            if qpos is not None:
                quality = compute_quality_metrics(qpos, joint_limits)
                meta.has_nan = quality.nan_count > 0
                meta.has_jumps = quality.jump_count > 0
                meta.quality_score = quality.overall_score
        except Exception:
            pass

        return meta, quality

    def extract_model(self, xml_path: str, robot_type: str = "unknown") -> JointSchema:
        """从模型文件中提取关节结构"""
        return self._model_extractor.extract_from_xml(xml_path, robot_type)

    def extract_model_from_string(self, xml_string: str, robot_type: str = "unknown") -> JointSchema:
        """从 XML 字符串中提取关节结构"""
        return self._model_extractor.extract_from_xml_string(xml_string, robot_type)

    def batch_extract(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[EpisodeMetadata]:
        """
        批量提取目录下所有文件的元数据。

        参数:
            directory: 要扫描的目录路径
            extensions: 要处理的文件扩展名列表（默认 [".hdf5", ".pkl"]）
            recursive: 是否递归扫描子目录
        """
        if extensions is None:
            extensions = [".hdf5", ".h5", ".pkl", ".pickle"]

        target = Path(directory)
        if not target.is_dir():
            raise NotADirectoryError(f"目录不存在: {directory}")

        # 收集所有匹配的文件
        files = []
        for ext in extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(target.glob(pattern))

        results = []
        for i, fp in enumerate(sorted(files)):
            try:
                meta = self.extract(str(fp))
                results.append(meta)
            except Exception as e:
                # 记录错误但继续处理其他文件
                error_meta = EpisodeMetadata(
                    file_path=str(fp),
                    file_format=detect_format(str(fp)),
                )
                error_meta.extra["error"] = str(e)
                results.append(error_meta)

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(files)} 个文件...")

        return results

    def _find_extractor(self, file_path: str) -> Optional[FormatExtractor]:
        """查找能处理指定文件的提取器"""
        for extractor in self._extractors:
            if extractor.can_handle(file_path):
                return extractor
        return None

    def _load_qpos(self, file_path: str) -> Optional[np.ndarray]:
        """从文件中加载 qpos 数据（用于质量计算）"""
        fmt = detect_format(file_path)

        if fmt == "pkl":
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                for key in ["qpos", "observations/qpos", "joint_positions"]:
                    if key in data:
                        return np.asarray(data[key])
            elif isinstance(data, np.ndarray):
                return data

        elif fmt == "hdf5":
            try:
                import h5py
                with h5py.File(file_path, 'r') as f:
                    if "qpos" in f:
                        return f["qpos"][:]
            except ImportError:
                pass

        return None


# ============================================================
# 测试数据生成
# ============================================================

def generate_test_pkl(path: str, num_steps: int = 200, nq: int = 14, inject_nan: bool = False):
    """生成测试用的 PKL 轨迹文件"""
    qpos = np.random.randn(num_steps, nq) * 0.5
    qvel = np.random.randn(num_steps, nq) * 0.1
    action = np.random.randn(num_steps, nq) * 0.3

    if inject_nan:
        qpos[50, 3] = np.nan
        qpos[51, 3] = np.nan

    data = {
        "qpos": qpos,
        "qvel": qvel,
        "action": action,
        "timestep": 0.02,
    }

    with open(path, 'wb') as f:
        pickle.dump(data, f)


def generate_test_xml(path: str, robot_type: str = "test_arm"):
    """生成测试用的 MJCF XML 模型文件"""
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="{robot_type}">
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="shoulder_yaw" type="hinge" range="-3.14159 3.14159"/>
      <geom type="cylinder" size="0.05 0.2"/>
      <body name="upper_arm" pos="0 0 0.4">
        <joint name="shoulder_pitch" type="hinge" range="-1.57 1.57"/>
        <geom type="cylinder" size="0.04 0.15"/>
        <body name="forearm" pos="0 0 0.3">
          <joint name="elbow" type="hinge" range="-2.35 0"/>
          <geom type="cylinder" size="0.035 0.12"/>
          <body name="wrist" pos="0 0 0.24">
            <joint name="wrist_roll" type="hinge" range="-3.14159 3.14159"/>
            <joint name="wrist_pitch" type="hinge" range="-1.57 1.57"/>
            <joint name="wrist_yaw" type="hinge" range="-3.14159 3.14159"/>
            <geom type="sphere" size="0.03"/>
            <body name="gripper" pos="0 0 0.05">
              <joint name="gripper" type="slide" range="0 0.04"/>
              <geom type="box" size="0.02 0.04 0.01"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder_yaw" ctrlrange="-1 1"/>
    <motor joint="shoulder_pitch" ctrlrange="-1 1"/>
    <motor joint="elbow" ctrlrange="-1 1"/>
    <motor joint="wrist_roll" ctrlrange="-1 1"/>
    <motor joint="wrist_pitch" ctrlrange="-1 1"/>
    <motor joint="wrist_yaw" ctrlrange="-1 1"/>
    <motor joint="gripper" ctrlrange="0 1"/>
  </actuator>
</mujoco>"""

    with open(path, 'w') as f:
        f.write(xml_content)


# ============================================================
# 主函数 —— 演示完整流程
# ============================================================

def main():
    print(DIVIDER)
    print("第 7 章 · 02 - 元数据提取器 (Metadata Extractor)")
    print(DIVIDER)

    # 创建临时目录存放测试数据
    with tempfile.TemporaryDirectory(prefix="ch07_extractor_") as tmpdir:

        # --------------------------------------------------
        # 1. 生成测试数据
        # --------------------------------------------------
        print(f"\n{'1. 生成测试数据':=^60}")
        print(f"临时目录: {tmpdir}\n")

        # 生成 PKL 文件
        pkl_files = []
        for i in range(5):
            path = os.path.join(tmpdir, f"episode_{i:04d}.pkl")
            inject_nan = (i == 2)  # 第 3 个文件注入 NaN
            generate_test_pkl(path, num_steps=100 + i * 50, nq=7, inject_nan=inject_nan)
            pkl_files.append(path)
            print(f"  ✓ 生成 episode_{i:04d}.pkl (steps={100 + i * 50}, nan={inject_nan})")

        # 生成 XML 模型文件
        xml_path = os.path.join(tmpdir, "test_arm.xml")
        generate_test_xml(xml_path, "test_arm_7dof")
        print(f"  ✓ 生成 test_arm.xml (7 关节模型)")

        # --------------------------------------------------
        # 2. 单文件提取
        # --------------------------------------------------
        print(f"\n{'2. 单文件元数据提取':=^60}")

        extractor = MetadataExtractor()
        meta = extractor.extract(pkl_files[0])

        print(f"\n文件: {os.path.basename(meta.file_path)}")
        print(f"  格式: {meta.file_format}")
        print(f"  nq={meta.nq}, nv={meta.nv}, nu={meta.nu}")
        print(f"  帧数: {meta.num_steps}")
        print(f"  时长: {meta.duration:.2f}s")
        print(f"  文件大小: {meta.file_size:,} bytes")
        print(f"  校验和: {meta.checksum[:16]}...")

        # 展示 qpos_range
        if meta.qpos_range:
            print(f"  qpos 范围 (前 3 维):")
            for r in meta.qpos_range[:3]:
                print(f"    dim {r['dim']}: [{r['min']:.4f}, {r['max']:.4f}] "
                      f"mean={r['mean']:.4f} std={r['std']:.4f}")

        # --------------------------------------------------
        # 3. 提取 + 质量评估
        # --------------------------------------------------
        print(f"\n{'3. 提取 + 质量评估':=^60}")

        # 提取正常文件
        meta_good, quality_good = extractor.extract_with_quality(pkl_files[0])
        print(f"\n正常文件: {os.path.basename(pkl_files[0])}")
        print(f"  质量评分: {quality_good.overall_score:.2f}")
        print(f"  NaN 数量: {quality_good.nan_count}")
        print(f"  跳变次数: {quality_good.jump_count}")
        print(f"  限位违规: {quality_good.limit_violations}")

        # 提取含 NaN 的文件
        meta_bad, quality_bad = extractor.extract_with_quality(pkl_files[2])
        print(f"\n含 NaN 文件: {os.path.basename(pkl_files[2])}")
        print(f"  质量评分: {quality_bad.overall_score:.2f}")
        print(f"  NaN 数量: {quality_bad.nan_count}")
        print(f"  has_nan: {meta_bad.has_nan}")

        # --------------------------------------------------
        # 4. 模型文件解析（JointSchema 构建）
        # --------------------------------------------------
        print(f"\n{'4. 模型文件解析 → JointSchema':=^60}")

        schema = extractor.extract_model(xml_path, robot_type="test_arm_7dof")
        print(f"\n机器人类型: {schema.robot_type}")
        print(f"总关节数: {len(schema.joints)}")
        print(f"nq（广义坐标维度）: {schema.nq}")
        print(f"\n{'idx':<4} {'名称':<20} {'类型':<8} {'qpos_start':<12} {'qpos_dim':<10} {'范围'}")
        print("-" * 75)
        for j in schema.joints:
            range_str = f"[{j.range_low:.2f}, {j.range_high:.2f}]" if j.range_low is not None else "无限位"
            print(f"{j.joint_index:<4} {j.joint_name:<20} {j.joint_type:<8} "
                  f"{j.qpos_start:<12} {j.qpos_dim:<10} {range_str}")

        # 展示数据库行格式
        print(f"\n转换为数据库行 (前 3 行):")
        for row in schema.to_db_rows()[:3]:
            compact = {k: v for k, v in row.items() if k != "metadata"}
            print(f"  {json.dumps(compact, ensure_ascii=False)}")

        # --------------------------------------------------
        # 5. 从 XML 字符串直接提取
        # --------------------------------------------------
        print(f"\n{'5. 从 XML 字符串提取 (不需要文件)':=^60}")

        simple_xml = """
        <mujoco model="simple_gripper">
          <worldbody>
            <body name="base">
              <body name="floating_base">
                <joint name="root" type="free"/>
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="finger_left">
                  <joint name="finger_l" type="slide" range="0 0.05"/>
                  <geom type="box" size="0.01 0.01 0.05"/>
                </body>
                <body name="finger_right">
                  <joint name="finger_r" type="slide" range="0 0.05"/>
                  <geom type="box" size="0.01 0.01 0.05"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """

        schema2 = extractor.extract_model_from_string(simple_xml, "floating_gripper")
        print(f"\n机器人: {schema2.robot_type}")
        print(f"nq = {schema2.nq}  (free=7 + slide=1 + slide=1 = 9)")
        for j in schema2.joints:
            print(f"  关节 {j.joint_index}: {j.joint_name} ({j.joint_type}) "
                  f"qpos[{j.qpos_start}:{j.qpos_start + j.qpos_dim}]")

        # --------------------------------------------------
        # 6. 批量提取
        # --------------------------------------------------
        print(f"\n{'6. 批量提取':=^60}")

        results = extractor.batch_extract(tmpdir, extensions=[".pkl"])
        print(f"\n扫描目录: {tmpdir}")
        print(f"发现文件: {len(results)} 个\n")

        print(f"{'文件':<25} {'帧数':<8} {'nq':<5} {'NaN':<6} {'大小':<12} {'校验和'}")
        print("-" * 80)
        for meta in results:
            name = os.path.basename(meta.file_path)
            size_str = f"{meta.file_size:,}" if meta.file_size else "?"
            cksum = meta.checksum[:12] + "..." if meta.checksum else "?"
            print(f"{name:<25} {meta.num_steps or '?':<8} {meta.nq or '?':<5} "
                  f"{meta.has_nan!s:<6} {size_str:<12} {cksum}")

        # --------------------------------------------------
        # 7. 输出为 SQL 兼容格式
        # --------------------------------------------------
        print(f"\n{'7. 转换为 SQL 插入格式':=^60}")

        meta = results[0]
        db_row = meta.to_db_row()
        print(f"\nepisodes 表插入数据:")
        for key, value in db_row.items():
            print(f"  {key}: {value}")

        # --------------------------------------------------
        # 8. 输出为 JSON（API 返回格式）
        # --------------------------------------------------
        print(f"\n{'8. 输出为 JSON (API 格式)':=^60}")

        meta_dict = asdict(results[0])
        meta_dict.pop("extra", None)
        print(json.dumps(meta_dict, indent=2, ensure_ascii=False, default=str)[:500])

    print(f"\n{DIVIDER}")
    print("✅ 元数据提取器演示完成！")
    print(f"""
关键收获:
  1. 不同文件格式用 Strategy 模式处理，新增格式只需实现 FormatExtractor 接口
  2. JointSchema 是连接「模型结构」和「数据理解」的桥梁
  3. 校验和（checksum）是数据完整性的最后防线
  4. 批量提取是构建数据平台的基础 —— 把非结构化的文件变成结构化的数据库记录
  5. 所有输出都对齐 SQL Schema，一步到位插入数据库
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
