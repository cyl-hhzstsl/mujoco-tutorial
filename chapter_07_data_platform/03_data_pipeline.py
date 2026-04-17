"""
第 7 章 · 03 - 数据摄入流水线 (Data Pipeline)

目标: 构建一条完整的数据摄入流水线，将原始文件转化为平台可管理的数据资产。

流水线阶段:
    接收 (Receive) → 校验 (Validate) → 提取元数据 (Extract) → 存储 (Store) → 索引 (Index)

核心知识点:
  1. Pipeline 模式 —— 数据处理的标准架构
  2. 阶段可配置 —— 跳过特定步骤、自定义校验器
  3. 错误处理与重试 —— 瞬时错误自动重试，永久错误跳过并记录
  4. 进度追踪 —— 实时知道处理到哪了
  5. 批量处理 —— 高效处理大批量文件
  6. 存储后端抽象 —— 接口设计适配 S3/MinIO，演示用文件系统

设计模式（Java 后端工程师注意）:
  - Chain of Responsibility: 流水线各阶段串联
  - Strategy: 存储后端可替换（本地 FS → S3）
  - Observer: 进度回调
  - Builder: PipelineConfig 的构建

运行: python 03_data_pipeline.py
依赖: pip install numpy
"""

import numpy as np
import os
import json
import time
import shutil
import pickle
import hashlib
import tempfile
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
from enum import Enum
from datetime import datetime

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


# ============================================================
# 枚举与状态定义
# ============================================================

class PipelineStage(Enum):
    """流水线阶段"""
    RECEIVE = "receive"
    VALIDATE = "validate"
    EXTRACT = "extract_metadata"
    STORE = "store"
    INDEX = "index"


class TaskStatus(Enum):
    """单个文件的处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================
# 数据模型
# ============================================================

@dataclass
class PipelineConfig:
    """
    流水线配置。

    Java 等价:
        @Configuration
        @ConfigurationProperties(prefix = "pipeline")
        public class PipelineConfig { ... }
    """
    # 启用的阶段（可以跳过某些阶段）
    enabled_stages: List[PipelineStage] = field(default_factory=lambda: list(PipelineStage))

    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0           # 秒，每次重试递增（指数退避）

    # 校验配置
    max_nan_ratio: float = 0.01        # NaN 超过 1% 就拒绝
    min_steps: int = 10                # 最少帧数
    max_steps: int = 100000            # 最多帧数

    # 存储配置
    storage_root: str = "/tmp/robot_data_store"
    organize_by: str = "dataset"       # 按数据集分目录

    # 批量处理
    batch_size: int = 50               # 一批处理多少个文件
    stop_on_error: bool = False        # 遇到错误是否停止整个批次

    # 进度回调间隔
    progress_interval: int = 5         # 每处理 N 个文件报告一次进度


@dataclass
class FileTask:
    """
    一个待处理文件的任务单。

    类比: 就像一个工单（ticket），记录这个文件经过流水线每个阶段的状态。
    """
    file_path: str
    dataset_name: str = "default"
    status: TaskStatus = TaskStatus.PENDING
    current_stage: Optional[PipelineStage] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at


@dataclass
class PipelineReport:
    """流水线执行报告"""
    total_files: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"处理文件: {self.total_files}",
            f"  成功: {self.completed}",
            f"  失败: {self.failed}",
            f"  跳过: {self.skipped}",
            f"总耗时: {self.total_time:.2f}s",
        ]
        if self.stage_times:
            lines.append("各阶段耗时:")
            for stage, t in self.stage_times.items():
                lines.append(f"  {stage}: {t:.3f}s")
        if self.errors:
            lines.append(f"错误列表 (共 {len(self.errors)} 个):")
            for err in self.errors[:5]:
                lines.append(f"  {err['file']}: {err['error']}")
            if len(self.errors) > 5:
                lines.append(f"  ... 还有 {len(self.errors) - 5} 个错误")
        return "\n".join(lines)


# ============================================================
# 存储后端接口（Strategy 模式）
# ============================================================

class StorageBackend(ABC):
    """
    存储后端的抽象接口。

    设计意图: 本地开发用文件系统，生产环境替换为 S3 / MinIO。
    只要实现这个接口，流水线不需要改任何代码。

    Java 等价:
        public interface StorageBackend {
            String store(String sourcePath, String targetKey);
            boolean exists(String key);
            void delete(String key);
            Map<String, Object> getInfo(String key);
        }
    """

    @abstractmethod
    def store(self, source_path: str, target_key: str) -> str:
        """存储文件，返回存储后的路径/URL"""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查文件是否已存在"""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除文件"""
        ...

    @abstractmethod
    def get_info(self, key: str) -> Dict[str, Any]:
        """获取文件信息（大小、修改时间等）"""
        ...


class LocalFileSystemStorage(StorageBackend):
    """
    本地文件系统存储后端。

    按 dataset_name/filename 的结构组织文件。
    生产环境可以替换为 S3Storage、MinIOStorage 等实现。
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    def store(self, source_path: str, target_key: str) -> str:
        target_path = os.path.join(self.root_dir, target_key)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)
        return target_path

    def exists(self, key: str) -> bool:
        return os.path.exists(os.path.join(self.root_dir, key))

    def delete(self, key: str) -> bool:
        path = os.path.join(self.root_dir, key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def get_info(self, key: str) -> Dict[str, Any]:
        path = os.path.join(self.root_dir, key)
        if not os.path.exists(path):
            return {"exists": False}
        stat = os.stat(path)
        return {
            "exists": True,
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "path": path,
        }


# ============================================================
# 索引后端接口
# ============================================================

class IndexBackend(ABC):
    """
    搜索索引后端的抽象接口。

    生产环境使用 Elasticsearch / Meilisearch，这里用内存模拟。

    Java 等价:
        public interface IndexBackend {
            void index(String id, Map<String, Object> document);
            List<Map<String, Object>> search(Map<String, Object> query);
        }
    """

    @abstractmethod
    def index(self, doc_id: str, document: Dict[str, Any]) -> bool:
        ...

    @abstractmethod
    def search(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        ...


class InMemoryIndex(IndexBackend):
    """内存搜索索引（演示用）"""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def index(self, doc_id: str, document: Dict[str, Any]) -> bool:
        self._store[doc_id] = document
        return True

    def search(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for doc_id, doc in self._store.items():
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if match:
                results.append({"id": doc_id, **doc})
        return results

    @property
    def count(self) -> int:
        return len(self._store)


# ============================================================
# 模拟数据库（内存版）
# ============================================================

class InMemoryDatabase:
    """
    内存数据库，模拟 PostgreSQL 的 episodes 表。

    生产环境替换为 SQLAlchemy / MyBatis 的 Repository 层。

    Java 等价:
        @Repository
        public interface EpisodeRepository extends JpaRepository<Episode, Long> { ... }
    """

    def __init__(self):
        self._episodes: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def insert_episode(self, data: Dict[str, Any]) -> int:
        episode_id = self._next_id
        self._next_id += 1
        data["id"] = episode_id
        data["created_at"] = datetime.now().isoformat()
        self._episodes[episode_id] = data
        return episode_id

    def get_episode(self, episode_id: int) -> Optional[Dict[str, Any]]:
        return self._episodes.get(episode_id)

    def list_episodes(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        results = list(self._episodes.values())
        if dataset_name:
            results = [e for e in results if e.get("dataset_name") == dataset_name]
        return results

    @property
    def count(self) -> int:
        return len(self._episodes)


# ============================================================
# 流水线各阶段的处理器
# ============================================================

class StageProcessor(ABC):
    """阶段处理器的抽象基类"""

    @abstractmethod
    def process(self, task: FileTask, context: Dict[str, Any]) -> FileTask:
        ...

    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        ...


class ReceiveProcessor(StageProcessor):
    """
    接收阶段: 验证文件存在性和基本信息。
    """

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.RECEIVE

    def process(self, task: FileTask, context: Dict[str, Any]) -> FileTask:
        path = task.file_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        file_size = os.path.getsize(path)
        if file_size == 0:
            raise ValueError(f"文件为空: {path}")

        # 计算校验和
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)

        task.stage_results["receive"] = {
            "file_size": file_size,
            "checksum": h.hexdigest(),
            "file_format": Path(path).suffix.lstrip('.'),
        }
        return task


class ValidateProcessor(StageProcessor):
    """
    校验阶段: 检查数据质量是否达标。
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._custom_validators: List[Callable] = []

    def add_validator(self, validator: Callable[[np.ndarray], Tuple[bool, str]]):
        """添加自定义校验器"""
        self._custom_validators.append(validator)

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.VALIDATE

    def process(self, task: FileTask, context: Dict[str, Any]) -> FileTask:
        path = task.file_path
        issues = []

        # 加载数据
        data = self._load_data(path)
        if data is None:
            raise ValueError(f"无法加载数据: {path}")

        qpos = data.get("qpos")
        if qpos is None:
            raise ValueError(f"文件中没有 qpos 数据: {path}")

        qpos = np.asarray(qpos)

        # 检查帧数
        if qpos.shape[0] < self.config.min_steps:
            issues.append(f"帧数过少: {qpos.shape[0]} < {self.config.min_steps}")
        if qpos.shape[0] > self.config.max_steps:
            issues.append(f"帧数过多: {qpos.shape[0]} > {self.config.max_steps}")

        # 检查 NaN 比例
        nan_ratio = np.isnan(qpos).mean()
        if nan_ratio > self.config.max_nan_ratio:
            issues.append(f"NaN 比例过高: {nan_ratio:.4f} > {self.config.max_nan_ratio}")

        # 运行自定义校验器
        for validator in self._custom_validators:
            try:
                passed, msg = validator(qpos)
                if not passed:
                    issues.append(f"自定义校验失败: {msg}")
            except Exception as e:
                issues.append(f"自定义校验器异常: {e}")

        task.stage_results["validate"] = {
            "passed": len(issues) == 0,
            "issues": issues,
            "nan_ratio": float(nan_ratio),
            "num_steps": int(qpos.shape[0]),
        }

        if issues:
            task.stage_results["validate"]["warning"] = "; ".join(issues)

        return task

    @staticmethod
    def _load_data(file_path: str) -> Optional[Dict]:
        ext = Path(file_path).suffix.lower()
        if ext in ('.pkl', '.pickle'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data if isinstance(data, dict) else None
        return None


class ExtractProcessor(StageProcessor):
    """
    元数据提取阶段: 从数据文件中提取结构化信息。
    """

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.EXTRACT

    def process(self, task: FileTask, context: Dict[str, Any]) -> FileTask:
        path = task.file_path

        data = self._load_data(path)
        if data is None:
            raise ValueError(f"无法加载数据用于元数据提取: {path}")

        metadata = {}

        if "qpos" in data:
            qpos = np.asarray(data["qpos"])
            metadata["nq"] = int(qpos.shape[1]) if qpos.ndim > 1 else 1
            metadata["num_steps"] = int(qpos.shape[0])
            metadata["has_nan"] = bool(np.isnan(qpos).any())

            # 计算 qpos 范围
            ranges = []
            cols = qpos.shape[1] if qpos.ndim > 1 else 1
            if qpos.ndim == 1:
                qpos = qpos.reshape(-1, 1)
            for i in range(cols):
                col = qpos[:, i]
                valid = col[~np.isnan(col)]
                if len(valid) > 0:
                    ranges.append({
                        "dim": i,
                        "min": float(np.min(valid)),
                        "max": float(np.max(valid)),
                    })
            metadata["qpos_range"] = ranges

            # 跳变检测
            if qpos.shape[0] > 1:
                diffs = np.abs(np.diff(qpos, axis=0))
                metadata["has_jumps"] = bool(np.any(diffs > 2.0))
                metadata["max_jump"] = float(np.nanmax(diffs))

        if "qvel" in data:
            qvel = np.asarray(data["qvel"])
            metadata["nv"] = int(qvel.shape[1]) if qvel.ndim > 1 else 1

        if "action" in data:
            action = np.asarray(data["action"])
            metadata["nu"] = int(action.shape[1]) if action.ndim > 1 else 1

        if "timestep" in data:
            metadata["timestep"] = float(data["timestep"])
        else:
            metadata["timestep"] = 0.02

        if "num_steps" in metadata and "timestep" in metadata:
            metadata["duration"] = metadata["num_steps"] * metadata["timestep"]

        # 质量评分（简化版）
        score = 1.0
        if metadata.get("has_nan"):
            score -= 0.3
        if metadata.get("has_jumps"):
            score -= 0.2
        metadata["quality_score"] = max(0.0, score)

        # 合并接收阶段的信息
        receive_info = task.stage_results.get("receive", {})
        metadata["file_size"] = receive_info.get("file_size")
        metadata["checksum"] = receive_info.get("checksum")

        task.stage_results["extract"] = metadata
        task.metadata = metadata
        return task

    @staticmethod
    def _load_data(file_path: str) -> Optional[Dict]:
        ext = Path(file_path).suffix.lower()
        if ext in ('.pkl', '.pickle'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None


class StoreProcessor(StageProcessor):
    """
    存储阶段: 将文件复制到存储后端，将元数据写入数据库。
    """

    def __init__(self, storage: StorageBackend, db: InMemoryDatabase):
        self._storage = storage
        self._db = db

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.STORE

    def process(self, task: FileTask, context: Dict[str, Any]) -> FileTask:
        filename = os.path.basename(task.file_path)
        storage_key = f"{task.dataset_name}/{filename}"

        # 存储文件
        stored_path = self._storage.store(task.file_path, storage_key)

        # 写入数据库
        db_row = {
            "dataset_name": task.dataset_name,
            "file_path": storage_key,
            "stored_path": stored_path,
        }
        db_row.update(task.metadata)
        episode_id = self._db.insert_episode(db_row)

        task.stage_results["store"] = {
            "episode_id": episode_id,
            "storage_key": storage_key,
            "stored_path": stored_path,
        }
        return task


class IndexProcessor(StageProcessor):
    """
    索引阶段: 将元数据写入搜索索引，支持后续全文检索。
    """

    def __init__(self, index: InMemoryIndex):
        self._index = index

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.INDEX

    def process(self, task: FileTask, context: Dict[str, Any]) -> FileTask:
        store_info = task.stage_results.get("store", {})
        episode_id = store_info.get("episode_id")

        if episode_id is None:
            raise ValueError("存储阶段未完成，无法索引")

        # 构建索引文档
        doc = {
            "dataset_name": task.dataset_name,
            "file_path": task.file_path,
            "nq": task.metadata.get("nq"),
            "nv": task.metadata.get("nv"),
            "nu": task.metadata.get("nu"),
            "num_steps": task.metadata.get("num_steps"),
            "quality_score": task.metadata.get("quality_score"),
            "has_nan": task.metadata.get("has_nan", False),
            "has_jumps": task.metadata.get("has_jumps", False),
        }

        self._index.index(str(episode_id), doc)

        task.stage_results["index"] = {
            "indexed": True,
            "doc_id": str(episode_id),
        }
        return task


# ============================================================
# DataPipeline 主类
# ============================================================

class DataPipeline:
    """
    数据摄入流水线 —— 串联所有阶段的主控制器。

    Java 等价:
        @Service
        public class DataPipelineService {
            @Autowired private List<StageProcessor> stages;

            @Async
            public PipelineReport process(List<FileTask> tasks) { ... }
        }
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._processors: Dict[PipelineStage, StageProcessor] = {}
        self._progress_callbacks: List[Callable[[int, int, FileTask], None]] = []

    def register_processor(self, processor: StageProcessor):
        """注册阶段处理器"""
        self._processors[processor.stage] = processor

    def on_progress(self, callback: Callable[[int, int, FileTask], None]):
        """注册进度回调（当前进度, 总数, 当前任务）"""
        self._progress_callbacks.append(callback)

    def process_file(self, task: FileTask) -> FileTask:
        """处理单个文件，经过所有启用的阶段"""
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()

        for stage in self.config.enabled_stages:
            if stage not in self._processors:
                continue

            task.current_stage = stage
            processor = self._processors[stage]

            success = self._execute_with_retry(task, processor)
            if not success:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                return task

        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        return task

    def process_batch(self, tasks: List[FileTask]) -> PipelineReport:
        """
        批量处理文件。

        这是流水线的主入口，生产环境可以改为异步/多线程版本。
        """
        report = PipelineReport(total_files=len(tasks))
        batch_start = time.time()
        stage_times: Dict[str, float] = {}

        for i, task in enumerate(tasks):
            try:
                result = self.process_file(task)

                if result.status == TaskStatus.COMPLETED:
                    report.completed += 1
                elif result.status == TaskStatus.FAILED:
                    report.failed += 1
                    report.errors.append({
                        "file": task.file_path,
                        "error": "; ".join(task.errors),
                    })
                else:
                    report.skipped += 1

            except Exception as e:
                report.failed += 1
                report.errors.append({
                    "file": task.file_path,
                    "error": str(e),
                })
                if self.config.stop_on_error:
                    break

            # 进度回调
            if (i + 1) % self.config.progress_interval == 0 or i == len(tasks) - 1:
                for callback in self._progress_callbacks:
                    callback(i + 1, len(tasks), task)

        report.total_time = time.time() - batch_start
        return report

    def _execute_with_retry(self, task: FileTask, processor: StageProcessor) -> bool:
        """带重试的阶段执行"""
        for attempt in range(self.config.max_retries + 1):
            try:
                processor.process(task, {})
                return True
            except Exception as e:
                task.retry_count += 1
                error_msg = f"阶段 {processor.stage.value} 第 {attempt + 1} 次失败: {e}"
                task.errors.append(error_msg)

                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    time.sleep(min(delay, 0.1))  # 演示中限制最大等待时间
                else:
                    return False
        return False

    def get_status(self, tasks: List[FileTask]) -> Dict[str, Any]:
        """获取批次处理状态"""
        status_counts = {}
        for s in TaskStatus:
            status_counts[s.value] = sum(1 for t in tasks if t.status == s)
        return {
            "total": len(tasks),
            "status": status_counts,
            "stages": {s.value: s.value in self._processors for s in PipelineStage},
        }


# ============================================================
# 便捷工厂函数
# ============================================================

def create_default_pipeline(
    storage_root: str,
    config: Optional[PipelineConfig] = None,
) -> Tuple[DataPipeline, InMemoryDatabase, InMemoryIndex]:
    """
    创建一个配置好的默认流水线。

    Java 等价: @Bean 方法 or Spring 自动配置
    """
    config = config or PipelineConfig(storage_root=storage_root)

    db = InMemoryDatabase()
    storage = LocalFileSystemStorage(storage_root)
    index = InMemoryIndex()

    pipeline = DataPipeline(config)
    pipeline.register_processor(ReceiveProcessor())
    pipeline.register_processor(ValidateProcessor(config))
    pipeline.register_processor(ExtractProcessor())
    pipeline.register_processor(StoreProcessor(storage, db))
    pipeline.register_processor(IndexProcessor(index))

    return pipeline, db, index


# ============================================================
# 测试数据生成
# ============================================================

def generate_test_episodes(
    directory: str,
    count: int = 10,
    nq: int = 7,
    inject_issues: bool = True,
) -> List[str]:
    """生成一批测试用的 PKL 轨迹文件"""
    os.makedirs(directory, exist_ok=True)
    paths = []

    for i in range(count):
        num_steps = np.random.randint(50, 500)
        qpos = np.random.randn(num_steps, nq) * 0.5
        qvel = np.random.randn(num_steps, nq) * 0.1
        action = np.random.randn(num_steps, nq) * 0.3

        # 注入一些问题数据
        if inject_issues:
            if i % 5 == 2:
                qpos[np.random.randint(0, num_steps), np.random.randint(0, nq)] = np.nan
            if i % 5 == 4:
                jump_frame = np.random.randint(1, num_steps)
                qpos[jump_frame] = qpos[jump_frame - 1] + 10.0

        data = {
            "qpos": qpos,
            "qvel": qvel,
            "action": action,
            "timestep": 0.02,
        }

        path = os.path.join(directory, f"episode_{i:04d}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        paths.append(path)

    return paths


# ============================================================
# 主函数 —— 演示完整流水线
# ============================================================

def main():
    print(DIVIDER)
    print("第 7 章 · 03 - 数据摄入流水线 (Data Pipeline)")
    print(DIVIDER)

    with tempfile.TemporaryDirectory(prefix="ch07_pipeline_") as tmpdir:
        raw_dir = os.path.join(tmpdir, "raw")
        store_dir = os.path.join(tmpdir, "store")

        # --------------------------------------------------
        # 1. 生成测试数据
        # --------------------------------------------------
        print(f"\n{'1. 生成测试数据':=^60}")
        file_paths = generate_test_episodes(raw_dir, count=15, nq=7)
        print(f"生成 {len(file_paths)} 个测试轨迹文件")
        print(f"目录: {raw_dir}")

        # --------------------------------------------------
        # 2. 创建流水线
        # --------------------------------------------------
        print(f"\n{'2. 创建并配置流水线':=^60}")

        config = PipelineConfig(
            storage_root=store_dir,
            max_retries=2,
            retry_delay=0.05,
            progress_interval=5,
        )

        pipeline, db, index = create_default_pipeline(store_dir, config)

        # 注册进度回调
        def progress_callback(current: int, total: int, task: FileTask):
            pct = current / total * 100
            status_icon = "✓" if task.status == TaskStatus.COMPLETED else "✗"
            print(f"  进度: [{current}/{total}] {pct:.0f}% {status_icon} {os.path.basename(task.file_path)}")

        pipeline.on_progress(progress_callback)

        print(f"启用阶段: {[s.value for s in config.enabled_stages]}")
        print(f"最大重试: {config.max_retries}")
        print(f"存储根目录: {store_dir}")

        # --------------------------------------------------
        # 3. 执行批量处理
        # --------------------------------------------------
        print(f"\n{'3. 执行批量处理':=^60}")

        tasks = [
            FileTask(file_path=fp, dataset_name="test_arm_data")
            for fp in file_paths
        ]

        report = pipeline.process_batch(tasks)

        print(f"\n{SUB_DIVIDER}")
        print("流水线执行报告:")
        print(report.summary())

        # --------------------------------------------------
        # 4. 查看数据库内容
        # --------------------------------------------------
        print(f"\n{'4. 查看数据库内容':=^60}")

        episodes = db.list_episodes()
        print(f"\n数据库中共 {db.count} 条记录\n")

        print(f"{'ID':<4} {'文件':<25} {'nq':<4} {'帧数':<8} {'质量':<8} {'NaN':<6} {'大小'}")
        print("-" * 75)
        for ep in episodes[:10]:
            name = os.path.basename(ep.get("file_path", ""))
            nq = ep.get("nq", "?")
            steps = ep.get("num_steps", "?")
            score = ep.get("quality_score", "?")
            has_nan = ep.get("has_nan", False)
            size = ep.get("file_size", 0)
            score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
            print(f"{ep['id']:<4} {name:<25} {nq:<4} {steps:<8} "
                  f"{score_str:<8} {str(has_nan):<6} {size:,}")

        if len(episodes) > 10:
            print(f"  ... 还有 {len(episodes) - 10} 条")

        # --------------------------------------------------
        # 5. 查看搜索索引
        # --------------------------------------------------
        print(f"\n{'5. 查看搜索索引':=^60}")
        print(f"索引文档数: {index.count}")

        # 搜索示例
        nan_results = index.search({"has_nan": True})
        print(f"搜索 has_nan=True: 找到 {len(nan_results)} 条")

        clean_results = index.search({"has_nan": False, "has_jumps": False})
        print(f"搜索 has_nan=False AND has_jumps=False: 找到 {len(clean_results)} 条")

        # --------------------------------------------------
        # 6. 查看各任务的详细状态
        # --------------------------------------------------
        print(f"\n{'6. 任务详细状态':=^60}")

        for task in tasks[:5]:
            status_icon = {"completed": "✅", "failed": "❌", "skipped": "⏭️"}.get(
                task.status.value, "⏳"
            )
            print(f"\n{status_icon} {os.path.basename(task.file_path)}")
            print(f"  状态: {task.status.value}")
            print(f"  耗时: {task.elapsed:.3f}s")
            if task.errors:
                print(f"  错误: {task.errors[0]}")
            for stage, result in task.stage_results.items():
                if stage == "validate" and not result.get("passed", True):
                    print(f"  校验问题: {result.get('warning', '')}")
                elif stage == "store":
                    print(f"  存储: episode_id={result.get('episode_id')}")

        # --------------------------------------------------
        # 7. 自定义校验器示例
        # --------------------------------------------------
        print(f"\n{'7. 自定义校验器示例':=^60}")

        # 添加自定义校验器: 检查数据方差是否太小
        def variance_check(qpos: np.ndarray) -> Tuple[bool, str]:
            """自定义校验: 整体方差不能太低"""
            var = np.nanvar(qpos)
            if var < 0.01:
                return False, f"数据方差过低: {var:.6f}"
            return True, "方差正常"

        validator = ValidateProcessor(config)
        validator.add_validator(variance_check)

        print("已添加自定义校验器: 方差检查")
        print("此校验器会拒绝方差低于 0.01 的数据")

        # --------------------------------------------------
        # 8. 跳过特定阶段
        # --------------------------------------------------
        print(f"\n{'8. 跳过特定阶段（仅提取元数据）':=^60}")

        extract_only_config = PipelineConfig(
            storage_root=store_dir,
            enabled_stages=[PipelineStage.RECEIVE, PipelineStage.EXTRACT],
        )

        pipeline2, db2, _ = create_default_pipeline(store_dir, extract_only_config)

        sample_task = FileTask(file_path=file_paths[0], dataset_name="quick_scan")
        result = pipeline2.process_file(sample_task)

        print(f"状态: {result.status.value}")
        print(f"执行的阶段: {list(result.stage_results.keys())}")
        print(f"提取的元数据: nq={result.metadata.get('nq')}, "
              f"steps={result.metadata.get('num_steps')}, "
              f"score={result.metadata.get('quality_score')}")

        # --------------------------------------------------
        # 9. 存储后端信息
        # --------------------------------------------------
        print(f"\n{'9. 存储后端信息':=^60}")

        storage = LocalFileSystemStorage(store_dir)
        first_ep = episodes[0] if episodes else None
        if first_ep:
            key = first_ep.get("file_path", "")
            info = storage.get_info(key)
            print(f"存储路径: {info.get('path', '?')}")
            print(f"文件大小: {info.get('size', 0):,} bytes")
            print(f"修改时间: {info.get('modified_at', '?')}")
            print(f"\n模拟 S3 路径: s3://robot-data-bucket/{key}")

    print(f"\n{DIVIDER}")
    print("✅ 数据摄入流水线演示完成！")
    print(f"""
关键收获:
  1. Pipeline 模式把复杂处理分解为独立阶段，每个阶段职责单一
  2. 重试机制用指数退避处理瞬时错误（网络抖动、文件锁等）
  3. 存储后端用接口抽象，本地文件系统和 S3 可以无缝切换
  4. 进度回调让长时间批处理有可观测性
  5. 每个任务（FileTask）携带完整的处理历史，便于排错和审计

对 Java 后端工程师:
  - Pipeline ≈ Spring Batch 的 Step 链
  - StorageBackend ≈ Spring 的 @Service + @Qualifier 替换实现
  - 进度回调 ≈ Spring ApplicationEvent
  - 重试 ≈ Spring Retry 的 @Retryable
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
