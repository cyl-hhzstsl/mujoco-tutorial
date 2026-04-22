"""
第 7 章 · 04 - REST API 设计 (API Design)

目标: 设计一套完整的 REST API，让前端和训练系统能查询、筛选、消费机器人数据。

核心知识点:
  1. Service 层设计 —— 方法直接映射到 REST 端点
  2. Request/Response 模型 —— 用 dataclass 定义请求和响应的结构
  3. 分页支持 —— Cursor-based 和 Offset-based 分页
  4. 过滤与搜索 —— 按机器人类型、任务、质量评分等筛选
  5. 内存存储后端 —— 模拟数据库操作
  6. Java Spring Boot 对照 —— 每个 API 都有等效 Java 实现

设计原则:
  - RESTful: 资源名用名词复数，HTTP 方法表示动作
  - 统一响应格式: {"code": 200, "data": ..., "message": "ok"}
  - 幂等性: GET 请求幂等，POST 请求带去重 ID
  - 分页: 大列表必须分页，避免一次返回太多数据

运行: python 04_api_design.py
依赖: pip install numpy
"""

import numpy as np
import json
import time
import pickle
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


# ============================================================
# 通用响应模型
# ============================================================

@dataclass
class ApiResponse:
    """
    统一的 API 响应格式。

    所有端点都返回这个结构，前端只需一套解析逻辑。

    Java 等价:
        @Data
        public class ApiResponse<T> {
            private int code;
            private String message;
            private T data;
            private PageInfo page;  // 可选

            public static <T> ApiResponse<T> success(T data) {
                return new ApiResponse<>(200, "ok", data, null);
            }
        }
    """
    code: int = 200
    message: str = "ok"
    data: Any = None
    page: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        result = {"code": self.code, "message": self.message, "data": self.data}
        if self.page:
            result["page"] = self.page
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    @staticmethod
    def success(data: Any, page: Optional[Dict] = None) -> "ApiResponse":
        return ApiResponse(code=200, message="ok", data=data, page=page)

    @staticmethod
    def error(code: int, message: str) -> "ApiResponse":
        return ApiResponse(code=code, message=message, data=None)

    @staticmethod
    def not_found(resource: str = "资源") -> "ApiResponse":
        return ApiResponse(code=404, message=f"{resource}不存在", data=None)


# ============================================================
# 请求模型
# ============================================================

@dataclass
class PageRequest:
    """
    分页请求参数。

    Java 等价:
        // Spring Data 自带分页:
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
    """
    page: int = 1              # 页码（从 1 开始）
    size: int = 20             # 每页大小
    sort_by: str = "created_at"
    sort_order: str = "desc"   # "asc" or "desc"

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


@dataclass
class EpisodeFilter:
    """
    轨迹筛选条件。

    对应 SQL: WHERE robot_type = ? AND quality_score >= ? AND ...

    Java 等价:
        // Spring Data JPA Specification:
        public class EpisodeSpec implements Specification<Episode> {
            private String robotType;
            private String task;
            private Double minScore;
            private Double maxScore;
            private Boolean excludeNan;
            ...
        }
    """
    robot_type: Optional[str] = None
    task: Optional[str] = None
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None
    exclude_nan: bool = False
    exclude_jumps: bool = False
    min_steps: Optional[int] = None
    max_steps: Optional[int] = None
    dataset_id: Optional[int] = None


@dataclass
class QposRequest:
    """获取 qpos 数据的请求"""
    episode_id: int = 0
    start_frame: int = 0
    end_frame: Optional[int] = None    # None 表示到末尾
    dimensions: Optional[List[int]] = None  # None 表示所有维度


# ============================================================
# 数据模型（模拟数据库实体）
# ============================================================

@dataclass
class Dataset:
    """数据集实体"""
    id: int = 0
    name: str = ""
    robot_type: str = ""
    task: str = ""
    description: str = ""
    version: str = "1.0.0"
    source: str = "simulation"
    status: str = "active"
    episode_count: int = 0
    total_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Episode:
    """轨迹实体"""
    id: int = 0
    dataset_id: int = 0
    file_path: str = ""
    nq: int = 0
    nv: int = 0
    nu: int = 0
    num_steps: int = 0
    timestep: float = 0.02
    duration: float = 0.0
    has_nan: bool = False
    has_jumps: bool = False
    quality_score: float = 0.0
    qpos_range: List[Dict] = field(default_factory=list)
    file_size: int = 0
    checksum: str = ""
    created_at: str = ""
    # 运行时附加字段（不入库）
    qpos_data: Optional[np.ndarray] = None


@dataclass
class JointSchemaEntry:
    """关节结构条目"""
    joint_index: int = 0
    joint_name: str = ""
    joint_type: str = "hinge"
    qpos_start: int = 0
    qpos_dim: int = 1
    range_low: Optional[float] = None
    range_high: Optional[float] = None


@dataclass
class EpisodeStats:
    """轨迹统计信息"""
    episode_id: int = 0
    num_steps: int = 0
    duration: float = 0.0
    nq: int = 0
    nv: int = 0
    nu: int = 0
    qpos_mean: List[float] = field(default_factory=list)
    qpos_std: List[float] = field(default_factory=list)
    qpos_min: List[float] = field(default_factory=list)
    qpos_max: List[float] = field(default_factory=list)
    quality_score: float = 0.0
    has_nan: bool = False
    has_jumps: bool = False


# ============================================================
# 内存存储后端（模拟数据库）
# ============================================================

class InMemoryStore:
    """
    内存数据库，模拟 PostgreSQL。

    Java 等价:
        @Repository
        public class DatasetRepository {
            // 使用 HashMap 模拟（实际用 JPA / MyBatis）
            private Map<Long, Dataset> store = new ConcurrentHashMap<>();
        }
    """

    def __init__(self):
        self.datasets: Dict[int, Dataset] = {}
        self.episodes: Dict[int, Episode] = {}
        self.joint_schemas: Dict[str, List[JointSchemaEntry]] = {}
        self._next_dataset_id = 1
        self._next_episode_id = 1

    def add_dataset(self, ds: Dataset) -> Dataset:
        ds.id = self._next_dataset_id
        self._next_dataset_id += 1
        now = datetime.now().isoformat()
        ds.created_at = now
        ds.updated_at = now
        self.datasets[ds.id] = ds
        return ds

    def add_episode(self, ep: Episode) -> Episode:
        ep.id = self._next_episode_id
        self._next_episode_id += 1
        ep.created_at = datetime.now().isoformat()
        self.episodes[ep.id] = ep

        # 更新数据集的 episode_count
        if ep.dataset_id in self.datasets:
            ds = self.datasets[ep.dataset_id]
            ds.episode_count += 1
            ds.total_size += ep.file_size
        return ep

    def add_joint_schema(self, robot_type: str, entries: List[JointSchemaEntry]):
        self.joint_schemas[robot_type] = entries


# ============================================================
# DatasetService —— 核心 API 服务层
# ============================================================

class DatasetService:
    """
    数据集服务层 —— 每个方法对应一个 REST API 端点。

    这是整个 API 模块的核心。前端 / 训练系统调用这些方法（通过 HTTP），
    方法内部操作数据库并返回结构化响应。

    ======================================================================
    API 端点映射表:
    ======================================================================
    | 方法                    | HTTP         | 路径                           |
    |------------------------|--------------|--------------------------------|
    | list_datasets()        | GET          | /api/datasets                  |
    | get_dataset(id)        | GET          | /api/datasets/{id}             |
    | list_episodes(ds_id)   | GET          | /api/datasets/{id}/episodes    |
    | get_episode_qpos(...)  | GET          | /api/episodes/{id}/qpos        |
    | get_episode_stats(id)  | GET          | /api/episodes/{id}/stats       |
    | validate_episode(path) | POST         | /api/episodes/validate         |
    | search_episodes(filter)| POST         | /api/episodes/search           |
    | get_joint_schema(type) | GET          | /api/schemas/{robot_type}      |
    ======================================================================

    Java 等价:
        @RestController
        @RequestMapping("/api")
        public class DatasetController {
            @Autowired
            private DatasetService datasetService;

            @GetMapping("/datasets")
            public ApiResponse<List<DatasetVO>> listDatasets(
                @RequestParam(required = false) String robotType,
                Pageable pageable) {
                return datasetService.listDatasets(robotType, pageable);
            }
        }
    """

    def __init__(self, store: InMemoryStore):
        self._store = store

    # --------------------------------------------------
    # GET /api/datasets
    # --------------------------------------------------
    def list_datasets(
        self,
        robot_type: Optional[str] = None,
        task: Optional[str] = None,
        page_request: Optional[PageRequest] = None,
    ) -> ApiResponse:
        """
        获取数据集列表，支持筛选和分页。

        Java Spring Boot 等价:
        ┌─────────────────────────────────────────────────────────────┐
        │  @GetMapping("/datasets")                                    │
        │  public ResponseEntity<ApiResponse<Page<DatasetDTO>>>        │
        │      listDatasets(                                           │
        │          @RequestParam(required = false) String robotType,   │
        │          @RequestParam(required = false) String task,        │
        │          Pageable pageable) {                                │
        │      Specification<Dataset> spec = Specification.where(null);│
        │      if (robotType != null)                                  │
        │          spec = spec.and(DatasetSpec.hasRobotType(robotType));│
        │      if (task != null)                                       │
        │          spec = spec.and(DatasetSpec.hasTask(task));         │
        │      Page<Dataset> page = datasetRepo.findAll(spec, pageable);│
        │      return ResponseEntity.ok(ApiResponse.success(page));    │
        │  }                                                           │
        └─────────────────────────────────────────────────────────────┘
        """
        pr = page_request or PageRequest()

        # 筛选
        results = list(self._store.datasets.values())
        if robot_type:
            results = [d for d in results if d.robot_type == robot_type]
        if task:
            results = [d for d in results if d.task == task]

        # 排序
        total = len(results)
        reverse = pr.sort_order == "desc"
        results.sort(key=lambda d: getattr(d, pr.sort_by, ""), reverse=reverse)

        # 分页
        start = pr.offset
        end = start + pr.size
        page_data = results[start:end]

        return ApiResponse.success(
            data=[asdict(d) for d in page_data],
            page={
                "total": total,
                "page": pr.page,
                "size": pr.size,
                "total_pages": (total + pr.size - 1) // pr.size,
            },
        )

    # --------------------------------------------------
    # GET /api/datasets/{id}
    # --------------------------------------------------
    def get_dataset(self, dataset_id: int) -> ApiResponse:
        """
        获取单个数据集的详细信息。

        Java 等价:
        ┌───────────────────────────────────────────────────────────┐
        │  @GetMapping("/datasets/{id}")                             │
        │  public ResponseEntity<ApiResponse<DatasetDetailDTO>>      │
        │      getDataset(@PathVariable Long id) {                   │
        │      Dataset ds = datasetRepo.findById(id)                 │
        │          .orElseThrow(() -> new ResourceNotFoundException( │
        │              "Dataset not found: " + id));                  │
        │      return ResponseEntity.ok(ApiResponse.success(         │
        │          datasetMapper.toDetailDTO(ds)));                   │
        │  }                                                         │
        └───────────────────────────────────────────────────────────┘
        """
        ds = self._store.datasets.get(dataset_id)
        if ds is None:
            return ApiResponse.not_found("数据集")
        return ApiResponse.success(asdict(ds))

    # --------------------------------------------------
    # GET /api/datasets/{id}/episodes
    # --------------------------------------------------
    def list_episodes(
        self,
        dataset_id: int,
        page_request: Optional[PageRequest] = None,
    ) -> ApiResponse:
        """
        获取数据集下的所有轨迹。

        Java 等价:
        ┌───────────────────────────────────────────────────────────┐
        │  @GetMapping("/datasets/{id}/episodes")                    │
        │  public ResponseEntity<ApiResponse<Page<EpisodeDTO>>>      │
        │      listEpisodes(                                         │
        │          @PathVariable Long id,                            │
        │          Pageable pageable) {                              │
        │      Page<Episode> page = episodeRepo                      │
        │          .findByDatasetId(id, pageable);                   │
        │      return ResponseEntity.ok(ApiResponse.success(page));  │
        │  }                                                         │
        └───────────────────────────────────────────────────────────┘
        """
        if dataset_id not in self._store.datasets:
            return ApiResponse.not_found("数据集")

        pr = page_request or PageRequest()
        episodes = [
            ep for ep in self._store.episodes.values()
            if ep.dataset_id == dataset_id
        ]

        total = len(episodes)
        reverse = pr.sort_order == "desc"
        episodes.sort(key=lambda e: getattr(e, pr.sort_by, ""), reverse=reverse)

        start = pr.offset
        end = start + pr.size
        page_data = episodes[start:end]

        # 移除 qpos_data（大数据不序列化到列表接口）
        result = []
        for ep in page_data:
            d = asdict(ep)
            d.pop("qpos_data", None)
            result.append(d)

        return ApiResponse.success(
            data=result,
            page={
                "total": total,
                "page": pr.page,
                "size": pr.size,
                "total_pages": (total + pr.size - 1) // pr.size,
            },
        )

    # --------------------------------------------------
    # GET /api/episodes/{id}/qpos
    # --------------------------------------------------
    def get_episode_qpos(self, request: QposRequest) -> ApiResponse:
        """
        获取轨迹的 qpos 数据（支持切片）。

        这是训练系统最常调用的接口，用于按需加载训练数据。

        Java 等价:
        ┌───────────────────────────────────────────────────────────┐
        │  @GetMapping("/episodes/{id}/qpos")                        │
        │  public ResponseEntity<ApiResponse<QposDataDTO>>           │
        │      getEpisodeQpos(                                       │
        │          @PathVariable Long id,                            │
        │          @RequestParam(defaultValue = "0") int start,      │
        │          @RequestParam(required = false) Integer end,      │
        │          @RequestParam(required = false) List<Integer> dims)│
        │  {                                                         │
        │      Episode ep = episodeRepo.findById(id)                 │
        │          .orElseThrow();                                   │
        │      // 从文件存储中加载 qpos 数据                            │
        │      double[][] qpos = dataLoader                          │
        │          .loadQpos(ep.getFilePath(), start, end, dims);    │
        │      return ResponseEntity.ok(ApiResponse.success(         │
        │          new QposDataDTO(qpos, start, end)));              │
        │  }                                                         │
        └───────────────────────────────────────────────────────────┘
        """
        ep = self._store.episodes.get(request.episode_id)
        if ep is None:
            return ApiResponse.not_found("轨迹")

        if ep.qpos_data is None:
            return ApiResponse.error(400, "该轨迹没有加载 qpos 数据")

        data = ep.qpos_data
        end = request.end_frame if request.end_frame is not None else data.shape[0]
        start = max(0, request.start_frame)
        end = min(end, data.shape[0])

        sliced = data[start:end]

        if request.dimensions:
            valid_dims = [d for d in request.dimensions if 0 <= d < sliced.shape[1]]
            sliced = sliced[:, valid_dims]

        return ApiResponse.success({
            "episode_id": ep.id,
            "start_frame": start,
            "end_frame": end,
            "shape": list(sliced.shape),
            "data": sliced.tolist(),
        })

    # --------------------------------------------------
    # GET /api/episodes/{id}/stats
    # --------------------------------------------------
    def get_episode_stats(self, episode_id: int) -> ApiResponse:
        """
        获取轨迹的统计信息。

        Java 等价:
        ┌───────────────────────────────────────────────────────────┐
        │  @GetMapping("/episodes/{id}/stats")                       │
        │  public ResponseEntity<ApiResponse<EpisodeStatsDTO>>       │
        │      getEpisodeStats(@PathVariable Long id) {              │
        │      Episode ep = episodeRepo.findById(id).orElseThrow();  │
        │      EpisodeStats stats = statsService.compute(ep);        │
        │      return ResponseEntity.ok(ApiResponse.success(stats)); │
        │  }                                                         │
        └───────────────────────────────────────────────────────────┘
        """
        ep = self._store.episodes.get(episode_id)
        if ep is None:
            return ApiResponse.not_found("轨迹")

        stats = EpisodeStats(
            episode_id=ep.id,
            num_steps=ep.num_steps,
            duration=ep.duration,
            nq=ep.nq,
            nv=ep.nv,
            nu=ep.nu,
            quality_score=ep.quality_score,
            has_nan=ep.has_nan,
            has_jumps=ep.has_jumps,
        )

        if ep.qpos_data is not None:
            data = ep.qpos_data
            stats.qpos_mean = np.nanmean(data, axis=0).tolist()
            stats.qpos_std = np.nanstd(data, axis=0).tolist()
            stats.qpos_min = np.nanmin(data, axis=0).tolist()
            stats.qpos_max = np.nanmax(data, axis=0).tolist()

        return ApiResponse.success(asdict(stats))

    # --------------------------------------------------
    # POST /api/episodes/validate
    # --------------------------------------------------
    def validate_episode(self, file_path: str) -> ApiResponse:
        """
        校验一个轨迹文件（不入库，只返回校验结果）。

        Java 等价:
        ┌───────────────────────────────────────────────────────────┐
        │  @PostMapping("/episodes/validate")                        │
        │  public ResponseEntity<ApiResponse<ValidationResultDTO>>   │
        │      validateEpisode(                                      │
        │          @RequestBody ValidateRequest request) {           │
        │      ValidationResult result = validationService           │
        │          .validate(request.getFilePath());                 │
        │      return ResponseEntity.ok(ApiResponse.success(result));│
        │  }                                                         │
        └───────────────────────────────────────────────────────────┘
        """
        if not os.path.exists(file_path):
            return ApiResponse.error(400, f"文件不存在: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if not isinstance(data, dict) or "qpos" not in data:
                return ApiResponse.error(400, "文件格式不正确，缺少 qpos 数据")

            qpos = np.asarray(data["qpos"])
            issues = []
            passed = True

            # NaN 检查
            nan_count = int(np.isnan(qpos).sum())
            if nan_count > 0:
                issues.append({"check": "nan", "passed": False, "count": nan_count})
                passed = False
            else:
                issues.append({"check": "nan", "passed": True})

            # 帧数检查
            if qpos.shape[0] < 10:
                issues.append({"check": "min_steps", "passed": False, "actual": qpos.shape[0]})
                passed = False
            else:
                issues.append({"check": "min_steps", "passed": True})

            # 跳变检查
            if qpos.shape[0] > 1:
                max_jump = float(np.nanmax(np.abs(np.diff(qpos, axis=0))))
                has_jumps = max_jump > 2.0
                issues.append({
                    "check": "jumps",
                    "passed": not has_jumps,
                    "max_jump": round(max_jump, 4),
                })
                if has_jumps:
                    passed = False

            return ApiResponse.success({
                "file_path": file_path,
                "passed": passed,
                "checks": issues,
                "shape": list(qpos.shape),
            })

        except Exception as e:
            return ApiResponse.error(500, f"校验出错: {e}")

    # --------------------------------------------------
    # POST /api/episodes/search
    # --------------------------------------------------
    def search_episodes(
        self,
        filters: EpisodeFilter,
        page_request: Optional[PageRequest] = None,
    ) -> ApiResponse:
        """
        按条件搜索轨迹。

        这是最灵活的查询接口，支持多条件组合筛选。

        Java 等价:
        ┌───────────────────────────────────────────────────────────┐
        │  @PostMapping("/episodes/search")                          │
        │  public ResponseEntity<ApiResponse<Page<EpisodeDTO>>>      │
        │      searchEpisodes(                                       │
        │          @RequestBody EpisodeSearchRequest request,        │
        │          Pageable pageable) {                              │
        │      Specification<Episode> spec =                         │
        │          EpisodeSpec.fromFilter(request.getFilter());      │
        │      Page<Episode> page =                                  │
        │          episodeRepo.findAll(spec, pageable);              │
        │      return ResponseEntity.ok(ApiResponse.success(page));  │
        │  }                                                         │
        └───────────────────────────────────────────────────────────┘
        """
        pr = page_request or PageRequest()
        results = list(self._store.episodes.values())

        # 先通过 dataset 维度筛选
        if filters.robot_type or filters.task:
            valid_ds_ids = set()
            for ds in self._store.datasets.values():
                if filters.robot_type and ds.robot_type != filters.robot_type:
                    continue
                if filters.task and ds.task != filters.task:
                    continue
                valid_ds_ids.add(ds.id)
            results = [ep for ep in results if ep.dataset_id in valid_ds_ids]

        if filters.dataset_id is not None:
            results = [ep for ep in results if ep.dataset_id == filters.dataset_id]

        if filters.min_quality_score is not None:
            results = [ep for ep in results if ep.quality_score >= filters.min_quality_score]
        if filters.max_quality_score is not None:
            results = [ep for ep in results if ep.quality_score <= filters.max_quality_score]

        if filters.exclude_nan:
            results = [ep for ep in results if not ep.has_nan]
        if filters.exclude_jumps:
            results = [ep for ep in results if not ep.has_jumps]

        if filters.min_steps is not None:
            results = [ep for ep in results if ep.num_steps >= filters.min_steps]
        if filters.max_steps is not None:
            results = [ep for ep in results if ep.num_steps <= filters.max_steps]

        total = len(results)

        reverse = pr.sort_order == "desc"
        results.sort(key=lambda e: getattr(e, pr.sort_by, ""), reverse=reverse)

        start = pr.offset
        end = start + pr.size
        page_data = results[start:end]

        data = []
        for ep in page_data:
            d = asdict(ep)
            d.pop("qpos_data", None)
            data.append(d)

        return ApiResponse.success(
            data=data,
            page={
                "total": total,
                "page": pr.page,
                "size": pr.size,
                "total_pages": (total + pr.size - 1) // pr.size,
            },
        )

    # --------------------------------------------------
    # GET /api/schemas/{robot_type}
    # --------------------------------------------------
    def get_joint_schema(self, robot_type: str) -> ApiResponse:
        """
        获取机器人的关节结构定义。

        训练系统需要知道 qpos 的每个维度对应哪个关节，
        这个接口返回完整的映射关系。

        Java 等价:
        ┌───────────────────────────────────────────────────────────┐
        │  @GetMapping("/schemas/{robotType}")                       │
        │  public ResponseEntity<ApiResponse<JointSchemaDTO>>        │
        │      getJointSchema(@PathVariable String robotType) {      │
        │      List<JointSchema> joints = jointSchemaRepo            │
        │          .findByRobotType(robotType);                      │
        │      if (joints.isEmpty())                                 │
        │          throw new ResourceNotFoundException("Schema");    │
        │      return ResponseEntity.ok(ApiResponse.success(         │
        │          new JointSchemaDTO(robotType, joints)));           │
        │  }                                                         │
        └───────────────────────────────────────────────────────────┘
        """
        entries = self._store.joint_schemas.get(robot_type)
        if entries is None:
            return ApiResponse.not_found(f"机器人类型 '{robot_type}' 的关节结构")

        total_nq = 0
        joints_data = []
        for entry in entries:
            d = asdict(entry)
            joints_data.append(d)
            total_nq = max(total_nq, entry.qpos_start + entry.qpos_dim)

        return ApiResponse.success({
            "robot_type": robot_type,
            "total_joints": len(entries),
            "total_nq": total_nq,
            "joints": joints_data,
        })


# ============================================================
# 测试数据生成
# ============================================================

def populate_test_data(store: InMemoryStore) -> InMemoryStore:
    """生成完整的测试数据集"""

    # 数据集 1: ALOHA 杯子堆叠
    ds1 = store.add_dataset(Dataset(
        name="aloha_cup_stacking_v2",
        robot_type="aloha",
        task="cup_stacking",
        description="ALOHA 双臂机器人杯子堆叠任务",
        version="2.0.0",
        source="real",
    ))

    # 数据集 2: Franka 抓取
    ds2 = store.add_dataset(Dataset(
        name="franka_pick_place_sim",
        robot_type="franka",
        task="pick_and_place",
        description="Franka Panda 抓取放置仿真数据",
        version="1.0.0",
        source="simulation",
    ))

    # 数据集 3: UR5e 装配
    ds3 = store.add_dataset(Dataset(
        name="ur5e_assembly_v1",
        robot_type="ur5e",
        task="assembly",
        description="UR5e 装配任务真机数据",
        version="1.0.0",
        source="real",
    ))

    # 为每个数据集生成轨迹
    nq_map = {"aloha": 14, "franka": 7, "ur5e": 6}

    for ds in [ds1, ds2, ds3]:
        nq = nq_map.get(ds.robot_type, 7)
        for i in range(20):
            num_steps = np.random.randint(100, 600)
            qpos = np.random.randn(num_steps, nq) * 0.5

            has_nan = (i % 7 == 3)
            has_jumps = (i % 7 == 5)
            if has_nan:
                qpos[np.random.randint(0, num_steps), 0] = np.nan

            score = np.random.uniform(0.5, 1.0)
            if has_nan:
                score = max(0.3, score - 0.3)
            if has_jumps:
                score = max(0.4, score - 0.2)

            ep = store.add_episode(Episode(
                dataset_id=ds.id,
                file_path=f"{ds.name}/episode_{i:04d}.hdf5",
                nq=nq,
                nv=nq,
                nu=nq,
                num_steps=num_steps,
                timestep=0.02,
                duration=num_steps * 0.02,
                has_nan=has_nan,
                has_jumps=has_jumps,
                quality_score=round(score, 4),
                file_size=np.random.randint(50000, 500000),
                qpos_data=qpos,
            ))

    # 关节结构
    store.add_joint_schema("aloha", [
        JointSchemaEntry(i, f"left_{name}", "hinge", i, 1, -3.14, 3.14)
        for i, name in enumerate(["waist", "shoulder", "elbow", "forearm_roll",
                                   "wrist_angle", "wrist_rotate", "gripper"])
    ] + [
        JointSchemaEntry(7 + i, f"right_{name}", "hinge", 7 + i, 1, -3.14, 3.14)
        for i, name in enumerate(["waist", "shoulder", "elbow", "forearm_roll",
                                   "wrist_angle", "wrist_rotate", "gripper"])
    ])

    store.add_joint_schema("franka", [
        JointSchemaEntry(i, name, "hinge", i, 1, low, high)
        for i, (name, low, high) in enumerate([
            ("joint1", -2.90, 2.90), ("joint2", -1.76, 1.76),
            ("joint3", -2.90, 2.90), ("joint4", -3.07, -0.07),
            ("joint5", -2.90, 2.90), ("joint6", -0.02, 3.75),
            ("joint7", -2.90, 2.90),
        ])
    ])

    store.add_joint_schema("ur5e", [
        JointSchemaEntry(i, name, "hinge", i, 1, -6.28, 6.28)
        for i, name in enumerate([
            "shoulder_pan", "shoulder_lift", "elbow",
            "wrist_1", "wrist_2", "wrist_3",
        ])
    ])

    return store


# ============================================================
# 主函数 —— 演示所有 API
# ============================================================

def main():
    print(DIVIDER)
    print("第 7 章 · 04 - REST API 设计 (API Design)")
    print(DIVIDER)

    # 初始化
    store = InMemoryStore()
    populate_test_data(store)
    service = DatasetService(store)

    print(f"\n已加载测试数据:")
    print(f"  数据集: {len(store.datasets)} 个")
    print(f"  轨迹:   {len(store.episodes)} 条")
    print(f"  关节结构: {list(store.joint_schemas.keys())}")

    # --------------------------------------------------
    # API 1: GET /api/datasets
    # --------------------------------------------------
    print(f"\n{'API 1: GET /api/datasets':=^60}")
    print("获取数据集列表（分页）\n")

    resp = service.list_datasets(page_request=PageRequest(page=1, size=10))
    result = resp.to_dict()
    print(f"状态码: {result['code']}")
    print(f"数据条数: {len(result['data'])}")
    print(f"分页信息: {result['page']}")
    print(f"\n数据集列表:")
    for ds in result["data"]:
        print(f"  [{ds['id']}] {ds['name']} ({ds['robot_type']}/{ds['task']}) "
              f"episodes={ds['episode_count']}")

    # 按 robot_type 筛选
    print(f"\n--- 筛选 robot_type='aloha' ---")
    resp = service.list_datasets(robot_type="aloha")
    for ds in resp.data:
        print(f"  [{ds['id']}] {ds['name']}")

    # --------------------------------------------------
    # API 2: GET /api/datasets/{id}
    # --------------------------------------------------
    print(f"\n{'API 2: GET /api/datasets/1':=^60}")
    print("获取单个数据集详情\n")

    resp = service.get_dataset(1)
    ds_data = resp.data
    print(f"名称: {ds_data['name']}")
    print(f"机器人: {ds_data['robot_type']}")
    print(f"任务: {ds_data['task']}")
    print(f"轨迹数: {ds_data['episode_count']}")
    print(f"总大小: {ds_data['total_size']:,} bytes")

    # 404 情况
    resp_404 = service.get_dataset(999)
    print(f"\n请求不存在的数据集 (id=999): code={resp_404.code}, msg='{resp_404.message}'")

    # --------------------------------------------------
    # API 3: GET /api/datasets/{id}/episodes
    # --------------------------------------------------
    print(f"\n{'API 3: GET /api/datasets/1/episodes':=^60}")
    print("获取数据集的轨迹列表（分页）\n")

    resp = service.list_episodes(1, PageRequest(page=1, size=5))
    result = resp.to_dict()
    print(f"分页: 第 {result['page']['page']} 页, "
          f"共 {result['page']['total']} 条, "
          f"{result['page']['total_pages']} 页")
    print(f"\n{'ID':<4} {'文件':<35} {'帧数':<6} {'质量':<8} {'NaN':<6}")
    print("-" * 60)
    for ep in result["data"]:
        print(f"{ep['id']:<4} {ep['file_path']:<35} {ep['num_steps']:<6} "
              f"{ep['quality_score']:<8.4f} {ep['has_nan']!s:<6}")

    # --------------------------------------------------
    # API 4: GET /api/episodes/{id}/qpos
    # --------------------------------------------------
    print(f"\n{'API 4: GET /api/episodes/1/qpos':=^60}")
    print("获取轨迹的 qpos 数据（支持切片）\n")

    # 获取前 5 帧
    resp = service.get_episode_qpos(QposRequest(episode_id=1, start_frame=0, end_frame=5))
    qdata = resp.data
    print(f"episode_id: {qdata['episode_id']}")
    print(f"帧范围: [{qdata['start_frame']}, {qdata['end_frame']})")
    print(f"数据形状: {qdata['shape']}")
    print(f"前 2 帧数据 (截断):")
    for i, row in enumerate(qdata["data"][:2]):
        print(f"  帧 {i}: [{', '.join(f'{v:.3f}' for v in row[:5])}, ...]")

    # 指定维度
    print(f"\n--- 只获取第 0, 1, 2 维 ---")
    resp = service.get_episode_qpos(QposRequest(
        episode_id=1, start_frame=0, end_frame=3, dimensions=[0, 1, 2]
    ))
    print(f"数据形状: {resp.data['shape']}")

    # --------------------------------------------------
    # API 5: GET /api/episodes/{id}/stats
    # --------------------------------------------------
    print(f"\n{'API 5: GET /api/episodes/1/stats':=^60}")
    print("获取轨迹统计信息\n")

    resp = service.get_episode_stats(1)
    stats = resp.data
    print(f"episode_id: {stats['episode_id']}")
    print(f"帧数: {stats['num_steps']}, 时长: {stats['duration']:.2f}s")
    print(f"维度: nq={stats['nq']}, nv={stats['nv']}, nu={stats['nu']}")
    print(f"质量: {stats['quality_score']:.4f}")
    if stats["qpos_mean"]:
        print(f"qpos 均值 (前 5 维): {[f'{v:.4f}' for v in stats['qpos_mean'][:5]]}")
        print(f"qpos 标准差 (前 5 维): {[f'{v:.4f}' for v in stats['qpos_std'][:5]]}")

    # --------------------------------------------------
    # API 6: POST /api/episodes/validate
    # --------------------------------------------------
    print(f"\n{'API 6: POST /api/episodes/validate':=^60}")
    print("校验轨迹文件\n")

    with tempfile.TemporaryDirectory(prefix="ch07_api_") as tmpdir:
        # 生成正常文件
        good_path = os.path.join(tmpdir, "good.pkl")
        qpos_good = np.random.randn(200, 7) * 0.5
        with open(good_path, 'wb') as f:
            pickle.dump({"qpos": qpos_good}, f)

        resp = service.validate_episode(good_path)
        print(f"正常文件校验: passed={resp.data['passed']}")
        for check in resp.data["checks"]:
            icon = "✅" if check["passed"] else "❌"
            print(f"  {icon} {check['check']}")

        # 生成含 NaN 文件
        bad_path = os.path.join(tmpdir, "bad.pkl")
        qpos_bad = np.random.randn(200, 7) * 0.5
        qpos_bad[10:15, :] = np.nan
        with open(bad_path, 'wb') as f:
            pickle.dump({"qpos": qpos_bad}, f)

        print()
        resp = service.validate_episode(bad_path)
        print(f"含 NaN 文件校验: passed={resp.data['passed']}")
        for check in resp.data["checks"]:
            icon = "✅" if check["passed"] else "❌"
            extra = ""
            if "count" in check:
                extra = f" (count={check['count']})"
            if "max_jump" in check:
                extra = f" (max_jump={check['max_jump']})"
            print(f"  {icon} {check['check']}{extra}")

    # --------------------------------------------------
    # API 7: POST /api/episodes/search
    # --------------------------------------------------
    print(f"\n{'API 7: POST /api/episodes/search':=^60}")
    print("搜索轨迹（多条件组合筛选）\n")

    # 场景 1: 高质量 ALOHA 数据
    print("--- 场景 1: 高质量 ALOHA 数据 (score >= 0.8, 无 NaN) ---")
    resp = service.search_episodes(
        filters=EpisodeFilter(
            robot_type="aloha",
            min_quality_score=0.8,
            exclude_nan=True,
        ),
        page_request=PageRequest(page=1, size=5),
    )
    print(f"命中: {resp.page['total']} 条")
    for ep in resp.data[:3]:
        print(f"  [{ep['id']}] {ep['file_path']} score={ep['quality_score']:.4f}")

    # 场景 2: 长轨迹
    print(f"\n--- 场景 2: 长轨迹 (steps >= 400) ---")
    resp = service.search_episodes(
        filters=EpisodeFilter(min_steps=400),
    )
    print(f"命中: {resp.page['total']} 条")
    for ep in resp.data[:3]:
        print(f"  [{ep['id']}] steps={ep['num_steps']}")

    # 场景 3: 特定数据集的无跳变数据
    print(f"\n--- 场景 3: Franka 数据集 + 无跳变 ---")
    resp = service.search_episodes(
        filters=EpisodeFilter(
            robot_type="franka",
            exclude_jumps=True,
        ),
    )
    print(f"命中: {resp.page['total']} 条")

    # --------------------------------------------------
    # API 8: GET /api/schemas/{robot_type}
    # --------------------------------------------------
    print(f"\n{'API 8: GET /api/schemas/aloha':=^60}")
    print("获取 ALOHA 的关节结构\n")

    resp = service.get_joint_schema("aloha")
    schema = resp.data
    print(f"机器人: {schema['robot_type']}")
    print(f"关节数: {schema['total_joints']}")
    print(f"nq 总维度: {schema['total_nq']}")
    print(f"\n{'idx':<4} {'名称':<25} {'类型':<8} {'qpos_start':<12} {'范围'}")
    print("-" * 65)
    for j in schema["joints"]:
        range_str = f"[{j['range_low']:.2f}, {j['range_high']:.2f}]" if j["range_low"] is not None else "无"
        print(f"{j['joint_index']:<4} {j['joint_name']:<25} {j['joint_type']:<8} "
              f"{j['qpos_start']:<12} {range_str}")

    # 404 情况
    resp_404 = service.get_joint_schema("unknown_robot")
    print(f"\n请求不存在的机器人类型: code={resp_404.code}, msg='{resp_404.message}'")

    # --------------------------------------------------
    # 9. 展示完整 JSON 响应格式
    # --------------------------------------------------
    print(f"\n{'9. 完整 JSON 响应示例':=^60}")
    print("\nGET /api/datasets/1 的完整响应:")
    resp = service.get_dataset(1)
    print(resp.to_json())

    print(f"\n{DIVIDER}")
    print("✅ REST API 设计演示完成！")
    print(f"""
关键收获:
  1. Service 层的每个方法对应一个 REST 端点，职责清晰
  2. 统一 ApiResponse 格式，前端只需一套解析逻辑
  3. 分页是大列表查询的必备功能，避免一次返回太多数据
  4. EpisodeFilter 支持多条件组合，灵活应对各种查询需求
  5. 错误处理用标准 HTTP 状态码（404, 400, 500）
  6. 每个 API 都有对应的 Java Spring Boot 实现，方便后端工程师迁移

API 一览:
  GET    /api/datasets                    → 数据集列表
  GET    /api/datasets/{{id}}               → 数据集详情
  GET    /api/datasets/{{id}}/episodes      → 轨迹列表
  GET    /api/episodes/{{id}}/qpos          → qpos 数据（切片）
  GET    /api/episodes/{{id}}/stats         → 轨迹统计
  POST   /api/episodes/validate           → 校验文件
  POST   /api/episodes/search             → 搜索轨迹
  GET    /api/schemas/{{robot_type}}        → 关节结构
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
