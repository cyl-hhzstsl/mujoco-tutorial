# 第 7 章 · 04 — REST API 设计

> **目标**: 设计一套完整的 REST API，让前端和训练系统能查询、筛选、消费机器人数据。

## 设计原则

- **RESTful**: 资源名用名词复数，HTTP 方法表示动作
- **统一响应格式**: `{"code": 200, "data": ..., "message": "ok", "page": ...}`
- **幂等性**: GET 请求幂等，POST 请求带去重 ID
- **分页**: 大列表必须分页

---

## 1. 统一响应模型

```python
@dataclass
class ApiResponse:
    code: int = 200
    message: str = "ok"
    data: Any = None
    page: Optional[Dict] = None

    @staticmethod
    def success(data, page=None): ...
    @staticmethod
    def error(code, message): ...
    @staticmethod
    def not_found(resource): ...
```

---

## 2. API 端点一览

| 方法 | HTTP | 路径 | 用途 |
| :--- | :--- | :--- | :--- |
| list_datasets | GET | `/api/datasets` | 数据集列表（分页+筛选） |
| get_dataset | GET | `/api/datasets/{id}` | 数据集详情 |
| list_episodes | GET | `/api/datasets/{id}/episodes` | 轨迹列表 |
| get_episode_qpos | GET | `/api/episodes/{id}/qpos` | qpos 数据（支持切片） |
| get_episode_stats | GET | `/api/episodes/{id}/stats` | 轨迹统计 |
| validate_episode | POST | `/api/episodes/validate` | 校验文件（不入库） |
| search_episodes | POST | `/api/episodes/search` | 多条件组合搜索 |
| get_joint_schema | GET | `/api/schemas/{robot_type}` | 关节结构定义 |

---

## 3. 关键 API 详解

### 3.1 分页与筛选

```python
@dataclass
class PageRequest:
    page: int = 1          # 从 1 开始
    size: int = 20
    sort_by: str = "created_at"
    sort_order: str = "desc"
```

响应中的分页信息：
```json
{"total": 60, "page": 1, "size": 20, "total_pages": 3}
```

### 3.2 轨迹搜索 — EpisodeFilter

最灵活的查询接口，支持多条件组合：

```python
@dataclass
class EpisodeFilter:
    robot_type: Optional[str]
    task: Optional[str]
    min_quality_score: Optional[float]
    max_quality_score: Optional[float]
    exclude_nan: bool = False
    exclude_jumps: bool = False
    min_steps: Optional[int]
    max_steps: Optional[int]
    dataset_id: Optional[int]
```

典型使用场景：

```python
# 高质量 ALOHA 训练数据
filters = EpisodeFilter(
    robot_type="aloha",
    min_quality_score=0.8,
    exclude_nan=True,
)
```

### 3.3 qpos 数据切片

训练系统最常调用的接口：

```python
@dataclass
class QposRequest:
    episode_id: int
    start_frame: int = 0
    end_frame: Optional[int] = None   # None = 到末尾
    dimensions: Optional[List[int]] = None  # None = 所有维度
```

支持按帧范围和维度切片，避免加载全量数据。

### 3.4 关节结构查询

训练系统需要知道 qpos 的每个维度对应哪个关节：

```json
{
  "robot_type": "aloha",
  "total_joints": 14,
  "total_nq": 14,
  "joints": [
    {"joint_index": 0, "joint_name": "left_waist", "joint_type": "hinge", "qpos_start": 0, "qpos_dim": 1}
  ]
}
```

---

## 4. Java Spring Boot 对照

每个 API 都有对应的 Java 实现注释。关键映射：

| Python | Java Spring |
| :----- | :---------- |
| `DatasetService` | `@Service DatasetService` |
| `ApiResponse` | `ApiResponse<T>` 泛型 |
| `EpisodeFilter` | `Specification<Episode>` (JPA) |
| `PageRequest` | `Pageable` (Spring Data) |
| `list_datasets()` | `@GetMapping("/datasets")` |
| `search_episodes()` | `@PostMapping("/episodes/search")` |
| `InMemoryStore` | `JpaRepository<Episode, Long>` |

---

## 5. 错误处理

| HTTP 状态码 | 场景 |
| ----------: | :--- |
| 200 | 成功 |
| 400 | 请求参数错误、文件格式不正确 |
| 404 | 数据集/轨迹/Schema 不存在 |
| 500 | 内部错误 |

---

## 6. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              REST API 设计 — 核心要点                          │
│                                                              │
│  8 个 API 端点:                                               │
│    数据集: list / get                                         │
│    轨迹: list / qpos / stats / validate / search             │
│    关节结构: get_schema                                       │
│                                                              │
│  关键设计:                                                    │
│    • 统一 ApiResponse 格式                                    │
│    • 所有列表接口支持分页                                      │
│    • EpisodeFilter 支持多条件组合搜索                          │
│    • qpos 接口支持帧范围 + 维度切片                           │
│    • 每个 API 有对应的 Java Spring Boot 实现                  │
└──────────────────────────────────────────────────────────────┘
```
