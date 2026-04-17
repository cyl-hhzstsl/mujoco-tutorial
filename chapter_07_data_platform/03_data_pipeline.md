# 第 7 章 · 03 — 数据摄入流水线 (Data Pipeline)

> **目标**: 构建一条完整的数据摄入流水线，将原始文件转化为平台可管理的数据资产。

## 流水线阶段

```
接收 (Receive) → 校验 (Validate) → 提取元数据 (Extract) → 存储 (Store) → 索引 (Index)
```

## 核心知识点

1. Pipeline 模式 — 数据处理的标准架构
2. 阶段可配置 — 跳过特定步骤
3. 错误处理与重试 — 指数退避
4. 进度追踪 — 回调通知
5. 存储后端抽象 — 本地 FS ↔ S3 可替换

---

## 1. 架构设计

### 阶段枚举

```python
class PipelineStage(Enum):
    RECEIVE = "receive"           # 验证文件存在性，计算校验和
    VALIDATE = "validate"         # 检查数据质量
    EXTRACT = "extract_metadata"  # 提取结构化元数据
    STORE = "store"               # 复制到存储后端，写入数据库
    INDEX = "index"               # 写入搜索索引
```

### 文件任务单 (FileTask)

每个文件对应一个 `FileTask`，记录完整的处理历史：

```python
@dataclass
class FileTask:
    file_path: str
    dataset_name: str
    status: TaskStatus          # PENDING → PROCESSING → COMPLETED/FAILED
    stage_results: Dict         # 每个阶段的输出
    errors: List[str]           # 错误记录
    retry_count: int
```

---

## 2. 各阶段处理器

| 阶段 | 处理器 | 职责 |
| :--- | :----- | :--- |
| Receive | ReceiveProcessor | 文件存在性、校验和、格式检测 |
| Validate | ValidateProcessor | NaN 比例、帧数范围、自定义校验器 |
| Extract | ExtractProcessor | nq/nv/nu、qpos_range、质量评分 |
| Store | StoreProcessor | 复制文件到存储后端，写入数据库 |
| Index | IndexProcessor | 写入搜索索引 |

### 自定义校验器

```python
validator = ValidateProcessor(config)
validator.add_validator(lambda qpos: (np.nanvar(qpos) > 0.01, "方差过低"))
```

---

## 3. 存储后端抽象

```python
class StorageBackend(ABC):
    def store(self, source_path, target_key) -> str: ...
    def exists(self, key) -> bool: ...
    def delete(self, key) -> bool: ...
    def get_info(self, key) -> Dict: ...
```

| 实现 | 用途 |
| :--- | :--- |
| LocalFileSystemStorage | 本地开发 |
| S3Storage | 生产环境（待实现） |
| MinIOStorage | 私有云（待实现） |

替换存储后端时，流水线代码**零修改**。

---

## 4. 重试机制

```python
config = PipelineConfig(
    max_retries=3,
    retry_delay=1.0,    # 指数退避: 1s, 2s, 4s
)
```

- 瞬时错误（网络抖动、文件锁）→ 自动重试
- 永久错误（文件损坏）→ 记录错误并跳过

---

## 5. 使用方式

### 完整流水线

```python
pipeline, db, index = create_default_pipeline(storage_root="/data/store")

tasks = [FileTask(file_path=fp, dataset_name="my_dataset") for fp in file_paths]
report = pipeline.process_batch(tasks)
print(report.summary())
```

### 仅提取元数据（跳过存储和索引）

```python
config = PipelineConfig(
    enabled_stages=[PipelineStage.RECEIVE, PipelineStage.EXTRACT],
)
pipeline = DataPipeline(config)
```

### 进度回调

```python
pipeline.on_progress(lambda current, total, task:
    print(f"[{current}/{total}] {task.file_path}")
)
```

---

## 6. 执行报告

```python
@dataclass
class PipelineReport:
    total_files: int
    completed: int
    failed: int
    skipped: int
    total_time: float
    stage_times: Dict[str, float]
    errors: List[Dict[str, str]]
```

---

## 7. 设计模式对照 (Java)

| Python | Java Spring |
| :----- | :---------- |
| DataPipeline | Spring Batch Step 链 |
| StorageBackend | `@Service` + `@Qualifier` 替换 |
| 进度回调 | `ApplicationEvent` |
| 重试 | `@Retryable` |
| PipelineConfig | `@ConfigurationProperties` |
| FileTask | 工单/任务表 |

---

## 8. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              数据摄入流水线 — 核心要点                          │
│                                                              │
│  5 个阶段: 接收 → 校验 → 提取 → 存储 → 索引                  │
│                                                              │
│  关键特性:                                                    │
│    • 阶段可配置（跳过特定步骤）                                │
│    • 指数退避重试（处理瞬时错误）                              │
│    • 存储后端接口抽象（FS ↔ S3 无缝切换）                     │
│    • 进度回调（长时间批处理可观测）                            │
│    • FileTask 携带完整处理历史（便于排错和审计）               │
│                                                              │
│  输出:                                                        │
│    • 数据库记录 (episodes 表)                                 │
│    • 搜索索引文档                                             │
│    • 存储后端中的规范化文件                                    │
│    • PipelineReport 执行报告                                  │
└──────────────────────────────────────────────────────────────┘
```
