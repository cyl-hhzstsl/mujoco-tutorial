# 第 7 章 · 01 — 数据库 Schema 设计

> **目标**: 为机器人数据平台设计一套完整的 PostgreSQL Schema，覆盖数据集管理、轨迹元数据、关节结构、质量报告、训练记录。

## 设计原则

1. 用 **JSONB** 存储半结构化数据（灵活又可查询）
2. 为高频查询建索引，但不过度索引
3. 用**视图**封装复杂查询，简化上层 API
4. 所有时间戳使用 **TIMESTAMPTZ**（带时区）
5. **软删除**优于硬删除（`deleted_at` 字段）

---

## 1. 核心表结构

### 1.1 datasets 表 — 数据集（核心实体）

一个数据集 = 一批由相同机器人执行相同任务采集的轨迹。

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| id | BIGSERIAL PK | 自增主键 |
| name | VARCHAR(255) UNIQUE | 全局唯一名称 |
| robot_type | VARCHAR(100) | 机器人类型（最重要的筛选维度） |
| task | VARCHAR(255) | 任务描述 |
| source | VARCHAR(50) | `simulation` / `real` / `mixed` |
| status | VARCHAR(50) | `active` / `archived` / `processing` / `failed` |
| episode_count | INTEGER | 冗余字段，加速查询 |
| total_size | BIGINT | 总文件大小（字节） |
| metadata | JSONB | 灵活扩展字段 |
| deleted_at | TIMESTAMPTZ | NULL = 未删除（软删除） |

### 1.2 episodes 表 — 轨迹（最小独立数据单元）

原始数据存储在文件系统/对象存储，这里只存**元数据**。

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| dataset_id | BIGINT FK | 所属数据集 |
| file_path | VARCHAR(1024) | 相对存储根目录的路径 |
| nq / nv / nu | INTEGER | MuJoCo 模型维度 |
| num_steps | INTEGER | 总帧数 |
| duration | FLOAT | 总时长 (秒) |
| has_nan / has_jumps | BOOLEAN | 质量快速标记 |
| quality_score | FLOAT | 质量评分 [0, 1] |
| qpos_range | JSONB | 每个关节的 min/max |
| file_size / checksum | BIGINT/VARCHAR | 文件信息 |

### 1.3 joint_schemas 表 — 关节结构定义

对应第 3 章中 `model.jnt_type`、`model.jnt_qposadr` 等信息。

| 字段 | 说明 |
| :--- | :--- |
| robot_type + joint_index | 联合唯一约束 |
| joint_name | 如 `left_shoulder_yaw` |
| joint_type | `free` / `ball` / `slide` / `hinge` |
| qpos_start | 该关节在 qpos 数组中的起始索引 |
| qpos_dim | 占据的维度数 (hinge=1, ball=4, free=7) |
| range_low / range_high | 关节限位 |

### 1.4 quality_reports 表 — 质量报告

第 6 章的校验结果结构化存储。

| 字段 | 说明 |
| :--- | :--- |
| episode_id | 关联的轨迹 |
| overall_score | [0.0, 1.0] |
| nan_count / jump_count | 各项指标 |
| report_json | JSONB，存储完整校验输出 |
| validator_version | 校验器版本（可追溯） |

### 1.5 training_runs 表 — 训练记录

追踪「哪些数据训练了哪个模型」。

### 1.6 标签系统 — tags / dataset_tags / episode_tags / episode_labels

- **tags**: 灵活分类（`high_quality`, `needs_review` 等）
- **多对多关联表**: dataset_tags, episode_tags
- **episode_labels**: 键值对形式的精细标注

---

## 2. 视图

### dataset_overview（最常用）

聚合 episodes 表的统计信息，API 层直接使用：

```sql
SELECT d.*, COUNT(e.id) AS actual_episode_count,
       AVG(e.quality_score) AS avg_quality_score, ...
FROM datasets d LEFT JOIN episodes e ON e.dataset_id = d.id
WHERE d.deleted_at IS NULL
GROUP BY d.id;
```

### quality_summary

按数据集统计质量分布（excellent/good/fair/poor 各多少）。

---

## 3. 索引策略

| 类型 | 索引 | 原因 |
| :--- | :--- | :--- |
| 外键 | `idx_episodes_dataset_id` | PostgreSQL 不自动为 FK 建索引 |
| 筛选 | `idx_datasets_robot_type` | 高频筛选字段 |
| 条件 | `idx_episodes_has_nan WHERE has_nan=TRUE` | 部分索引，只索引问题数据 |
| 软删除 | `idx_datasets_not_deleted WHERE deleted_at IS NULL` | 大部分查询都需要 |
| JSONB | `idx_datasets_metadata_gin USING GIN` | 支持 `@>` 包含查询 |

---

## 4. 触发器

自动更新 `updated_at`：

```sql
CREATE TRIGGER trigger_datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

---

## 5. 常用查询示例

| 场景 | 对应 API |
| :--- | :------- |
| 数据集列表（按 robot_type 筛选） | `GET /api/datasets?robot_type=aloha` |
| 高质量轨迹筛选 | `POST /api/episodes/search` |
| 数据集质量概览 | `GET /api/datasets/{id}/quality` |
| JSONB 查询特定关节范围 | `qpos_range @> '[{"joint": "left_gripper"}]'` |
| 统计各机器人数据量 | `GROUP BY robot_type` |
| 按标签筛选 | `JOIN dataset_tags JOIN tags` |

---

## 6. 总结

```
┌──────────────────────────────────────────────────────────────┐
│              Schema 设计 — 核心要点                            │
│                                                              │
│  6 张核心表:                                                  │
│    datasets → episodes → quality_reports                     │
│    joint_schemas (机器人关节结构)                              │
│    training_runs (数据→模型追踪)                               │
│    tags 系统 (灵活分类)                                       │
│                                                              │
│  关键设计决策:                                                │
│    • JSONB 存储半结构化数据 (metadata, qpos_range)            │
│    • 冗余字段 (episode_count) 加速查询 + 定期同步             │
│    • 软删除 (deleted_at)                                      │
│    • 视图封装复杂 JOIN，API 层直接 SELECT                     │
│    • 部分索引 (WHERE has_nan=TRUE) 只索引需要的数据           │
│                                                              │
│  Java 映射:                                                   │
│    datasets → DatasetEntity.java                             │
│    episodes → EpisodeEntity.java                             │
│    视图 → @Query 直接使用                                     │
└──────────────────────────────────────────────────────────────┘
```
