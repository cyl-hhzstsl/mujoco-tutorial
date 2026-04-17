# 第 7 章：数据平台设计

> 从单机脚本到企业级架构 —— 当你的机器人数据集从 10 个文件增长到 100 万个，你需要一个数据平台。

## 本章目标

1. 设计一套**企业级机器人数据平台的数据库 Schema**，支撑数据全生命周期管理
2. 实现**元数据自动提取**，从 HDF5/PKL/XML 文件中解析出结构化信息
3. 构建**数据摄入流水线**（Data Pipeline），串联校验、提取、存储、索引
4. 设计**RESTful API**，让前端和训练系统能方便地查询和消费数据
5. 理解后端工程师在机器人数据领域的核心价值

## 文件结构

| 文件 | 内容 |
|------|------|
| `01_schema_design.sql` | 数据库 Schema 设计：datasets、episodes、joint_schemas、quality_reports、training_runs、tags/labels、视图、索引 |
| `02_metadata_extractor.py` | 元数据提取器：HDF5/PKL/XML 文件解析、JointSchema 构建、质量指标计算、批量提取 |
| `03_data_pipeline.py` | 数据摄入流水线：接收 → 校验 → 提取元数据 → 存储 → 索引，带重试与进度追踪 |
| `04_api_design.py` | REST API 设计：DatasetService 服务层、分页与过滤、请求/响应模型，附 Java Spring Boot 对照 |
| `05_exercises.py` | 练习：多模态数据 Schema、数据版本管理、缓存层、数据迁移脚本 |

## 核心架构

```
                        ┌──────────────┐
                        │   用户 / 前端  │
                        └──────┬───────┘
                               │  REST API
                        ┌──────▼───────┐
                        │   API 服务层   │  ← 04_api_design.py
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
       │  数据库层     │  │  文件存储   │  │  搜索索引   │
       │ (PostgreSQL) │  │ (S3/MinIO) │  │ (ES/Meilisearch) │
       └──────────────┘  └────────────┘  └─────────────┘
              ↑                ↑                ↑
              │                │                │
       ┌──────┴────────────────┴────────────────┘
       │           数据摄入流水线                    ← 03_data_pipeline.py
       │   接收 → 校验 → 提取元数据 → 存储 → 索引
       └──────────────────┬─────────────────────┘
                          │
                   ┌──────▼───────┐
                   │  元数据提取器  │  ← 02_metadata_extractor.py
                   └──────┬───────┘
                          │
                   ┌──────▼───────┐
                   │  原始数据文件  │  HDF5 / PKL / XML / URDF
                   └──────────────┘
```

## 从后端视角看机器人数据

| 后端概念 | 机器人数据领域对应 |
|---------|------------------|
| 用户表 | datasets 表（数据集是核心实体） |
| 订单表 | episodes 表（每条轨迹是一个独立数据单元） |
| Schema 版本管理 | joint_schemas 表（不同机器人有不同关节结构） |
| 数据校验 | quality_reports 表（第 6 章的校验结果结构化存储） |
| ETL 流水线 | data pipeline（数据摄入的标准路径） |
| 接口文档 | API 设计（给训练系统和前端提供标准接口） |

## 运行方式

```bash
# SQL 文件用于阅读和参考（也可导入 PostgreSQL 实际运行）
# Python 文件可直接运行，自动生成测试数据

python 02_metadata_extractor.py
python 03_data_pipeline.py
python 04_api_design.py
python 05_exercises.py
```

## 依赖

```bash
pip install numpy
# 可选（如果有真实 HDF5 文件）
pip install h5py
```

## 给 Java 后端工程师的说明

本章的代码模式你会非常熟悉：

- **dataclass** ≈ Java 的 `@Data` / `record`
- **ABC 抽象类** ≈ Java 的 `interface`
- **类型注解** ≈ Java 的强类型（Python 的类型注解不强制，但我们全部标注）
- **依赖注入** ≈ Spring 的 `@Autowired`（Python 通过构造函数参数实现）
- **Service 层** ≈ Spring 的 `@Service`

在 `04_api_design.py` 中，我们会展示每个 API 的等效 Java Spring Boot 实现作为对照。
