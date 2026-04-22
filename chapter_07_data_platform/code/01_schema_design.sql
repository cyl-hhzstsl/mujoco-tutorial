-- ============================================================
-- 第 7 章 · 01 - 数据库 Schema 设计
--
-- 目标: 为机器人数据平台设计一套完整的 PostgreSQL Schema，
--       覆盖数据集管理、轨迹元数据、关节结构、质量报告、训练记录。
--
-- 设计原则:
--   1. 用 JSONB 存储半结构化数据（如关节范围），既灵活又可查询
--   2. 为高频查询建索引，但不过度索引
--   3. 用视图封装复杂查询，简化上层 API
--   4. 所有时间戳使用 TIMESTAMPTZ（带时区），避免线上踩坑
--   5. 软删除优于硬删除（用 deleted_at 字段）
--
-- 对应 Java 后端: 这套 Schema 可以直接用 MyBatis / JPA 映射。
--   - datasets   → DatasetEntity.java
--   - episodes   → EpisodeEntity.java
--   - 关联表     → @ManyToMany / @OneToMany
--
-- 运行: psql -d robot_data -f 01_schema_design.sql
--       或者纯粹作为设计文档阅读
-- ============================================================


-- ============================================================
-- 0. 初始化：创建数据库和扩展
-- ============================================================

-- CREATE DATABASE robot_data;  -- 取消注释以创建数据库

-- 启用 UUID 生成（PostgreSQL 13+ 内置，旧版本需要此扩展）
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 启用全文搜索支持（PostgreSQL 自带，这里确保中文分词可用）
-- CREATE EXTENSION IF NOT EXISTS "pg_trgm";


-- ============================================================
-- 1. datasets 表 —— 数据集（核心实体）
-- ============================================================
-- 一个数据集 = 一批由相同机器人执行相同任务采集的轨迹
-- 类比: 就像一个 Git 仓库，episodes 是里面的 commits

CREATE TABLE IF NOT EXISTS datasets (
    -- 主键：使用自增 ID 而非 UUID，因为内部系统 ID 可读性更重要
    -- 如果需要对外暴露，可以额外加一个 uuid 列
    id              BIGSERIAL PRIMARY KEY,

    -- 数据集名称，全局唯一（如 "aloha_cup_stacking_v2"）
    name            VARCHAR(255) NOT NULL UNIQUE,

    -- 机器人类型（如 "aloha", "franka", "ur5e"）
    -- 这是最重要的筛选维度之一
    robot_type      VARCHAR(100) NOT NULL,

    -- 任务描述（如 "cup_stacking", "pick_and_place"）
    task            VARCHAR(255) NOT NULL,

    -- 详细描述（支持 Markdown）
    description     TEXT DEFAULT '',

    -- 数据集版本（语义化版本号）
    version         VARCHAR(50) DEFAULT '1.0.0',

    -- 数据来源：仿真 or 真机
    source          VARCHAR(50) DEFAULT 'simulation'
                    CHECK (source IN ('simulation', 'real', 'mixed')),

    -- 数据集状态
    status          VARCHAR(50) DEFAULT 'active'
                    CHECK (status IN ('active', 'archived', 'processing', 'failed')),

    -- 轨迹数量（冗余字段，定期从 episodes 表同步，加速查询）
    episode_count   INTEGER DEFAULT 0,

    -- 总文件大小（字节），同样是冗余字段
    total_size      BIGINT DEFAULT 0,

    -- 元数据（灵活扩展字段，JSONB 类型）
    -- 示例: {"camera_count": 2, "fps": 50, "environment": "tabletop"}
    metadata        JSONB DEFAULT '{}',

    -- 创建者
    created_by      VARCHAR(100) DEFAULT 'system',

    -- 时间戳
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    -- 软删除标记（NULL 表示未删除）
    deleted_at      TIMESTAMPTZ DEFAULT NULL
);

-- 给 datasets 表加注释（PostgreSQL 特有，方便 DBA 和新人理解）
COMMENT ON TABLE datasets IS '数据集主表：一个数据集包含多条轨迹（episodes）';
COMMENT ON COLUMN datasets.robot_type IS '机器人类型，用于关联 joint_schemas 表';
COMMENT ON COLUMN datasets.metadata IS 'JSONB 扩展字段，存储不适合建列的灵活信息';


-- ============================================================
-- 2. episodes 表 —— 轨迹（数据的最小独立单元）
-- ============================================================
-- 一条 episode = 一次完整的机器人操作录像
-- 包含 qpos、qvel、ctrl 等时间序列数据
-- 原始数据存储在文件系统/对象存储，这里只存元数据

CREATE TABLE IF NOT EXISTS episodes (
    id              BIGSERIAL PRIMARY KEY,

    -- 所属数据集（外键）
    dataset_id      BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,

    -- 文件路径（相对于存储根目录）
    -- 示例: "aloha_cup_stacking/episode_00042.hdf5"
    file_path       VARCHAR(1024) NOT NULL,

    -- 文件格式
    file_format     VARCHAR(20) DEFAULT 'hdf5'
                    CHECK (file_format IN ('hdf5', 'pkl', 'npz', 'zarr')),

    -- ========== MuJoCo 模型维度 ==========
    -- 这些是理解数据的关键参数（第 3 章学的内容）
    nq              INTEGER,        -- 广义坐标维度
    nv              INTEGER,        -- 广义速度维度
    nu              INTEGER,        -- 控制输入维度

    -- ========== 轨迹统计 ==========
    num_steps       INTEGER,        -- 总帧数
    timestep        FLOAT,          -- 仿真步长（秒）
    duration        FLOAT,          -- 总时长（秒）= num_steps * timestep

    -- ========== 质量标记 ==========
    -- 快速筛选用，详细质量信息在 quality_reports 表
    has_nan         BOOLEAN DEFAULT FALSE,   -- 是否包含 NaN
    has_jumps       BOOLEAN DEFAULT FALSE,   -- 是否包含帧间跳变
    quality_score   FLOAT DEFAULT NULL,      -- 质量评分 [0, 1]

    -- ========== 数据范围（JSONB） ==========
    -- 存储每个关节的 qpos 范围，用于快速预览
    -- 格式: [{"joint": "joint_0", "min": -1.2, "max": 3.4}, ...]
    qpos_range      JSONB DEFAULT '[]',

    -- ========== 文件信息 ==========
    file_size       BIGINT,                  -- 文件大小（字节）
    checksum        VARCHAR(64),             -- SHA-256 校验和

    -- ========== 时间戳 ==========
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE episodes IS '轨迹元数据表：每行对应一个 HDF5/PKL 文件的元信息';
COMMENT ON COLUMN episodes.nq IS '广义坐标维度 = model.nq（来自 MuJoCo 模型）';
COMMENT ON COLUMN episodes.qpos_range IS 'JSONB 数组，记录每个关节的 qpos 最小/最大值';

-- 轨迹表的唯一约束：同一数据集内文件路径不能重复
ALTER TABLE episodes ADD CONSTRAINT uq_episodes_dataset_filepath
    UNIQUE (dataset_id, file_path);


-- ============================================================
-- 3. joint_schemas 表 —— 关节结构定义
-- ============================================================
-- 不同机器人的关节结构不同，这张表记录每种机器人的关节信息
-- 对应第 3 章中 model.jnt_type、model.jnt_qposadr 等信息

CREATE TABLE IF NOT EXISTS joint_schemas (
    id              BIGSERIAL PRIMARY KEY,

    -- 机器人类型（与 datasets.robot_type 关联）
    robot_type      VARCHAR(100) NOT NULL,

    -- 关节索引（从 0 开始）
    joint_index     INTEGER NOT NULL,

    -- 关节名称（如 "left_shoulder_yaw"）
    joint_name      VARCHAR(255) NOT NULL,

    -- 关节类型（MuJoCo 定义的 4 种类型）
    -- free=0, ball=1, slide=2, hinge=3
    joint_type      VARCHAR(20) NOT NULL
                    CHECK (joint_type IN ('free', 'ball', 'slide', 'hinge')),

    -- 该关节在 qpos 数组中的起始位置
    qpos_start      INTEGER NOT NULL,

    -- 该关节占据的 qpos 维度数
    -- hinge/slide=1, ball=4(四元数), free=7(位置3+四元数4)
    qpos_dim        INTEGER NOT NULL,

    -- 关节限位
    range_low       FLOAT DEFAULT NULL,
    range_high      FLOAT DEFAULT NULL,

    -- 元数据（如阻尼系数、摩擦等）
    metadata        JSONB DEFAULT '{}',

    created_at      TIMESTAMPTZ DEFAULT NOW(),

    -- 同一机器人的同一关节索引不能重复
    UNIQUE (robot_type, joint_index)
);

COMMENT ON TABLE joint_schemas IS '关节结构表：记录每种机器人的关节配置';
COMMENT ON COLUMN joint_schemas.qpos_start IS '该关节在 qpos 数组中的起始索引（第 3 章核心知识）';
COMMENT ON COLUMN joint_schemas.qpos_dim IS 'hinge/slide=1, ball=4, free=7';


-- ============================================================
-- 4. quality_reports 表 —— 质量报告
-- ============================================================
-- 第 6 章的数据校验结果，结构化存储到数据库中
-- 每个 episode 可以有多次质量检查报告（如每次流水线运行一次）

CREATE TABLE IF NOT EXISTS quality_reports (
    id              BIGSERIAL PRIMARY KEY,

    -- 关联的轨迹
    episode_id      BIGINT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    -- 总体质量评分 [0.0, 1.0]
    overall_score   FLOAT NOT NULL CHECK (overall_score >= 0 AND overall_score <= 1),

    -- ========== 各项指标 ==========
    nan_count       INTEGER DEFAULT 0,       -- NaN 数量
    jump_count      INTEGER DEFAULT 0,       -- 跳变次数
    limit_violations INTEGER DEFAULT 0,      -- 关节限位违规次数
    dead_joints     INTEGER DEFAULT 0,       -- 死关节数量

    -- 详细报告（JSONB，存储第 6 章 Validator 的完整输出）
    -- 结构示例:
    -- {
    --   "checks": [
    --     {"name": "nan_check", "passed": true, "severity": "ERROR"},
    --     {"name": "jump_check", "passed": false, "severity": "WARNING", "details": {...}}
    --   ],
    --   "summary": {"total": 9, "passed": 7, "failed": 2}
    -- }
    report_json     JSONB NOT NULL DEFAULT '{}',

    -- 校验器版本（便于追溯是哪个版本的校验逻辑生成的报告）
    validator_version VARCHAR(50) DEFAULT '1.0.0',

    -- 时间戳
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE quality_reports IS '质量报告表：存储第 6 章数据校验的结构化结果';


-- ============================================================
-- 5. training_runs 表 —— 训练记录
-- ============================================================
-- 将数据集与 ML 训练关联，追踪"哪些数据训练了哪个模型"

CREATE TABLE IF NOT EXISTS training_runs (
    id              BIGSERIAL PRIMARY KEY,

    -- 训练运行名称（如 "aloha_act_policy_v3_20240615"）
    name            VARCHAR(255) NOT NULL,

    -- 使用的数据集
    dataset_id      BIGINT REFERENCES datasets(id) ON DELETE SET NULL,

    -- 训练配置
    model_type      VARCHAR(100),            -- 如 "ACT", "Diffusion Policy"
    config          JSONB DEFAULT '{}',      -- 超参数、学习率等
    
    -- 数据筛选条件（记录训练时用了哪些筛选条件）
    -- 示例: {"min_quality_score": 0.8, "exclude_nan": true}
    data_filter     JSONB DEFAULT '{}',

    -- 训练结果
    status          VARCHAR(50) DEFAULT 'running'
                    CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    metrics         JSONB DEFAULT '{}',      -- 损失、精度等指标
    
    -- 使用的 episode 数量
    episode_count   INTEGER DEFAULT 0,

    -- 时间信息
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ DEFAULT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE training_runs IS '训练记录表：追踪数据集 → 模型训练的全链路';


-- ============================================================
-- 6. 标签系统 —— 灵活分类
-- ============================================================
-- 标签是比固定字段更灵活的分类方式
-- 示例标签: "high_quality", "needs_review", "demo_data", "augmented"

CREATE TABLE IF NOT EXISTS tags (
    id              BIGSERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL UNIQUE,
    description     TEXT DEFAULT '',
    color           VARCHAR(7) DEFAULT '#666666',   -- 前端展示用的颜色
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 数据集-标签 多对多关联表
CREATE TABLE IF NOT EXISTS dataset_tags (
    dataset_id      BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    tag_id          BIGINT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (dataset_id, tag_id)
);

-- 轨迹-标签 多对多关联表
CREATE TABLE IF NOT EXISTS episode_tags (
    episode_id      BIGINT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    tag_id          BIGINT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (episode_id, tag_id)
);

-- 轨迹标注（比标签更丰富，可以带自由文本）
CREATE TABLE IF NOT EXISTS episode_labels (
    id              BIGSERIAL PRIMARY KEY,
    episode_id      BIGINT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    label_key       VARCHAR(100) NOT NULL,    -- 如 "task_success", "grasp_quality"
    label_value     VARCHAR(255) NOT NULL,    -- 如 "true", "good", "3.5"
    labeled_by      VARCHAR(100) DEFAULT 'system',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (episode_id, label_key)
);

COMMENT ON TABLE tags IS '标签表：灵活的分类维度';
COMMENT ON TABLE episode_labels IS '轨迹标注表：键值对形式的精细标注';


-- ============================================================
-- 7. 视图 —— 封装复杂查询
-- ============================================================
-- 视图的好处: 上层 API 只需 SELECT * FROM view_name，不用写复杂 JOIN
-- Java 侧: 直接用 @Query("SELECT * FROM dataset_overview") 即可

-- 视图 1: 数据集概览（最常用）
CREATE OR REPLACE VIEW dataset_overview AS
SELECT
    d.id,
    d.name,
    d.robot_type,
    d.task,
    d.version,
    d.source,
    d.status,
    d.episode_count,
    d.total_size,
    -- 轨迹统计（从 episodes 表实时聚合）
    COUNT(e.id)                              AS actual_episode_count,
    COALESCE(SUM(e.file_size), 0)            AS actual_total_size,
    COALESCE(AVG(e.quality_score), 0)        AS avg_quality_score,
    COALESCE(MIN(e.quality_score), 0)        AS min_quality_score,
    -- NaN/跳变统计
    COUNT(CASE WHEN e.has_nan THEN 1 END)    AS nan_episode_count,
    COUNT(CASE WHEN e.has_jumps THEN 1 END)  AS jump_episode_count,
    -- 时间范围
    MIN(e.created_at)                        AS earliest_episode,
    MAX(e.created_at)                        AS latest_episode,
    d.created_at,
    d.updated_at
FROM datasets d
LEFT JOIN episodes e ON e.dataset_id = d.id
WHERE d.deleted_at IS NULL
GROUP BY d.id;

COMMENT ON VIEW dataset_overview IS '数据集概览视图：聚合了轨迹统计信息，API 层直接使用';


-- 视图 2: 质量摘要
CREATE OR REPLACE VIEW quality_summary AS
SELECT
    d.id                                     AS dataset_id,
    d.name                                   AS dataset_name,
    d.robot_type,
    COUNT(e.id)                              AS total_episodes,
    -- 质量分布
    COUNT(CASE WHEN e.quality_score >= 0.9 THEN 1 END)  AS excellent_count,
    COUNT(CASE WHEN e.quality_score >= 0.7
               AND e.quality_score < 0.9 THEN 1 END)    AS good_count,
    COUNT(CASE WHEN e.quality_score >= 0.5
               AND e.quality_score < 0.7 THEN 1 END)    AS fair_count,
    COUNT(CASE WHEN e.quality_score < 0.5 THEN 1 END)   AS poor_count,
    COUNT(CASE WHEN e.quality_score IS NULL THEN 1 END)  AS unscored_count,
    -- 问题统计
    COUNT(CASE WHEN e.has_nan THEN 1 END)                AS nan_episodes,
    COUNT(CASE WHEN e.has_jumps THEN 1 END)              AS jump_episodes,
    -- 平均质量
    ROUND(COALESCE(AVG(e.quality_score), 0)::NUMERIC, 4) AS avg_score,
    -- 最新质量报告
    MAX(qr.created_at)                                    AS last_quality_check
FROM datasets d
LEFT JOIN episodes e ON e.dataset_id = d.id
LEFT JOIN quality_reports qr ON qr.episode_id = e.id
WHERE d.deleted_at IS NULL
GROUP BY d.id, d.name, d.robot_type;

COMMENT ON VIEW quality_summary IS '质量摘要视图：按数据集统计质量分布';


-- ============================================================
-- 8. 索引 —— 加速高频查询
-- ============================================================
-- 索引策略:
--   - 外键字段: 必须加索引（PostgreSQL 不会自动为外键建索引）
--   - 高频筛选字段: robot_type, task, status, quality_score
--   - JSONB 字段: 使用 GIN 索引支持 @> 包含查询
--   - 不要过度索引: 每个索引都有写入开销

-- episodes 表索引
CREATE INDEX IF NOT EXISTS idx_episodes_dataset_id
    ON episodes(dataset_id);

CREATE INDEX IF NOT EXISTS idx_episodes_quality_score
    ON episodes(quality_score)
    WHERE quality_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_episodes_has_nan
    ON episodes(has_nan)
    WHERE has_nan = TRUE;

CREATE INDEX IF NOT EXISTS idx_episodes_has_jumps
    ON episodes(has_jumps)
    WHERE has_jumps = TRUE;

-- datasets 表索引
CREATE INDEX IF NOT EXISTS idx_datasets_robot_type
    ON datasets(robot_type);

CREATE INDEX IF NOT EXISTS idx_datasets_task
    ON datasets(task);

CREATE INDEX IF NOT EXISTS idx_datasets_status
    ON datasets(status);

-- 软删除过滤（大部分查询都需要排除已删除的记录）
CREATE INDEX IF NOT EXISTS idx_datasets_not_deleted
    ON datasets(id)
    WHERE deleted_at IS NULL;

-- JSONB GIN 索引（支持 metadata @> '{"camera_count": 2}' 这样的查询）
CREATE INDEX IF NOT EXISTS idx_datasets_metadata_gin
    ON datasets USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_episodes_qpos_range_gin
    ON episodes USING GIN (qpos_range);

-- joint_schemas 索引
CREATE INDEX IF NOT EXISTS idx_joint_schemas_robot_type
    ON joint_schemas(robot_type);

-- quality_reports 索引
CREATE INDEX IF NOT EXISTS idx_quality_reports_episode_id
    ON quality_reports(episode_id);

CREATE INDEX IF NOT EXISTS idx_quality_reports_score
    ON quality_reports(overall_score);

-- training_runs 索引
CREATE INDEX IF NOT EXISTS idx_training_runs_dataset_id
    ON training_runs(dataset_id);

-- 标签索引
CREATE INDEX IF NOT EXISTS idx_dataset_tags_tag_id
    ON dataset_tags(tag_id);

CREATE INDEX IF NOT EXISTS idx_episode_tags_tag_id
    ON episode_tags(tag_id);

CREATE INDEX IF NOT EXISTS idx_episode_labels_key
    ON episode_labels(label_key);


-- ============================================================
-- 9. 触发器 —— 自动更新 updated_at
-- ============================================================
-- Java 里可以用 @PreUpdate 注解实现，SQL 层用触发器更可靠

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_episodes_updated_at
    BEFORE UPDATE ON episodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================================
-- 10. 示例数据插入
-- ============================================================

-- 插入标签
INSERT INTO tags (name, description, color) VALUES
    ('high_quality', '质量评分 >= 0.9', '#22c55e'),
    ('needs_review', '需要人工审查', '#f59e0b'),
    ('demo_data', '演示用数据', '#3b82f6'),
    ('augmented', '数据增强生成', '#a855f7'),
    ('production', '生产环境可用', '#10b981')
ON CONFLICT (name) DO NOTHING;

-- 插入数据集
INSERT INTO datasets (name, robot_type, task, description, version, source) VALUES
    ('aloha_cup_stacking_v2', 'aloha', 'cup_stacking',
     'ALOHA 双臂机器人杯子堆叠任务，50 Hz 采集频率，含双目 RGB 图像',
     '2.0.0', 'real'),
    ('franka_pick_place_sim', 'franka', 'pick_and_place',
     'Franka Panda 单臂抓取放置仿真数据，MuJoCo 生成',
     '1.0.0', 'simulation'),
    ('ur5e_assembly_v1', 'ur5e', 'assembly',
     'UR5e 装配任务真机数据',
     '1.0.0', 'real')
ON CONFLICT (name) DO NOTHING;

-- 插入关节结构（以 ALOHA 为例，简化版）
INSERT INTO joint_schemas (robot_type, joint_index, joint_name, joint_type, qpos_start, qpos_dim, range_low, range_high) VALUES
    ('aloha', 0, 'left_waist',         'hinge', 0,  1, -3.14159, 3.14159),
    ('aloha', 1, 'left_shoulder',      'hinge', 1,  1, -1.7628,  1.7628),
    ('aloha', 2, 'left_elbow',         'hinge', 2,  1, -1.7628,  1.7628),
    ('aloha', 3, 'left_forearm_roll',  'hinge', 3,  1, -3.14159, 3.14159),
    ('aloha', 4, 'left_wrist_angle',   'hinge', 4,  1, -1.8675,  1.8675),
    ('aloha', 5, 'left_wrist_rotate',  'hinge', 5,  1, -3.14159, 3.14159),
    ('aloha', 6, 'left_gripper',       'hinge', 6,  1, 0.0,      1.0),
    ('aloha', 7, 'right_waist',        'hinge', 7,  1, -3.14159, 3.14159),
    ('aloha', 8, 'right_shoulder',     'hinge', 8,  1, -1.7628,  1.7628),
    ('aloha', 9, 'right_elbow',        'hinge', 9,  1, -1.7628,  1.7628),
    ('aloha', 10, 'right_forearm_roll','hinge', 10, 1, -3.14159, 3.14159),
    ('aloha', 11, 'right_wrist_angle', 'hinge', 11, 1, -1.8675,  1.8675),
    ('aloha', 12, 'right_wrist_rotate','hinge', 12, 1, -3.14159, 3.14159),
    ('aloha', 13, 'right_gripper',     'hinge', 13, 1, 0.0,      1.0)
ON CONFLICT (robot_type, joint_index) DO NOTHING;


-- ============================================================
-- 11. 常用查询示例
-- ============================================================
-- 以下是 API 层最常见的查询，直接复制到 MyBatis XML 或 JPA @Query 中使用

-- 查询 1: 获取数据集列表（带分页和筛选）
-- 对应 API: GET /api/datasets?robot_type=aloha&page=1&size=20
SELECT * FROM dataset_overview
WHERE robot_type = 'aloha'
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;

-- 查询 2: 获取数据集的所有轨迹（按质量排序）
-- 对应 API: GET /api/datasets/1/episodes?sort=quality_score&order=desc
SELECT
    e.id, e.file_path, e.num_steps, e.duration,
    e.quality_score, e.has_nan, e.has_jumps,
    e.nq, e.nv, e.nu
FROM episodes e
WHERE e.dataset_id = 1
ORDER BY e.quality_score DESC NULLS LAST
LIMIT 50 OFFSET 0;

-- 查询 3: 筛选高质量轨迹（用于训练）
-- 对应 API: POST /api/episodes/search
SELECT e.*
FROM episodes e
JOIN datasets d ON d.id = e.dataset_id
WHERE d.robot_type = 'aloha'
  AND e.quality_score >= 0.8
  AND e.has_nan = FALSE
  AND e.has_jumps = FALSE
  AND d.deleted_at IS NULL
ORDER BY e.quality_score DESC;

-- 查询 4: 数据集质量概览
-- 对应 API: GET /api/datasets/1/quality
SELECT * FROM quality_summary
WHERE dataset_id = 1;

-- 查询 5: 获取机器人的关节结构
-- 对应 API: GET /api/schemas/aloha
SELECT
    joint_index, joint_name, joint_type,
    qpos_start, qpos_dim,
    range_low, range_high
FROM joint_schemas
WHERE robot_type = 'aloha'
ORDER BY joint_index;

-- 查询 6: 查找包含 NaN 的轨迹（数据治理用）
SELECT
    d.name AS dataset_name,
    e.id AS episode_id,
    e.file_path,
    qr.nan_count,
    qr.overall_score
FROM episodes e
JOIN datasets d ON d.id = e.dataset_id
LEFT JOIN quality_reports qr ON qr.episode_id = e.id
WHERE e.has_nan = TRUE
ORDER BY qr.nan_count DESC;

-- 查询 7: JSONB 查询 —— 查找特定关节的 qpos 范围
-- 这展示了 JSONB 的强大之处: 可以查询嵌套在 JSON 中的数据
SELECT
    e.id,
    e.file_path,
    jsonb_array_elements(e.qpos_range) AS joint_range
FROM episodes e
WHERE e.dataset_id = 1
  AND e.qpos_range @> '[{"joint": "left_gripper"}]';

-- 查询 8: 统计各机器人类型的数据量
SELECT
    robot_type,
    COUNT(*)                             AS dataset_count,
    SUM(episode_count)                   AS total_episodes,
    pg_size_pretty(SUM(total_size))      AS total_size_human
FROM datasets
WHERE deleted_at IS NULL
GROUP BY robot_type
ORDER BY total_episodes DESC;

-- 查询 9: 查找最近的训练记录
SELECT
    tr.name AS run_name,
    d.name AS dataset_name,
    tr.model_type,
    tr.status,
    tr.episode_count,
    tr.metrics->>'loss' AS final_loss,
    tr.started_at,
    tr.completed_at
FROM training_runs tr
LEFT JOIN datasets d ON d.id = tr.dataset_id
ORDER BY tr.started_at DESC
LIMIT 10;

-- 查询 10: 按标签筛选数据集
SELECT d.*
FROM datasets d
JOIN dataset_tags dt ON dt.dataset_id = d.id
JOIN tags t ON t.id = dt.tag_id
WHERE t.name = 'high_quality'
  AND d.deleted_at IS NULL;


-- ============================================================
-- 12. 数据完整性检查（DBA 运维用）
-- ============================================================

-- 检查 episode_count 冗余字段是否与实际一致
SELECT
    d.id,
    d.name,
    d.episode_count AS cached_count,
    COUNT(e.id) AS actual_count,
    CASE WHEN d.episode_count != COUNT(e.id) THEN '❌ 不一致' ELSE '✅ 一致' END AS status
FROM datasets d
LEFT JOIN episodes e ON e.dataset_id = d.id
WHERE d.deleted_at IS NULL
GROUP BY d.id
HAVING d.episode_count != COUNT(e.id);

-- 同步冗余字段（定时任务或手动执行）
UPDATE datasets d SET
    episode_count = sub.cnt,
    total_size = sub.total_bytes,
    updated_at = NOW()
FROM (
    SELECT
        dataset_id,
        COUNT(*) AS cnt,
        COALESCE(SUM(file_size), 0) AS total_bytes
    FROM episodes
    GROUP BY dataset_id
) sub
WHERE d.id = sub.dataset_id
  AND (d.episode_count != sub.cnt OR d.total_size != sub.total_bytes);
