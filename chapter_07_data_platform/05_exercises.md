# 第 7 章 · 05 — 练习

> 四个递进式练习，将数据平台设计能力付诸实践。

## 练习概览

| # | 练习 | 核心技能 |
| --: | :--- | :------- |
| 1 | 多模态数据 Schema 设计 | 模态抽象、JSONB 查询、数据量估算 |
| 2 | 数据版本管理系统 | Schema 变更追踪、兼容性检查、变更日志 |
| 3 | 缓存层实现 | LRU 缓存、命中率统计、Decorator 模式 |
| 4 | 数据迁移脚本 | v1→v2 迁移、字段回填、验证、可回滚 |

---

## 练习 1: 多模态数据 Schema 设计

### 场景

真实机器人不只有 qpos，还有摄像头图像、力传感器、麦克风音频等。

### 模态类型

```python
class ModalityType(Enum):
    QPOS, QVEL, IMAGE, DEPTH, AUDIO, FORCE, ACTION
```

### ModalitySchema 定义

每个模态独立描述：modality_type、shape、sampling_rate、storage_path、metadata。

### MultiModalEpisode

一条轨迹包含多个模态，支持：
- `has_modality(type)` — 检查是否包含某种模态
- `find_multimodal([QPOS, IMAGE, AUDIO])` — 查找同时包含所有指定模态的轨迹

### 数据量估算

一条 300 帧的多模态轨迹：

| 模态 | 估算大小 |
| :--- | -------: |
| qpos (14 joints) | ~34 KB |
| 2× RGB 相机 (480×640) | ~553 MB |
| 深度图 | ~368 MB |
| 音频 (16kHz) | ~375 KB |
| 力传感器 | ~7 KB |

---

## 练习 2: 数据版本管理系统

### VersionManager 功能

- 注册 Schema 版本（字段列表 + 变更描述）
- 注册数据集版本（关联 Schema 版本 + changelog）
- **兼容性检查**：比较两个 Schema 版本
  - 向后兼容条件：没有删除字段 + 没有类型变更
  - 新增字段 = 兼容；删除字段 = 不兼容
- 数据集版本历史（支持 parent_version 链式追溯）
- 变更日志

---

## 练习 3: 缓存层实现

### LRU 缓存

基于 `OrderedDict` 实现：
- **双淘汰策略**: 条目数上限 + 内存上限
- 命中时 `move_to_end`，满时 `popitem(last=False)` 淘汰最久未用

### CachedEpisodeLoader

Decorator 模式：在原始文件加载器外包一层缓存。

```python
loader = CachedEpisodeLoader(cache, data_dir)
data = loader.load_qpos("episode_0001")  # 第一次: miss, 从文件加载
data = loader.load_qpos("episode_0001")  # 第二次: hit, 从缓存返回
```

### 缓存统计

```python
stats = cache.stats
# hits, misses, hit_rate, evictions, memory_usage, ...
```

---

## 练习 4: 数据迁移脚本 (v1 → v2)

### 迁移流程

```
读取 v1 数据 → 从原始文件提取缺失字段 → 计算派生字段 → 写入 v2 → 验证
```

### v2 新增字段

`nv`, `nu`, `duration`, `has_nan`, `has_jumps`, `quality_score`, `file_size`, `checksum`

### 验证规则

- 原始字段完整保留（id, dataset_id, file_path, nq, num_steps）
- 新字段合理（duration > 0, quality_score ∈ [0,1], checksum 非空）

### 对应 SQL

```sql
ALTER TABLE episodes ADD COLUMN nv INTEGER;
ALTER TABLE episodes ADD COLUMN has_nan BOOLEAN DEFAULT FALSE;
-- ... 然后用 Python 脚本生成 UPDATE 语句回填
```

---

## 扩展思考

| 练习 | 思考题 |
| :--- | :----- |
| 1 | TB 级图像数据的存储策略？→ 图像与 qpos 分离，图像用对象存储 + CDN |
| 2 | 数据集「分支」功能？→ 参考 DVC (Data Version Control) |
| 3 | 分布式缓存一致性？→ Redis + TTL + 主动失效 |
| 4 | 迁移中断恢复？→ 事务 + WAL + 断点续传 |
