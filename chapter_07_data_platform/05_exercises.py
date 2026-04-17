"""
第 7 章 · 05 - 练习 (Exercises)

四个递进式练习，将本章学到的数据平台设计能力付诸实践。

练习 1: 多模态数据 Schema 设计
  → 扩展数据库 Schema，支持图像 + qpos + 音频的多模态数据

练习 2: 数据版本管理系统
  → 追踪 Schema 变更、数据集版本、变更日志

练习 3: 缓存层实现
  → 用 LRU 缓存加速频繁访问的轨迹数据

练习 4: 数据迁移脚本
  → 从 v1 Schema 迁移到 v2 Schema，处理字段变更和数据转换

运行: python 05_exercises.py
依赖: pip install numpy
"""

import numpy as np
import json
import time
import hashlib
import os
import tempfile
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


# ############################################################
#
#  练习 1: 多模态数据 Schema 设计
#
#  场景: 真实的机器人数据不仅有关节角度（qpos），
#        还有摄像头图像、力传感器、麦克风音频等。
#        你需要设计一个能存储和查询多模态数据的 Schema。
#
# ############################################################

class ModalityType(Enum):
    """数据模态类型"""
    QPOS = "qpos"
    QVEL = "qvel"
    IMAGE = "image"
    DEPTH = "depth"
    AUDIO = "audio"
    FORCE = "force"
    ACTION = "action"


@dataclass
class ModalitySchema:
    """
    单个模态的 Schema 定义。

    对应 SQL:
        CREATE TABLE modality_schemas (
            id BIGSERIAL PRIMARY KEY,
            episode_id BIGINT REFERENCES episodes(id),
            modality_type VARCHAR(50) NOT NULL,
            name VARCHAR(255) NOT NULL,
            data_type VARCHAR(50),
            shape JSONB,
            sampling_rate FLOAT,
            storage_path VARCHAR(1024),
            metadata JSONB DEFAULT '{}'
        );
    """
    modality_type: ModalityType
    name: str
    data_type: str = "float32"
    shape: List[int] = field(default_factory=list)
    sampling_rate: Optional[float] = None    # Hz
    storage_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalEpisode:
    """
    多模态轨迹: 一条轨迹包含多个模态的数据。

    对应 SQL:
        CREATE TABLE multimodal_episodes (
            id BIGSERIAL PRIMARY KEY,
            dataset_id BIGINT REFERENCES datasets(id),
            num_steps INTEGER,
            modalities JSONB,      -- 包含哪些模态
            sync_status VARCHAR(50), -- 时间同步状态
            total_size BIGINT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """
    id: int = 0
    dataset_id: int = 0
    num_steps: int = 0
    modalities: Dict[str, ModalitySchema] = field(default_factory=dict)
    sync_status: str = "synced"
    total_size: int = 0
    created_at: str = ""

    def add_modality(self, schema: ModalitySchema):
        """添加一个模态"""
        self.modalities[schema.name] = schema

    def has_modality(self, modality_type: ModalityType) -> bool:
        """检查是否包含某种模态"""
        return any(
            m.modality_type == modality_type
            for m in self.modalities.values()
        )

    def get_modalities_by_type(self, modality_type: ModalityType) -> List[ModalitySchema]:
        """获取指定类型的所有模态"""
        return [
            m for m in self.modalities.values()
            if m.modality_type == modality_type
        ]

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "num_steps": self.num_steps,
            "modality_count": len(self.modalities),
            "modalities": {
                name: {
                    "type": m.modality_type.value,
                    "shape": m.shape,
                    "rate": m.sampling_rate,
                }
                for name, m in self.modalities.items()
            },
            "total_size": self.total_size,
        }


class MultiModalStore:
    """多模态数据的查询接口"""

    def __init__(self):
        self._episodes: Dict[int, MultiModalEpisode] = {}
        self._next_id = 1

    def add(self, ep: MultiModalEpisode) -> MultiModalEpisode:
        ep.id = self._next_id
        self._next_id += 1
        ep.created_at = datetime.now().isoformat()
        self._episodes[ep.id] = ep
        return ep

    def find_by_modality(self, modality_type: ModalityType) -> List[MultiModalEpisode]:
        """查找包含指定模态的所有轨迹"""
        return [
            ep for ep in self._episodes.values()
            if ep.has_modality(modality_type)
        ]

    def find_multimodal(self, required: List[ModalityType]) -> List[MultiModalEpisode]:
        """查找同时包含所有指定模态的轨迹"""
        return [
            ep for ep in self._episodes.values()
            if all(ep.has_modality(m) for m in required)
        ]


def exercise_1():
    """练习 1: 多模态数据 Schema 设计"""
    print(f"\n{'练习 1: 多模态数据 Schema 设计':=^60}")
    print("""
场景: ALOHA 机器人配备了两个 RGB 摄像头、一个麦克风、力传感器。
需要设计 Schema 来存储和查询这些多模态数据。
""")

    store = MultiModalStore()

    # 创建一条多模态轨迹
    ep = MultiModalEpisode(dataset_id=1, num_steps=300)

    # 添加 qpos 模态
    ep.add_modality(ModalitySchema(
        modality_type=ModalityType.QPOS,
        name="qpos",
        data_type="float64",
        shape=[300, 14],
        sampling_rate=50.0,
    ))

    # 添加左摄像头图像
    ep.add_modality(ModalitySchema(
        modality_type=ModalityType.IMAGE,
        name="cam_left_rgb",
        data_type="uint8",
        shape=[300, 480, 640, 3],
        sampling_rate=30.0,
        metadata={"encoding": "jpeg", "camera_id": "left"},
    ))

    # 添加右摄像头图像
    ep.add_modality(ModalitySchema(
        modality_type=ModalityType.IMAGE,
        name="cam_right_rgb",
        data_type="uint8",
        shape=[300, 480, 640, 3],
        sampling_rate=30.0,
        metadata={"encoding": "jpeg", "camera_id": "right"},
    ))

    # 添加深度图
    ep.add_modality(ModalitySchema(
        modality_type=ModalityType.DEPTH,
        name="cam_left_depth",
        data_type="float32",
        shape=[300, 480, 640],
        sampling_rate=30.0,
    ))

    # 添加音频
    ep.add_modality(ModalitySchema(
        modality_type=ModalityType.AUDIO,
        name="microphone",
        data_type="float32",
        shape=[300 * 320],  # 50Hz * 300 steps = 15000 帧, 16kHz 音频
        sampling_rate=16000.0,
        metadata={"channels": 1, "format": "wav"},
    ))

    # 添加力传感器
    ep.add_modality(ModalitySchema(
        modality_type=ModalityType.FORCE,
        name="wrist_force",
        data_type="float32",
        shape=[300, 6],
        sampling_rate=500.0,
        metadata={"unit": "N", "dimensions": ["fx", "fy", "fz", "tx", "ty", "tz"]},
    ))

    # 计算估计大小
    size_estimate = (
        300 * 14 * 8 +             # qpos: 300 steps × 14 joints × 8 bytes
        2 * 300 * 480 * 640 * 3 +  # 两个 RGB 相机
        300 * 480 * 640 * 4 +      # 深度图
        300 * 320 * 4 +            # 音频
        300 * 6 * 4                # 力传感器
    )
    ep.total_size = size_estimate
    ep = store.add(ep)

    # 展示 Schema
    print("多模态轨迹 Schema:")
    summary = ep.summary()
    print(f"  ID: {summary['id']}")
    print(f"  帧数: {summary['num_steps']}")
    print(f"  模态数: {summary['modality_count']}")
    print(f"  估计大小: {summary['total_size'] / 1024 / 1024:.1f} MB")
    print(f"\n模态详情:")
    for name, info in summary["modalities"].items():
        print(f"  {name}:")
        print(f"    类型: {info['type']}")
        print(f"    形状: {info['shape']}")
        print(f"    采样率: {info['rate']} Hz")

    # 再添加几条用于查询演示
    ep2 = MultiModalEpisode(dataset_id=1, num_steps=200)
    ep2.add_modality(ModalitySchema(ModalityType.QPOS, "qpos", shape=[200, 14], sampling_rate=50.0))
    ep2.add_modality(ModalitySchema(ModalityType.IMAGE, "cam_left_rgb", shape=[200, 480, 640, 3], sampling_rate=30.0))
    store.add(ep2)

    ep3 = MultiModalEpisode(dataset_id=1, num_steps=150)
    ep3.add_modality(ModalitySchema(ModalityType.QPOS, "qpos", shape=[150, 14], sampling_rate=50.0))
    store.add(ep3)

    # 查询演示
    print(f"\n查询演示:")
    image_eps = store.find_by_modality(ModalityType.IMAGE)
    print(f"  包含图像的轨迹: {len(image_eps)} 条")

    audio_eps = store.find_by_modality(ModalityType.AUDIO)
    print(f"  包含音频的轨迹: {len(audio_eps)} 条")

    full_mm = store.find_multimodal([ModalityType.QPOS, ModalityType.IMAGE, ModalityType.AUDIO])
    print(f"  同时包含 qpos + 图像 + 音频: {len(full_mm)} 条")

    # 对应的 SQL 查询
    print(f"""
对应的 SQL 查询:
  -- 查找包含图像数据的轨迹
  SELECT e.* FROM multimodal_episodes e
  WHERE e.modalities @> '{{"cam_left_rgb": {{"type": "image"}}}}';

  -- 查找同时有 qpos 和图像的轨迹
  SELECT e.* FROM multimodal_episodes e
  WHERE e.modalities ? 'qpos'
    AND e.modalities ? 'cam_left_rgb';
""")


# ############################################################
#
#  练习 2: 数据版本管理系统
#
#  场景: Schema 会随着业务演进而变化，数据集也会更新。
#        你需要一个版本管理系统来追踪这些变更。
#
# ############################################################

@dataclass
class SchemaVersion:
    """Schema 版本记录"""
    version: str
    changes: List[str]
    fields: Dict[str, str]    # 字段名 → 类型
    created_at: str = ""
    created_by: str = "system"


@dataclass
class DatasetVersion:
    """数据集版本记录"""
    dataset_name: str
    version: str
    schema_version: str
    episode_count: int
    changelog: str
    parent_version: Optional[str] = None
    created_at: str = ""


class VersionManager:
    """
    数据版本管理器。

    职责:
      1. 追踪 Schema 变更历史
      2. 追踪数据集版本
      3. 检查 Schema 兼容性
      4. 生成变更日志

    Java 等价: Flyway / Liquibase 的概念，但应用于数据集而非数据库
    """

    def __init__(self):
        self._schema_versions: List[SchemaVersion] = []
        self._dataset_versions: Dict[str, List[DatasetVersion]] = {}
        self._change_log: List[Dict[str, Any]] = []

    def register_schema_version(
        self,
        version: str,
        fields: Dict[str, str],
        changes: List[str],
    ) -> SchemaVersion:
        """注册一个新的 Schema 版本"""
        sv = SchemaVersion(
            version=version,
            changes=changes,
            fields=fields,
            created_at=datetime.now().isoformat(),
        )
        self._schema_versions.append(sv)
        self._log_change("schema_version", f"新增 Schema 版本 {version}", {"changes": changes})
        return sv

    def register_dataset_version(
        self,
        dataset_name: str,
        version: str,
        schema_version: str,
        episode_count: int,
        changelog: str,
    ) -> DatasetVersion:
        """注册数据集的新版本"""
        if dataset_name not in self._dataset_versions:
            self._dataset_versions[dataset_name] = []

        parent = None
        if self._dataset_versions[dataset_name]:
            parent = self._dataset_versions[dataset_name][-1].version

        dv = DatasetVersion(
            dataset_name=dataset_name,
            version=version,
            schema_version=schema_version,
            episode_count=episode_count,
            changelog=changelog,
            parent_version=parent,
            created_at=datetime.now().isoformat(),
        )
        self._dataset_versions[dataset_name].append(dv)
        self._log_change(
            "dataset_version",
            f"数据集 {dataset_name} 更新到 {version}",
            {"changelog": changelog},
        )
        return dv

    def check_compatibility(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """检查两个 Schema 版本的兼容性"""
        sv_a = self._find_schema(version_a)
        sv_b = self._find_schema(version_b)

        if sv_a is None or sv_b is None:
            return {"compatible": False, "reason": "版本不存在"}

        fields_a = set(sv_a.fields.keys())
        fields_b = set(sv_b.fields.keys())

        added = fields_b - fields_a
        removed = fields_a - fields_b
        common = fields_a & fields_b

        type_changes = []
        for f in common:
            if sv_a.fields[f] != sv_b.fields[f]:
                type_changes.append({
                    "field": f,
                    "old_type": sv_a.fields[f],
                    "new_type": sv_b.fields[f],
                })

        # 向后兼容条件: 没有删除字段，没有类型变更
        backward_compatible = len(removed) == 0 and len(type_changes) == 0

        return {
            "compatible": backward_compatible,
            "added_fields": list(added),
            "removed_fields": list(removed),
            "type_changes": type_changes,
            "summary": (
                "向后兼容" if backward_compatible
                else f"不兼容: 删除了 {len(removed)} 个字段, {len(type_changes)} 个类型变更"
            ),
        }

    def get_dataset_history(self, dataset_name: str) -> List[DatasetVersion]:
        """获取数据集的版本历史"""
        return self._dataset_versions.get(dataset_name, [])

    def get_change_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取变更日志"""
        return self._change_log[-limit:]

    def _find_schema(self, version: str) -> Optional[SchemaVersion]:
        for sv in self._schema_versions:
            if sv.version == version:
                return sv
        return None

    def _log_change(self, change_type: str, description: str, details: Dict[str, Any]):
        self._change_log.append({
            "type": change_type,
            "description": description,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })


def exercise_2():
    """练习 2: 数据版本管理"""
    print(f"\n{'练习 2: 数据版本管理系统':=^60}")
    print("""
场景: 随着平台演进，episodes 表的 Schema 经历了多次变更。
你需要管理这些变更，确保新旧数据的兼容性。
""")

    vm = VersionManager()

    # 注册 Schema v1.0
    v1 = vm.register_schema_version(
        version="1.0.0",
        fields={
            "id": "BIGINT",
            "dataset_id": "BIGINT",
            "file_path": "VARCHAR",
            "nq": "INTEGER",
            "num_steps": "INTEGER",
            "timestep": "FLOAT",
        },
        changes=["初始版本"],
    )

    # 注册 Schema v1.1（新增字段 —— 向后兼容）
    v1_1 = vm.register_schema_version(
        version="1.1.0",
        fields={
            "id": "BIGINT",
            "dataset_id": "BIGINT",
            "file_path": "VARCHAR",
            "nq": "INTEGER",
            "nv": "INTEGER",          # 新增
            "nu": "INTEGER",          # 新增
            "num_steps": "INTEGER",
            "timestep": "FLOAT",
            "duration": "FLOAT",      # 新增
            "quality_score": "FLOAT", # 新增
        },
        changes=["新增 nv, nu 维度字段", "新增 duration 计算字段", "新增 quality_score"],
    )

    # 注册 Schema v2.0（有字段删除 —— 不兼容）
    v2 = vm.register_schema_version(
        version="2.0.0",
        fields={
            "id": "BIGINT",
            "dataset_id": "BIGINT",
            "file_path": "VARCHAR",
            "nq": "INTEGER",
            "nv": "INTEGER",
            "nu": "INTEGER",
            "num_steps": "INTEGER",
            "timestep": "FLOAT",
            "duration": "FLOAT",
            "quality_score": "FLOAT",
            "has_nan": "BOOLEAN",     # 新增
            "has_jumps": "BOOLEAN",   # 新增
            "qpos_range": "JSONB",    # 新增，timestep 类型不变
            "file_size": "BIGINT",    # 新增
            "checksum": "VARCHAR",    # 新增
        },
        changes=[
            "新增 has_nan, has_jumps 质量标记",
            "新增 qpos_range JSONB 字段",
            "新增 file_size, checksum 文件信息",
        ],
    )

    # 注册数据集版本
    vm.register_dataset_version(
        "aloha_cup_stacking", "1.0.0", "1.0.0",
        episode_count=100,
        changelog="初始数据采集，100 条真机轨迹",
    )
    vm.register_dataset_version(
        "aloha_cup_stacking", "1.1.0", "1.1.0",
        episode_count=250,
        changelog="新增 150 条轨迹，补充 nv/nu 维度信息",
    )
    vm.register_dataset_version(
        "aloha_cup_stacking", "2.0.0", "2.0.0",
        episode_count=230,
        changelog="升级到 v2 Schema，移除 20 条低质量轨迹，新增质量评分",
    )

    # 兼容性检查
    print("Schema 兼容性检查:")

    compat_1_to_1_1 = vm.check_compatibility("1.0.0", "1.1.0")
    print(f"\n  v1.0 → v1.1: {compat_1_to_1_1['summary']}")
    if compat_1_to_1_1["added_fields"]:
        print(f"    新增字段: {compat_1_to_1_1['added_fields']}")

    compat_1_1_to_2 = vm.check_compatibility("1.1.0", "2.0.0")
    print(f"\n  v1.1 → v2.0: {compat_1_1_to_2['summary']}")
    if compat_1_1_to_2["added_fields"]:
        print(f"    新增字段: {compat_1_1_to_2['added_fields']}")

    # 版本历史
    print(f"\n数据集 'aloha_cup_stacking' 版本历史:")
    for dv in vm.get_dataset_history("aloha_cup_stacking"):
        print(f"  v{dv.version} (Schema v{dv.schema_version})")
        print(f"    轨迹数: {dv.episode_count}")
        print(f"    变更: {dv.changelog}")
        if dv.parent_version:
            print(f"    父版本: v{dv.parent_version}")

    # 变更日志
    print(f"\n变更日志 (最近 5 条):")
    for entry in vm.get_change_log(5):
        print(f"  [{entry['timestamp'][:19]}] {entry['description']}")


# ############################################################
#
#  练习 3: 缓存层实现
#
#  场景: 训练系统频繁读取热门轨迹的 qpos 数据，
#        每次都从文件加载太慢了。需要一个缓存层。
#
# ############################################################

class LRUCache:
    """
    LRU (Least Recently Used) 缓存。

    基于 OrderedDict 实现，当缓存满时淘汰最久未访问的条目。

    Java 等价:
        // Guava Cache
        LoadingCache<String, byte[]> cache = CacheBuilder.newBuilder()
            .maximumSize(100)
            .expireAfterAccess(10, TimeUnit.MINUTES)
            .recordStats()
            .build(new CacheLoader<>() {
                public byte[] load(String key) { return loadFromDisk(key); }
            });

        // 或者 Caffeine
        Cache<String, byte[]> cache = Caffeine.newBuilder()
            .maximumSize(100)
            .expireAfterAccess(Duration.ofMinutes(10))
            .build();
    """

    def __init__(self, max_size: int = 100, max_memory_bytes: int = 500 * 1024 * 1024):
        self._max_size = max_size
        self._max_memory = max_memory_bytes
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._current_memory = 0

        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项（命中时移到最近使用位置）"""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any, size_bytes: int = 0):
        """放入缓存（满时淘汰最久未使用的）"""
        if key in self._cache:
            self._current_memory -= self._sizes.get(key, 0)
            del self._cache[key]

        # 淘汰策略: 直到腾出足够空间
        while (
            len(self._cache) >= self._max_size
            or self._current_memory + size_bytes > self._max_memory
        ) and self._cache:
            evicted_key, _ = self._cache.popitem(last=False)
            self._current_memory -= self._sizes.pop(evicted_key, 0)
            self._evictions += 1

        self._cache[key] = value
        self._sizes[key] = size_bytes
        self._current_memory += size_bytes

    def invalidate(self, key: str) -> bool:
        """使缓存失效"""
        if key in self._cache:
            self._current_memory -= self._sizes.pop(key, 0)
            del self._cache[key]
            return True
        return False

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._sizes.clear()
        self._current_memory = 0

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "memory_usage": self._current_memory,
            "max_memory": self._max_memory,
            "memory_usage_pct": (
                self._current_memory / self._max_memory * 100
                if self._max_memory > 0 else 0
            ),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "evictions": self._evictions,
        }


class CachedEpisodeLoader:
    """
    带缓存的轨迹数据加载器。

    在原始加载器外面包一层缓存，对调用方完全透明。
    这是 Decorator 模式的经典应用。

    Java 等价:
        @Service
        public class CachedEpisodeLoader implements EpisodeLoader {
            @Autowired private EpisodeLoader delegate;
            @Autowired private Cache<String, double[][]> cache;

            @Override
            @Cacheable(value = "episodes", key = "#episodeId")
            public double[][] loadQpos(long episodeId) {
                return delegate.loadQpos(episodeId);
            }
        }
    """

    def __init__(self, cache: LRUCache, data_dir: str):
        self._cache = cache
        self._data_dir = data_dir

    def load_qpos(self, episode_id: str) -> Optional[np.ndarray]:
        """加载 qpos 数据（先查缓存，未命中则从文件加载）"""
        cache_key = f"qpos:{episode_id}"

        # 查缓存
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # 从文件加载
        data = self._load_from_file(episode_id)
        if data is not None:
            size_bytes = data.nbytes
            self._cache.put(cache_key, data, size_bytes)

        return data

    def _load_from_file(self, episode_id: str) -> Optional[np.ndarray]:
        """从文件系统加载数据"""
        file_path = os.path.join(self._data_dir, f"{episode_id}.pkl")
        if not os.path.exists(file_path):
            return None

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "qpos" in data:
            return np.asarray(data["qpos"])
        return None

    @property
    def cache_stats(self) -> Dict[str, Any]:
        return self._cache.stats


def exercise_3():
    """练习 3: 缓存层实现"""
    print(f"\n{'练习 3: 缓存层实现':=^60}")
    print("""
场景: 训练系统在一个 epoch 中会反复读取同一批轨迹数据。
直接从文件加载太慢，用 LRU 缓存加速。
""")

    with tempfile.TemporaryDirectory(prefix="ch07_cache_") as tmpdir:
        # 生成测试数据文件
        episode_ids = []
        for i in range(20):
            eid = f"episode_{i:04d}"
            episode_ids.append(eid)
            data = {"qpos": np.random.randn(200, 7) * 0.5}
            with open(os.path.join(tmpdir, f"{eid}.pkl"), 'wb') as f:
                pickle.dump(data, f)

        # 创建缓存（最多 10 个条目，最多 5MB）
        cache = LRUCache(max_size=10, max_memory_bytes=5 * 1024 * 1024)
        loader = CachedEpisodeLoader(cache, tmpdir)

        # 模拟训练过程的访问模式
        print("模拟训练访问模式 (3 个 epoch):\n")

        for epoch in range(3):
            # 每个 epoch 访问前 8 个 episode（模拟数据子集）
            start = time.time()
            for eid in episode_ids[:8]:
                data = loader.load_qpos(eid)
                assert data is not None

            elapsed = time.time() - start
            stats = loader.cache_stats

            print(f"  Epoch {epoch + 1}:")
            print(f"    耗时: {elapsed:.4f}s")
            print(f"    缓存命中率: {stats['hit_rate']:.1%}")
            print(f"    缓存大小: {stats['size']}/{stats['max_size']}")
            print(f"    内存使用: {stats['memory_usage'] / 1024:.1f} KB")

        # 展示最终缓存统计
        print(f"\n最终缓存统计:")
        final_stats = loader.cache_stats
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        # 演示缓存淘汰
        print(f"\n--- 缓存淘汰演示 ---")
        print(f"当前缓存大小: {cache.stats['size']}")

        # 访问新的 episode（触发淘汰）
        for eid in episode_ids[10:18]:
            loader.load_qpos(eid)

        print(f"访问 8 个新 episode 后:")
        print(f"  缓存大小: {cache.stats['size']}")
        print(f"  淘汰次数: {cache.stats['evictions']}")

        # 之前缓存的数据被淘汰，再次访问会 miss
        _ = loader.load_qpos(episode_ids[0])
        print(f"  重新访问 episode_0: miss→重新加载")
        print(f"  总 miss 数: {cache.stats['misses']}")


# ############################################################
#
#  练习 4: 数据迁移脚本
#
#  场景: 从 v1 Schema 迁移到 v2 Schema。
#        v2 新增了质量字段、文件信息字段，
#        需要从原始文件中补充提取。
#
# ############################################################

@dataclass
class EpisodeV1:
    """v1 Schema 的轨迹数据"""
    id: int
    dataset_id: int
    file_path: str
    nq: int
    num_steps: int
    timestep: float


@dataclass
class EpisodeV2:
    """v2 Schema 的轨迹数据（新增字段）"""
    id: int
    dataset_id: int
    file_path: str
    nq: int
    nv: int              # 新增
    nu: int              # 新增
    num_steps: int
    timestep: float
    duration: float      # 新增
    has_nan: bool        # 新增
    has_jumps: bool      # 新增
    quality_score: float # 新增
    file_size: int       # 新增
    checksum: str        # 新增


class DataMigrator:
    """
    数据迁移器: v1 → v2

    迁移策略:
      1. 读取 v1 数据
      2. 从原始文件中提取缺失字段
      3. 计算派生字段
      4. 写入 v2 格式
      5. 验证迁移结果

    Java 等价:
        // Flyway 迁移脚本
        @Component
        public class V1ToV2Migration implements JavaMigration {
            @Override
            public void migrate(Context context) throws Exception { ... }
        }
    """

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._migration_log: List[Dict[str, Any]] = []

    def migrate_episode(self, v1: EpisodeV1) -> Tuple[Optional[EpisodeV2], Optional[str]]:
        """
        迁移单条轨迹记录。

        返回: (v2 数据, 错误信息)
        """
        try:
            # 从文件中提取新字段
            file_path = os.path.join(self._data_dir, v1.file_path)
            extracted = self._extract_new_fields(file_path)

            if extracted is None:
                return None, f"无法从文件提取数据: {file_path}"

            v2 = EpisodeV2(
                id=v1.id,
                dataset_id=v1.dataset_id,
                file_path=v1.file_path,
                nq=v1.nq,
                nv=extracted.get("nv", v1.nq),
                nu=extracted.get("nu", v1.nq),
                num_steps=v1.num_steps,
                timestep=v1.timestep,
                duration=v1.num_steps * v1.timestep,
                has_nan=extracted.get("has_nan", False),
                has_jumps=extracted.get("has_jumps", False),
                quality_score=extracted.get("quality_score", 0.0),
                file_size=extracted.get("file_size", 0),
                checksum=extracted.get("checksum", ""),
            )

            self._log_migration(v1.id, "success", "")
            return v2, None

        except Exception as e:
            self._log_migration(v1.id, "failed", str(e))
            return None, str(e)

    def migrate_batch(self, v1_records: List[EpisodeV1]) -> Dict[str, Any]:
        """批量迁移"""
        results = {"total": len(v1_records), "success": 0, "failed": 0, "records": []}

        for v1 in v1_records:
            v2, error = self.migrate_episode(v1)
            if v2 is not None:
                results["success"] += 1
                results["records"].append(v2)
            else:
                results["failed"] += 1

        return results

    def verify_migration(self, v1: EpisodeV1, v2: EpisodeV2) -> Dict[str, Any]:
        """验证迁移结果的正确性"""
        issues = []

        # 检查原始字段是否保留
        if v2.id != v1.id:
            issues.append("id 不匹配")
        if v2.dataset_id != v1.dataset_id:
            issues.append("dataset_id 不匹配")
        if v2.file_path != v1.file_path:
            issues.append("file_path 不匹配")
        if v2.nq != v1.nq:
            issues.append("nq 不匹配")
        if v2.num_steps != v1.num_steps:
            issues.append("num_steps 不匹配")

        # 检查新字段的合理性
        if v2.duration <= 0:
            issues.append("duration 应该 > 0")
        if not (0 <= v2.quality_score <= 1):
            issues.append("quality_score 应该在 [0, 1] 范围内")
        if v2.file_size <= 0:
            issues.append("file_size 应该 > 0")
        if not v2.checksum:
            issues.append("checksum 为空")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
        }

    def _extract_new_fields(self, file_path: str) -> Optional[Dict[str, Any]]:
        """从原始文件中提取 v2 需要的新字段"""
        if not os.path.exists(file_path):
            return None

        result = {
            "file_size": os.path.getsize(file_path),
        }

        # 计算校验和
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        result["checksum"] = h.hexdigest()

        # 从数据中提取质量信息
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                if "qpos" in data:
                    qpos = np.asarray(data["qpos"])
                    result["has_nan"] = bool(np.isnan(qpos).any())
                    if qpos.shape[0] > 1:
                        max_jump = float(np.nanmax(np.abs(np.diff(qpos, axis=0))))
                        result["has_jumps"] = max_jump > 2.0
                    else:
                        result["has_jumps"] = False

                    score = 1.0
                    if result["has_nan"]:
                        score -= 0.3
                    if result["has_jumps"]:
                        score -= 0.2
                    result["quality_score"] = max(0.0, score)

                if "qvel" in data:
                    qvel = np.asarray(data["qvel"])
                    result["nv"] = int(qvel.shape[1]) if qvel.ndim > 1 else 1

                if "action" in data:
                    action = np.asarray(data["action"])
                    result["nu"] = int(action.shape[1]) if action.ndim > 1 else 1

        except Exception:
            result["quality_score"] = 0.0

        return result

    def _log_migration(self, episode_id: int, status: str, error: str):
        self._migration_log.append({
            "episode_id": episode_id,
            "status": status,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })

    @property
    def migration_log(self) -> List[Dict[str, Any]]:
        return self._migration_log


def exercise_4():
    """练习 4: 数据迁移脚本"""
    print(f"\n{'练习 4: 数据迁移脚本 (v1 → v2)':=^60}")
    print("""
场景: 平台从 v1 升级到 v2，需要为已有数据补充新字段。
迁移过程要安全、可验证、可回滚。
""")

    with tempfile.TemporaryDirectory(prefix="ch07_migrate_") as tmpdir:
        # 生成 v1 数据和对应的原始文件
        v1_records = []
        for i in range(8):
            # 创建原始文件
            filename = f"episode_{i:04d}.pkl"
            file_path = os.path.join(tmpdir, filename)
            nq = 7
            num_steps = np.random.randint(100, 400)

            qpos = np.random.randn(num_steps, nq) * 0.5
            qvel = np.random.randn(num_steps, nq) * 0.1
            action = np.random.randn(num_steps, nq) * 0.3

            if i == 3:
                qpos[50, 2] = np.nan

            data = {"qpos": qpos, "qvel": qvel, "action": action, "timestep": 0.02}
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            # 创建 v1 记录
            v1_records.append(EpisodeV1(
                id=i + 1,
                dataset_id=1,
                file_path=filename,
                nq=nq,
                num_steps=num_steps,
                timestep=0.02,
            ))

        print(f"v1 记录: {len(v1_records)} 条")
        print(f"v1 Schema 字段: id, dataset_id, file_path, nq, num_steps, timestep")
        print(f"v2 新增字段: nv, nu, duration, has_nan, has_jumps, quality_score, file_size, checksum")

        # 执行迁移
        migrator = DataMigrator(tmpdir)
        print(f"\n--- 开始迁移 ---\n")

        result = migrator.migrate_batch(v1_records)
        print(f"迁移完成: 成功 {result['success']}, 失败 {result['failed']}\n")

        # 展示迁移结果
        print(f"{'ID':<4} {'文件':<22} {'nq':<4} {'nv':<4} {'nu':<4} "
              f"{'质量':<8} {'NaN':<6} {'大小':<10} {'校验和'}")
        print("-" * 85)
        for v2 in result["records"]:
            print(f"{v2.id:<4} {v2.file_path:<22} {v2.nq:<4} {v2.nv:<4} {v2.nu:<4} "
                  f"{v2.quality_score:<8.2f} {str(v2.has_nan):<6} "
                  f"{v2.file_size:<10,} {v2.checksum[:12]}...")

        # 验证迁移结果
        print(f"\n--- 验证迁移结果 ---\n")
        for v1, v2 in zip(v1_records, result["records"]):
            verify = migrator.verify_migration(v1, v2)
            icon = "✅" if verify["passed"] else "❌"
            issues_str = f" ({', '.join(verify['issues'])})" if verify["issues"] else ""
            print(f"  {icon} episode {v1.id}{issues_str}")

        # 迁移日志
        print(f"\n迁移日志:")
        for entry in migrator.migration_log[:5]:
            print(f"  episode {entry['episode_id']}: {entry['status']}")

        # 展示对应的 SQL 迁移脚本
        print(f"""
对应的 SQL 迁移脚本:
  -- v1 → v2 迁移
  ALTER TABLE episodes ADD COLUMN nv INTEGER;
  ALTER TABLE episodes ADD COLUMN nu INTEGER;
  ALTER TABLE episodes ADD COLUMN duration FLOAT;
  ALTER TABLE episodes ADD COLUMN has_nan BOOLEAN DEFAULT FALSE;
  ALTER TABLE episodes ADD COLUMN has_jumps BOOLEAN DEFAULT FALSE;
  ALTER TABLE episodes ADD COLUMN quality_score FLOAT;
  ALTER TABLE episodes ADD COLUMN file_size BIGINT;
  ALTER TABLE episodes ADD COLUMN checksum VARCHAR(64);

  -- 回填数据（由 Python 脚本生成 UPDATE 语句）
  UPDATE episodes SET
    nv = 7, nu = 7,
    duration = num_steps * timestep,
    has_nan = FALSE,
    quality_score = 1.0,
    file_size = 12345,
    checksum = 'abc123...'
  WHERE id = 1;
  -- ... 其他记录
""")


# ############################################################
# 主函数
# ############################################################

def main():
    print(DIVIDER)
    print("第 7 章 · 05 - 练习 (Exercises)")
    print(DIVIDER)
    print("""
本节包含 4 个练习，每个练习都是一个独立的小项目:
  练习 1: 多模态数据 Schema 设计
  练习 2: 数据版本管理系统
  练习 3: 缓存层实现
  练习 4: 数据迁移脚本 (v1 → v2)
""")

    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

    print(f"\n{DIVIDER}")
    print("✅ 全部练习完成！")
    print(f"""
回顾与扩展思考:

  练习 1 → 思考: 如果图像数据量太大（TB 级），存储策略怎么设计？
           提示: 图像和 qpos 分离存储，图像用对象存储 + CDN

  练习 2 → 思考: 如何实现数据集的「分支」功能（类似 Git branch）？
           提示: 参考 DVC (Data Version Control) 的设计

  练习 3 → 思考: 分布式环境下缓存如何保持一致性？
           提示: Redis + 缓存失效策略 (TTL, 主动失效)

  练习 4 → 思考: 如果迁移过程中数据库挂了，如何保证数据不丢失？
           提示: 事务 + WAL (Write-Ahead Log) + 断点续传

第 7 章全部完成！你现在具备了设计企业级机器人数据平台的能力。
→ 下一章: 第 8 章 进阶主题（运动学、控制、Sim2Real）
""")
    print(DIVIDER)


if __name__ == "__main__":
    main()
