"""
第 5 章 · 04 - 业界标准数据集格式

目标: 了解机器人学习领域的主流数据集标准，学会创建符合标准的数据，
     并能验证数据结构是否合规。

核心知识点:
  1. ALOHA / ACT 数据集格式 (HDF5)
  2. LeRobot 数据集格式 (Parquet + 视频)
  3. RLDS 格式概念 (TFRecord)
  4. Open X-Embodiment 格式
  5. 各格式的 Schema 验证器

运行: python 04_dataset_standards.py
依赖: pip install numpy h5py
可选: pip install pyarrow pandas
"""

import numpy as np
import os
import time
import tempfile
import shutil
import json
from dataclasses import dataclass, field
from typing import Optional

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("警告: h5py 未安装。pip install h5py")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


def file_size_str(path):
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ============================================================
# 第 1 节：ALOHA / ACT 数据集格式
# ============================================================

def section_1_aloha_format(work_dir):
    print(DIVIDER)
    print("第 1 节：ALOHA / ACT 数据集格式")
    print(DIVIDER)

    print("""
    ALOHA (A Low-cost Open-source Hardware System for Bimanual Teleoperation)
    是 Google DeepMind 开源的双臂遥操作系统，ACT 是其配套的模仿学习算法。

    ALOHA 的数据格式已成为机器人模仿学习领域的事实标准之一。

    ┌─────────── ALOHA/ACT HDF5 数据结构 ───────────┐
    │                                                 │
    │  episode_XXXX.hdf5                              │
    │  ├── /observations/                             │
    │  │   ├── qpos          (T, 14)  float64         │
    │  │   ├── qvel          (T, 14)  float64         │
    │  │   └── /images/                               │
    │  │       ├── cam_high  (T, 480, 640, 3) uint8   │
    │  │       ├── cam_left_wrist  (T, 480, 640, 3)   │
    │  │       └── cam_right_wrist (T, 480, 640, 3)   │
    │  ├── /action           (T, 14)  float64         │
    │  └── [attributes]                               │
    │      ├── sim: bool                              │
    │      ├── compress: bool                         │
    │      └── num_timesteps: int                     │
    │                                                 │
    │  14 = 7 (左臂 6关节+夹爪) + 7 (右臂)           │
    │  T  = episode 的时间步数 (通常 300-500)         │
    │                                                 │
    └─────────────────────────────────────────────────┘

    目录结构:
      dataset/
      ├── episode_0.hdf5
      ├── episode_1.hdf5
      ├── ...
      └── episode_49.hdf5
    """)

    # --- 创建一个符合 ALOHA 标准的 episode ---
    T = 400
    nq = 14  # ALOHA 双臂

    t = np.linspace(0, 4 * np.pi, T)
    qpos = np.column_stack([np.sin(t * (i+1) * 0.5) * 0.3 for i in range(nq)]).astype(np.float64)
    qvel = np.column_stack([np.cos(t * (i+1) * 0.5) * 0.2 for i in range(nq)]).astype(np.float64)
    action = np.column_stack([np.sin(t * (i+1) * 0.5 + 0.1) * 0.3 for i in range(nq)]).astype(np.float64)

    # 用较小的图像尺寸（节省内存），但结构完全符合标准
    img_h, img_w = 60, 80
    cam_high = np.random.randint(0, 256, (T, img_h, img_w, 3), dtype=np.uint8)
    cam_left = np.random.randint(0, 256, (T, img_h, img_w, 3), dtype=np.uint8)
    cam_right = np.random.randint(0, 256, (T, img_h, img_w, 3), dtype=np.uint8)

    aloha_dir = os.path.join(work_dir, "aloha_dataset")
    os.makedirs(aloha_dir, exist_ok=True)

    fpath = os.path.join(aloha_dir, "episode_0.hdf5")

    with h5py.File(fpath, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=qvel)

        imgs = obs.create_group("images")
        imgs.create_dataset("cam_high", data=cam_high, chunks=(1, img_h, img_w, 3), compression="gzip", compression_opts=4)
        imgs.create_dataset("cam_left_wrist", data=cam_left, chunks=(1, img_h, img_w, 3), compression="gzip", compression_opts=4)
        imgs.create_dataset("cam_right_wrist", data=cam_right, chunks=(1, img_h, img_w, 3), compression="gzip", compression_opts=4)

        f.create_dataset("action", data=action)

        f.attrs["sim"] = True
        f.attrs["compress"] = True
        f.attrs["num_timesteps"] = T

    print(f"  已创建 ALOHA 格式 episode: {fpath}")
    print(f"  文件大小: {file_size_str(fpath)}")

    # 验证结构
    with h5py.File(fpath, "r") as f:
        print(f"\n  文件结构:")
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    /{name}: {obj.shape} {obj.dtype}")
        f.visititems(print_item)
        print(f"  元数据:")
        for k, v in f.attrs.items():
            print(f"    {k}: {v}")

    return aloha_dir


# ============================================================
# 第 2 节：ALOHA 格式 Schema 验证器
# ============================================================

@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def add_error(self, msg: str):
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def report(self, prefix="  "):
        status = "✓ 通过" if self.valid else "✗ 失败"
        print(f"{prefix}验证结果: {status}")
        for e in self.errors:
            print(f"{prefix}  ✗ {e}")
        for w in self.warnings:
            print(f"{prefix}  ⚠ {w}")


def validate_aloha_episode(filepath: str) -> ValidationResult:
    """
    验证单个 HDF5 文件是否符合 ALOHA/ACT 数据格式标准。

    检查项:
      1. 必需的 Group: /observations, /observations/images
      2. 必需的 Dataset: /observations/qpos, /observations/qvel, /action
      3. 维度一致性: T 维度必须统一
      4. 数据类型: qpos/qvel/action 为 float64, 图像为 uint8
      5. 必需的属性: num_timesteps
    """
    result = ValidationResult()

    if not os.path.exists(filepath):
        result.add_error(f"文件不存在: {filepath}")
        return result

    try:
        with h5py.File(filepath, "r") as f:
            # 检查顶级结构
            if "observations" not in f:
                result.add_error("缺少 /observations Group")
            if "action" not in f:
                result.add_error("缺少 /action Dataset")

            if not result.valid:
                return result

            obs = f["observations"]

            # 检查必需 dataset
            for name in ["qpos", "qvel"]:
                if name not in obs:
                    result.add_error(f"缺少 /observations/{name}")

            if not result.valid:
                return result

            # 获取时间步数
            T = f["action"].shape[0]

            # 检查维度一致性
            for ds_path in ["observations/qpos", "observations/qvel", "action"]:
                ds = f[ds_path]
                if ds.shape[0] != T:
                    result.add_error(f"{ds_path} 时间步数不一致: {ds.shape[0]} vs {T}")
                if ds.ndim != 2:
                    result.add_error(f"{ds_path} 应为 2D 数组, 实际 {ds.ndim}D")
                if ds.dtype != np.float64:
                    result.add_warning(f"{ds_path} dtype={ds.dtype}, 推荐 float64")

            # 检查图像（可选，但推荐）
            if "images" in obs:
                for cam_name in obs["images"]:
                    cam = obs["images"][cam_name]
                    if cam.shape[0] != T:
                        result.add_error(f"images/{cam_name} 时间步数不一致")
                    if cam.ndim != 4:
                        result.add_error(f"images/{cam_name} 应为 4D (T,H,W,C)")
                    if cam.dtype != np.uint8:
                        result.add_warning(f"images/{cam_name} dtype={cam.dtype}, 推荐 uint8")
            else:
                result.add_warning("缺少 /observations/images (图像数据)")

            # 检查属性
            if "num_timesteps" not in f.attrs:
                result.add_warning("缺少 num_timesteps 属性")
            elif int(f.attrs["num_timesteps"]) != T:
                result.add_error(f"num_timesteps 属性 ({f.attrs['num_timesteps']}) 与实际不符 ({T})")

            # 检查 ALOHA 特有的维度
            nq = f["observations/qpos"].shape[1]
            nu = f["action"].shape[1]
            if nq != nu:
                result.add_warning(f"qpos 维度 ({nq}) != action 维度 ({nu})")
            if nq == 14:
                pass  # 标准 ALOHA 双臂
            elif nq == 7:
                result.add_warning("nq=7: 单臂配置（标准 ALOHA 为 14）")
            else:
                result.add_warning(f"nq={nq}: 非标准 ALOHA 维度 (标准为 14)")

    except Exception as e:
        result.add_error(f"文件读取失败: {e}")

    return result


def section_2_aloha_validator(work_dir, aloha_dir):
    print(DIVIDER)
    print("第 2 节：ALOHA Schema 验证器")
    print(DIVIDER)

    # 验证正确的文件
    fpath = os.path.join(aloha_dir, "episode_0.hdf5")
    print(f"\n  验证标准文件: {os.path.basename(fpath)}")
    result = validate_aloha_episode(fpath)
    result.report()

    # 创建一个有问题的文件
    bad_path = os.path.join(work_dir, "bad_episode.hdf5")
    with h5py.File(bad_path, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.zeros((100, 7)))
        # 故意缺少 qvel
        f.create_dataset("action", data=np.zeros((99, 7)))  # T 不一致

    print(f"\n  验证有问题的文件: {os.path.basename(bad_path)}")
    result = validate_aloha_episode(bad_path)
    result.report()

    # 创建一个完全错误的文件
    empty_path = os.path.join(work_dir, "empty_episode.hdf5")
    with h5py.File(empty_path, "w") as f:
        f.create_dataset("random", data=np.zeros(10))

    print(f"\n  验证空文件: {os.path.basename(empty_path)}")
    result = validate_aloha_episode(empty_path)
    result.report()


# ============================================================
# 第 3 节：LeRobot 数据集格式
# ============================================================

def section_3_lerobot_format(work_dir):
    print(DIVIDER)
    print("第 3 节：LeRobot 数据集格式")
    print(DIVIDER)

    print("""
    LeRobot 是 Hugging Face 推出的机器人学习框架。
    其数据集格式设计用于大规模分发和高效训练。

    ┌──────────── LeRobot v2 数据集结构 ────────────┐
    │                                                │
    │  dataset/                                      │
    │  ├── meta/                                     │
    │  │   ├── info.json          数据集元信息        │
    │  │   ├── episodes.jsonl     每集的元数据        │
    │  │   ├── stats.json         数据统计信息        │
    │  │   └── tasks.jsonl        任务描述            │
    │  ├── data/                                     │
    │  │   ├── chunk-000/                            │
    │  │   │   ├── episode_000000.parquet            │
    │  │   │   ├── episode_000001.parquet            │
    │  │   │   └── ...                               │
    │  │   └── chunk-001/                            │
    │  │       └── ...                               │
    │  └── videos/  (可选)                           │
    │      ├── chunk-000/                            │
    │      │   ├── observation.images.cam_high/      │
    │      │   │   ├── episode_000000.mp4            │
    │      │   │   └── ...                           │
    │      │   └── observation.images.cam_wrist/     │
    │      │       └── ...                           │
    │      └── chunk-001/                            │
    │          └── ...                               │
    │                                                │
    │  每个 parquet 文件中的列:                       │
    │    - observation.state      [nq] float         │
    │    - observation.images.cam_high  [视频帧引用]  │
    │    - action                 [nu] float         │
    │    - episode_index          int                │
    │    - frame_index            int                │
    │    - timestamp              float              │
    │    - task_index             int                │
    │    - next.done              bool               │
    │                                                │
    └────────────────────────────────────────────────┘

    关键设计决策:
      - 使用 Parquet 存储表格化数据（高效列式存储）
      - 视频独立存储为 MP4（比逐帧 uint8 高效 10-100x）
      - 按 chunk 分片（支持大规模数据集的分布式存储）
      - 元数据集中管理（info.json, episodes.jsonl）
    """)

    # --- 创建一个模拟的 LeRobot 数据集目录结构 ---
    lerobot_dir = os.path.join(work_dir, "lerobot_dataset")

    # 创建目录结构
    for d in ["meta", "data/chunk-000", "videos/chunk-000/observation.images.cam_high"]:
        os.makedirs(os.path.join(lerobot_dir, d), exist_ok=True)

    # 1. meta/info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": "koch",
        "fps": 30,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6], "names": ["joint_positions"]},
            "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3],
                                             "video_info": {"video.fps": 30, "video.codec": "av1",
                                                            "video.pix_fmt": "yuv420p",
                                                            "has_audio": False}},
            "action": {"dtype": "float32", "shape": [6], "names": ["joint_positions"]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        },
        "total_episodes": 3,
        "total_frames": 600,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }

    info_path = os.path.join(lerobot_dir, "meta", "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  meta/info.json 已创建")

    # 2. meta/episodes.jsonl
    episodes_path = os.path.join(lerobot_dir, "meta", "episodes.jsonl")
    with open(episodes_path, "w") as f:
        for ep in range(3):
            entry = {"episode_index": ep, "tasks": ["pick up the cube"], "length": 200}
            f.write(json.dumps(entry) + "\n")
    print(f"  meta/episodes.jsonl 已创建")

    # 3. meta/tasks.jsonl
    tasks_path = os.path.join(lerobot_dir, "meta", "tasks.jsonl")
    with open(tasks_path, "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "pick up the cube"}) + "\n")
    print(f"  meta/tasks.jsonl 已创建")

    # 4. Parquet 数据文件
    if HAS_PYARROW:
        for ep_idx in range(3):
            T = 200
            nq = 6
            t = np.linspace(0, 2 * np.pi, T)

            states = np.column_stack([np.sin(t * (i+1)) * 0.3 for i in range(nq)]).astype(np.float32)
            actions = np.column_stack([np.sin(t * (i+1) + 0.05) * 0.3 for i in range(nq)]).astype(np.float32)

            state_list = [states[i].tolist() for i in range(T)]
            action_list = [actions[i].tolist() for i in range(T)]

            table = pa.table({
                "observation.state": state_list,
                "action": action_list,
                "episode_index": pa.array([ep_idx] * T, type=pa.int64()),
                "frame_index": pa.array(list(range(T)), type=pa.int64()),
                "timestamp": pa.array(np.arange(T, dtype=np.float32) / 30.0),
                "task_index": pa.array([0] * T, type=pa.int64()),
                "next.done": pa.array([False] * (T - 1) + [True]),
            })

            parquet_path = os.path.join(lerobot_dir, "data", "chunk-000",
                                        f"episode_{ep_idx:06d}.parquet")
            pq.write_table(table, parquet_path)

        print(f"  data/chunk-000/ 下已创建 3 个 parquet 文件")

        # 读取验证
        sample = pq.read_table(os.path.join(lerobot_dir, "data", "chunk-000", "episode_000000.parquet"))
        print(f"\n  Parquet 示例 (episode_000000):")
        print(f"    行数: {sample.num_rows}")
        print(f"    列: {sample.column_names}")
        print(f"    observation.state[0]: {sample['observation.state'][0].as_py()[:3]}...")
    else:
        print("\n  ⚠️  pyarrow 未安装，跳过 Parquet 文件创建")
        print("  pip install pyarrow 以启用完整功能")

    # 5. 视频占位（实际场景中是 .mp4 文件）
    for ep_idx in range(3):
        placeholder = os.path.join(lerobot_dir, "videos", "chunk-000",
                                   "observation.images.cam_high",
                                   f"episode_{ep_idx:06d}.mp4")
        with open(placeholder, "w") as f:
            f.write("(video placeholder)")

    print(f"\n  完整 LeRobot 数据集目录结构:")
    for root, dirs, files in os.walk(lerobot_dir):
        level = root.replace(lerobot_dir, "").count(os.sep)
        indent = "  " * (level + 2)
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 3)
        for file in files:
            print(f"{sub_indent}{file}")

    return lerobot_dir


# ============================================================
# 第 4 节：LeRobot 格式验证器
# ============================================================

def validate_lerobot_dataset(dataset_dir: str) -> ValidationResult:
    """验证 LeRobot 数据集目录结构"""
    result = ValidationResult()

    # 检查 meta/info.json
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    if not os.path.exists(info_path):
        result.add_error("缺少 meta/info.json")
        return result

    with open(info_path) as f:
        info = json.load(f)

    required_info_keys = ["codebase_version", "fps", "features", "total_episodes", "total_frames"]
    for key in required_info_keys:
        if key not in info:
            result.add_error(f"info.json 缺少字段: {key}")

    # 检查 features 中的必需字段
    features = info.get("features", {})
    required_features = ["action"]
    for feat in required_features:
        if feat not in features:
            result.add_error(f"info.json features 缺少: {feat}")

    # 检查 episodes.jsonl
    episodes_path = os.path.join(dataset_dir, "meta", "episodes.jsonl")
    if not os.path.exists(episodes_path):
        result.add_warning("缺少 meta/episodes.jsonl")

    # 检查 data 目录
    data_dir = os.path.join(dataset_dir, "data")
    if not os.path.exists(data_dir):
        result.add_error("缺少 data/ 目录")
    else:
        parquet_files = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))
        if not parquet_files:
            result.add_error("data/ 目录中没有 .parquet 文件")
        else:
            expected = info.get("total_episodes", 0)
            if len(parquet_files) != expected:
                result.add_warning(
                    f"parquet 文件数 ({len(parquet_files)}) != total_episodes ({expected})")

    return result


def section_4_lerobot_validator(work_dir, lerobot_dir):
    print(DIVIDER)
    print("第 4 节：LeRobot Schema 验证器")
    print(DIVIDER)

    print(f"\n  验证 LeRobot 数据集: {lerobot_dir}")
    result = validate_lerobot_dataset(lerobot_dir)
    result.report()

    # 创建一个不完整的数据集
    bad_dir = os.path.join(work_dir, "bad_lerobot")
    os.makedirs(os.path.join(bad_dir, "meta"), exist_ok=True)
    with open(os.path.join(bad_dir, "meta", "info.json"), "w") as f:
        json.dump({"fps": 30}, f)

    print(f"\n  验证不完整的数据集:")
    result = validate_lerobot_dataset(bad_dir)
    result.report()


# ============================================================
# 第 5 节：RLDS 格式概念
# ============================================================

def section_5_rlds_format():
    print(DIVIDER)
    print("第 5 节：RLDS (Reinforcement Learning Datasets) 格式")
    print(DIVIDER)

    print("""
    RLDS 是 Google 设计的强化学习数据集标准。
    底层使用 TFRecord (TensorFlow 的二进制记录格式)。

    ┌──────────── RLDS 核心概念 ────────────┐
    │                                        │
    │  Dataset = 多个 Episode                │
    │                                        │
    │  Episode = {                            │
    │    steps: [Step, Step, ...],            │
    │    metadata: {...}                      │
    │  }                                     │
    │                                        │
    │  Step = {                              │
    │    observation: {                       │
    │      image: Tensor (H, W, C),          │
    │      state: Tensor (D,),               │
    │    },                                  │
    │    action: Tensor (A,),                │
    │    reward: float,                      │
    │    discount: float,                    │
    │    is_first: bool,                     │
    │    is_last: bool,                      │
    │    is_terminal: bool,                  │
    │    language_instruction: str,           │
    │  }                                     │
    │                                        │
    └────────────────────────────────────────┘

    RLDS 的特点:
      1. 基于 TFRecord —— 高效的顺序读取
      2. Step-centric —— 每个时间步是独立的记录
      3. 标准化字段 —— is_first/is_last/is_terminal 管理 episode 边界
      4. 与 TensorFlow Datasets (TFDS) 深度集成

    为什么了解 RLDS 很重要?
      - Open X-Embodiment 数据集使用 RLDS 格式
      - 很多 Google 的机器人数据集使用 RLDS
      - 是跨机器人数据共享的重要标准

    注意: RLDS 依赖 TensorFlow 生态，这里我们用 Python dict 模拟其结构，
    不实际依赖 tensorflow。
    """)

    # 用 Python dict 模拟 RLDS 的 Episode/Step 结构
    print("  用 Python dict 模拟 RLDS 结构:\n")

    T = 5  # 用少量步骤演示结构
    nq = 7
    t = np.linspace(0, np.pi, T)

    episode = {
        "steps": [],
        "metadata": {
            "episode_id": 0,
            "robot_type": "franka_panda",
            "language_instruction": "pick up the red cube",
        }
    }

    for i in range(T):
        step = {
            "observation": {
                "state": np.sin(t[i] * np.arange(nq)).astype(np.float32),
                "image": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
            },
            "action": np.cos(t[i] * np.arange(nq)).astype(np.float32),
            "reward": float(i == T - 1),  # 最后一步给 reward
            "discount": 1.0,
            "is_first": i == 0,
            "is_last": i == T - 1,
            "is_terminal": i == T - 1,
            "language_instruction": "pick up the red cube",
        }
        episode["steps"].append(step)

    print(f"  Episode 结构:")
    print(f"    steps: {len(episode['steps'])} 步")
    print(f"    metadata: {episode['metadata']}")
    print(f"\n  Step[0] 结构:")
    step0 = episode["steps"][0]
    print(f"    observation.state: shape={step0['observation']['state'].shape}")
    print(f"    observation.image: shape={step0['observation']['image'].shape}")
    print(f"    action: shape={step0['action'].shape}")
    print(f"    reward: {step0['reward']}")
    print(f"    is_first: {step0['is_first']}, is_last: {step0['is_last']}")

    # --- RLDS → 标准 array 格式 的转换 ---
    print(f"\n  RLDS → 标准 array 格式转换:")

    states = np.stack([s["observation"]["state"] for s in episode["steps"]])
    actions = np.stack([s["action"] for s in episode["steps"]])
    rewards = np.array([s["reward"] for s in episode["steps"]])

    print(f"    states: {states.shape}")
    print(f"    actions: {actions.shape}")
    print(f"    rewards: {rewards.shape}")

    return episode


# ============================================================
# 第 6 节：Open X-Embodiment 格式
# ============================================================

def section_6_openx_format():
    print(DIVIDER)
    print("第 6 节：Open X-Embodiment (OXE) 格式")
    print(DIVIDER)

    print("""
    Open X-Embodiment 是 Google DeepMind 主导的大规模跨机器人数据集项目。
    目标：统一不同机器人的数据格式，让一个模型能在多种机器人上工作。

    ┌───────── Open X-Embodiment 架构 ─────────┐
    │                                           │
    │  底层格式: RLDS (TFRecord)                │
    │                                           │
    │  标准化的 observation 字段:                │
    │    image          主相机 RGB               │
    │    wrist_image    腕部相机 RGB             │
    │    state          机器人状态向量           │
    │                                           │
    │  标准化的 action 字段:                     │
    │    action         控制动作向量             │
    │                                           │
    │  额外字段:                                │
    │    language_instruction  自然语言任务描述  │
    │    language_embedding    语言嵌入向量      │
    │                                           │
    │  核心贡献:                                │
    │    - 统一了 22 个机器人平台的数据格式      │
    │    - 定义了标准化的 action 空间            │
    │    - 提供了数据转换工具链                   │
    │                                           │
    └───────────────────────────────────────────┘

    Open X-Embodiment 的 action 标准化:
      标准 action = [x, y, z, rx, ry, rz, gripper]
        (x,y,z) = 末端执行器平移 (m/s 或 delta)
        (rx,ry,rz) = 末端执行器旋转 (rad/s 或 delta)
        gripper = 夹爪开合 (0=关, 1=开)

    数据集规模:
      - 60+ 个数据集
      - 22 个机器人形态
      - 100K+ 条轨迹
      - 涵盖抓取、推动、开关门、叠衣服等任务
    """)

    # 用 Python dict 模拟 OXE 格式
    print("  模拟 Open X-Embodiment 格式的 episode:\n")

    T = 100
    episode_oxe = {
        "steps": [],
        "metadata": {
            "dataset_name": "bridge_v2",
            "robot_type": "widowx",
            "action_space": "delta_ee_pose",
        }
    }

    for i in range(T):
        t = i / T
        step = {
            "observation": {
                "image": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
                "wrist_image": np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
                "state": np.random.randn(7).astype(np.float32),
            },
            "action": np.random.randn(7).astype(np.float32),  # [x,y,z,rx,ry,rz,gripper]
            "reward": float(i == T - 1),
            "is_first": i == 0,
            "is_last": i == T - 1,
            "is_terminal": i == T - 1,
            "language_instruction": "pick up the red block and place it on the blue block",
        }
        episode_oxe["steps"].append(step)

    print(f"  dataset: {episode_oxe['metadata']['dataset_name']}")
    print(f"  robot:   {episode_oxe['metadata']['robot_type']}")
    print(f"  steps:   {len(episode_oxe['steps'])}")
    step = episode_oxe["steps"][0]
    print(f"  observation.image:       {step['observation']['image'].shape}")
    print(f"  observation.wrist_image: {step['observation']['wrist_image'].shape}")
    print(f"  observation.state:       {step['observation']['state'].shape}")
    print(f"  action:                  {step['action'].shape}")
    print(f"  language_instruction:    \"{step['language_instruction']}\"")


# ============================================================
# 第 7 节：格式转换 — ALOHA ↔ LeRobot
# ============================================================

def section_7_cross_format_conversion(work_dir, aloha_dir):
    print(DIVIDER)
    print("第 7 节：跨标准格式转换 (ALOHA → LeRobot 概念)")
    print(DIVIDER)

    print("""
    实际项目中经常需要在不同标准之间转换。
    下面演示把 ALOHA HDF5 格式转换为 LeRobot 的结构化数据。
    (图像/视频转换需要额外的编码工具，这里只处理数值数据)
    """)

    # 读取 ALOHA 数据
    aloha_path = os.path.join(aloha_dir, "episode_0.hdf5")
    with h5py.File(aloha_path, "r") as f:
        qpos = f["observations/qpos"][:]
        qvel = f["observations/qvel"][:]
        action = f["action"][:]
        T = qpos.shape[0]
        attrs = dict(f.attrs)

    print(f"  ALOHA 源数据: T={T}, nq={qpos.shape[1]}")

    # 转换为 LeRobot 表格格式
    if HAS_PYARROW:
        state_list = [qpos[i].tolist() for i in range(T)]
        action_list = [action[i].tolist() for i in range(T)]

        table = pa.table({
            "observation.state": state_list,
            "action": action_list,
            "episode_index": pa.array([0] * T, type=pa.int64()),
            "frame_index": pa.array(list(range(T)), type=pa.int64()),
            "timestamp": pa.array(np.arange(T, dtype=np.float32) / 50.0),
            "task_index": pa.array([0] * T, type=pa.int64()),
            "next.done": pa.array([False] * (T - 1) + [True]),
        })

        parquet_path = os.path.join(work_dir, "aloha_as_lerobot.parquet")
        pq.write_table(table, parquet_path)

        print(f"  LeRobot Parquet 输出: {file_size_str(parquet_path)}")
        print(f"  列: {table.column_names}")
        print(f"  行数: {table.num_rows}")

        # 验证
        read_back = pq.read_table(parquet_path)
        state_back = np.array(read_back["observation.state"].to_pylist())
        assert np.allclose(qpos, state_back, atol=1e-6)
        print("  ✓ 数值数据一致性验证通过")
    else:
        print("  ⚠️  pyarrow 未安装，跳过 Parquet 转换")
        print("  pip install pyarrow 以启用")


# ============================================================
# 第 8 节：格式选择决策树
# ============================================================

def section_8_decision_guide():
    print(DIVIDER)
    print("第 8 节：格式选择决策树")
    print(DIVIDER)

    print("""
    ┌───────────────── 该选哪种数据格式？ ─────────────────┐
    │                                                       │
    │  你要做什么？                                          │
    │  │                                                    │
    │  ├─ 模仿学习 / 离线 RL 训练                           │
    │  │  ├─ 使用 ACT / Diffusion Policy                    │
    │  │  │  └─→ ALOHA HDF5 格式                            │
    │  │  ├─ 使用 LeRobot 框架                              │
    │  │  │  └─→ LeRobot Parquet + 视频                     │
    │  │  ├─ 使用 RT-X / Octo 等基础模型                    │
    │  │  │  └─→ RLDS / Open X-Embodiment                   │
    │  │  └─ 自定义训练管线                                  │
    │  │     └─→ HDF5 (最灵活)                              │
    │  │                                                    │
    │  ├─ 数据采集 / 快速迭代                                │
    │  │  └─→ PKL 或 NPZ（之后转为标准格式）                 │
    │  │                                                    │
    │  ├─ 数据共享 / 分发                                    │
    │  │  ├─ Hugging Face Hub                               │
    │  │  │  └─→ LeRobot 格式                               │
    │  │  ├─ 学术发表                                        │
    │  │  │  └─→ RLDS 或 HDF5                               │
    │  │  └─ 团队内部                                        │
    │  │     └─→ HDF5                                       │
    │  │                                                    │
    │  └─ 数据分析 / 可视化                                  │
    │     └─→ Parquet (配合 pandas/DuckDB)                  │
    │                                                       │
    └───────────────────────────────────────────────────────┘

    作为数据平台开发者的建议:
      1. 内部存储标准化为 HDF5（最通用、功能最全）
      2. 提供到 LeRobot/RLDS 的自动转换管线
      3. 元数据索引用 Parquet 或数据库
      4. 数据采集端可以用 PKL（然后自动转换）
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("  第 5 章 · 04 — 业界标准数据集格式")
    print(DIVIDER)

    if not HAS_H5PY:
        print("\n  ⚠️  h5py 未安装，HDF5 相关功能不可用")
        print("  请运行: pip install h5py\n")
        return

    work_dir = tempfile.mkdtemp(prefix="ch05_standards_")
    print(f"\n  工作目录: {work_dir}\n")

    try:
        aloha_dir = section_1_aloha_format(work_dir)
        section_2_aloha_validator(work_dir, aloha_dir)
        lerobot_dir = section_3_lerobot_format(work_dir)
        section_4_lerobot_validator(work_dir, lerobot_dir)
        rlds_episode = section_5_rlds_format()
        section_6_openx_format()
        section_7_cross_format_conversion(work_dir, aloha_dir)
        section_8_decision_guide()

        print(DIVIDER)
        print("  所有标准格式示例运行完毕！")
        print(DIVIDER)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"\n  已清理工作目录: {work_dir}")


if __name__ == "__main__":
    main()
