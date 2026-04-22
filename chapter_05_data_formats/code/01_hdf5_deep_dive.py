"""
第 5 章 · 01 - HDF5 深度实战

目标: 全面掌握 HDF5 格式在机器人数据领域的应用，
     包括创建、压缩、分块、切片读取、追加写入、性能基准测试。

核心知识点:
  1. HDF5 是什么：文件内的层级文件系统
  2. Java 类比：持久化到磁盘的 Map<String, Object>
  3. 标准机器人轨迹数据结构
  4. 压缩选项与文件体积影响
  5. 分块（chunking）策略
  6. 切片读取与追加写入
  7. 性能基准测试

运行: python 01_hdf5_deep_dive.py
依赖: pip install h5py numpy
"""

import h5py
import numpy as np
import os
import time
import tempfile
import shutil

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# ============================================================
# 辅助函数
# ============================================================

def file_size_str(path):
    """返回人类可读的文件大小"""
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def generate_robot_episode(num_timesteps=400, nq=7, nv=7, nu=7,
                           img_h=240, img_w=320, include_images=True):
    """
    生成一个模拟的机器人轨迹 episode。

    参数:
        num_timesteps: 时间步数
        nq: 关节位置维度 (qpos)
        nv: 关节速度维度 (qvel)
        nu: 动作维度 (action)
        img_h, img_w: 图像高宽
        include_images: 是否包含图像数据

    返回: dict，包含所有轨迹数据
    """
    t = np.linspace(0, 2 * np.pi, num_timesteps)

    data = {
        "observations": {
            "qpos": np.column_stack([np.sin(t * (i + 1)) * 0.5 for i in range(nq)]).astype(np.float64),
            "qvel": np.column_stack([np.cos(t * (i + 1)) * 0.3 for i in range(nv)]).astype(np.float64),
        },
        "action": np.column_stack([np.sin(t * (i + 1) + 0.1) * 0.5 for i in range(nu)]).astype(np.float64),
        "metadata": {
            "sim": True,
            "robot_type": "franka_panda",
            "num_timesteps": num_timesteps,
            "hz": 50,
            "task": "pick_and_place",
        }
    }

    if include_images:
        # 生成伪图像数据（渐变 + 噪声，模拟真实相机输出）
        data["observations"]["images"] = {
            "cam_high": np.random.randint(0, 256, (num_timesteps, img_h, img_w, 3), dtype=np.uint8),
        }

    return data


# ============================================================
# 第 1 节：HDF5 是什么？
# ============================================================

def section_1_what_is_hdf5():
    print(DIVIDER)
    print("第 1 节：HDF5 是什么？")
    print(DIVIDER)

    print("""
    HDF5 = Hierarchical Data Format version 5

    核心思想：一个文件就是一个「文件系统」
    ┌─────────────────────────────────────────┐
    │  HDF5 文件                               │
    │  ├── Group（目录）                       │
    │  │   ├── Dataset（文件/数组）             │
    │  │   ├── Dataset                         │
    │  │   └── Group（子目录）                  │
    │  │       └── Dataset                     │
    │  ├── Group                               │
    │  │   └── Dataset                         │
    │  └── Attributes（元数据，附着在任何节点） │
    └─────────────────────────────────────────┘

    Java 类比:
      - HDF5 File  ≈ Map<String, Object> 持久化到磁盘
      - Group      ≈ Map<String, Object>（嵌套的 Map）
      - Dataset    ≈ NDArray（多维数组）
      - Attribute  ≈ Map 上附加的 metadata

    Python 类比:
      - HDF5 File  ≈ 嵌套的 dict，但支持磁盘存储和切片读取
      - Group      ≈ dict
      - Dataset    ≈ numpy.ndarray（支持懒加载）

    为什么机器人领域偏爱 HDF5？
      1. 支持层级结构 —— 天然适配 observations/action/images
      2. 支持压缩 —— 图像数据压缩后体积锐减
      3. 支持切片读取 —— 不用加载整个文件就能读取第 100-200 步
      4. 支持元数据 —— 可以记录机器人类型、采样率等信息
      5. 跨语言 —— C/C++/Java/MATLAB/Python 都有库
    """)


# ============================================================
# 第 2 节：创建 HDF5 文件
# ============================================================

def section_2_create_hdf5(work_dir):
    print(DIVIDER)
    print("第 2 节：创建 HDF5 文件 — 从零开始")
    print(DIVIDER)

    filepath = os.path.join(work_dir, "demo_basic.hdf5")

    # --- 基础示例：创建一个包含多种数据类型的 HDF5 文件 ---
    with h5py.File(filepath, "w") as f:
        # 创建 Group（类似目录）
        obs_group = f.create_group("observations")
        img_group = obs_group.create_group("images")

        # 创建 Dataset（类似文件中的数组）
        T, nq, nv, nu = 100, 7, 7, 7
        qpos = np.random.randn(T, nq).astype(np.float64)
        qvel = np.random.randn(T, nv).astype(np.float64)
        action = np.random.randn(T, nu).astype(np.float64)

        obs_group.create_dataset("qpos", data=qpos)
        obs_group.create_dataset("qvel", data=qvel)
        f.create_dataset("action", data=action)

        # 创建图像数据集
        cam_data = np.random.randint(0, 256, (T, 120, 160, 3), dtype=np.uint8)
        img_group.create_dataset("cam_high", data=cam_data)

        # 添加 Attributes（元数据）
        f.attrs["sim"] = True
        f.attrs["robot_type"] = "franka_panda"
        f.attrs["num_timesteps"] = T
        f.attrs["hz"] = 50
        f.attrs["task"] = "pick_and_place"

    print(f"  文件已创建: {filepath}")
    print(f"  文件大小: {file_size_str(filepath)}")

    # --- 读取并验证 ---
    with h5py.File(filepath, "r") as f:
        print(f"\n  文件结构:")
        def print_structure(name, obj):
            prefix = "  " * (name.count("/") + 2)
            if isinstance(obj, h5py.Dataset):
                print(f"{prefix}📄 {name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{prefix}📁 {name}/")
        f.visititems(print_structure)

        print(f"\n  元数据 (Attributes):")
        for key, val in f.attrs.items():
            print(f"    {key}: {val}")

    return filepath


# ============================================================
# 第 3 节：标准机器人轨迹数据结构
# ============================================================

def section_3_robot_data_structure(work_dir):
    print(DIVIDER)
    print("第 3 节：标准机器人轨迹数据结构")
    print(DIVIDER)

    print("""
    业界常见的 episode 文件结构（以 ALOHA/ACT 为例）：

    episode_0000.hdf5
    ├── observations/
    │   ├── qpos          (T, nq)       float64   关节位置
    │   ├── qvel          (T, nv)       float64   关节速度
    │   └── images/
    │       ├── cam_high  (T, 480, 640, 3) uint8   顶部相机 RGB
    │       └── cam_wrist (T, 480, 640, 3) uint8   腕部相机 RGB
    ├── action            (T, nu)       float64   控制指令
    └── [attrs]
        ├── sim=True/False
        ├── robot_type="aloha"
        └── num_timesteps=400

    维度说明:
      T  = 时间步数（一集的长度）
      nq = 关节位置维度（取决于机器人，例如 Panda=7）
      nv = 关节速度维度（通常 == nq）
      nu = 动作维度（控制信号维度）
      H, W, C = 图像高、宽、通道数
    """)

    # 创建一个符合标准的 episode 文件
    episode_data = generate_robot_episode(
        num_timesteps=200, nq=7, nv=7, nu=7,
        img_h=120, img_w=160, include_images=True
    )

    filepath = os.path.join(work_dir, "episode_0000.hdf5")

    with h5py.File(filepath, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=episode_data["observations"]["qpos"])
        obs.create_dataset("qvel", data=episode_data["observations"]["qvel"])

        imgs = obs.create_group("images")
        for cam_name, cam_data in episode_data["observations"]["images"].items():
            imgs.create_dataset(cam_name, data=cam_data)

        f.create_dataset("action", data=episode_data["action"])

        for key, val in episode_data["metadata"].items():
            f.attrs[key] = val

    print(f"  标准 episode 文件: {filepath}")
    print(f"  文件大小: {file_size_str(filepath)}")

    # 验证读取
    with h5py.File(filepath, "r") as f:
        print(f"\n  数据形状验证:")
        print(f"    qpos:     {f['observations/qpos'].shape}")
        print(f"    qvel:     {f['observations/qvel'].shape}")
        print(f"    cam_high: {f['observations/images/cam_high'].shape}")
        print(f"    action:   {f['action'].shape}")

    return filepath


# ============================================================
# 第 4 节：压缩选项与文件体积
# ============================================================

def section_4_compression(work_dir):
    print(DIVIDER)
    print("第 4 节：压缩选项与文件体积影响")
    print(DIVIDER)

    print("""
    HDF5 支持多种压缩算法，在创建 Dataset 时指定：

    常用选项:
      compression="gzip"   通用，压缩比好，速度中等
      compression="lzf"    速度快，压缩比稍低
      compression="szip"   科学计算常用（需额外安装）

    compression_opts=N (gzip only):
      0 = 无压缩
      1 = 最快
      9 = 最高压缩比

    重要规律:
      - 关节数据（float64）：压缩效果一般（数据随机性强）
      - 图像数据（uint8）：压缩效果非常好（像素间有空间相关性）
    """)

    T = 200
    qpos = np.random.randn(T, 7).astype(np.float64)
    images = np.random.randint(0, 256, (T, 120, 160, 3), dtype=np.uint8)

    # 用渐变图像替代纯随机（更接近真实场景，压缩效果更明显）
    gradient = np.linspace(0, 255, 160, dtype=np.uint8)
    realistic_images = np.tile(gradient, (T, 120, 1))
    realistic_images = np.stack([realistic_images] * 3, axis=-1)
    noise = np.random.randint(0, 20, realistic_images.shape, dtype=np.uint8)
    realistic_images = np.clip(realistic_images.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    configs = [
        ("无压缩", {}),
        ("gzip-1 (最快)", {"compression": "gzip", "compression_opts": 1}),
        ("gzip-4 (默认)", {"compression": "gzip", "compression_opts": 4}),
        ("gzip-9 (最强)", {"compression": "gzip", "compression_opts": 9}),
        ("lzf (快速)", {"compression": "lzf"}),
    ]

    print(f"  测试数据:")
    print(f"    qpos: {qpos.shape} ({qpos.nbytes / 1024:.1f} KB 原始)")
    print(f"    真实感图像: {realistic_images.shape} ({realistic_images.nbytes / 1024:.1f} KB 原始)")

    print(f"\n  {'压缩方式':<20} {'文件大小':>10} {'写入耗时':>10} {'读取耗时':>10}")
    print(f"  {'-' * 55}")

    for name, opts in configs:
        fpath = os.path.join(work_dir, f"compress_test_{name}.hdf5")

        # 写入
        t0 = time.perf_counter()
        with h5py.File(fpath, "w") as f:
            f.create_dataset("qpos", data=qpos, **opts)
            f.create_dataset("images", data=realistic_images, **opts)
        write_time = time.perf_counter() - t0

        # 读取
        t0 = time.perf_counter()
        with h5py.File(fpath, "r") as f:
            _ = f["qpos"][:]
            _ = f["images"][:]
        read_time = time.perf_counter() - t0

        size = file_size_str(fpath)
        print(f"  {name:<20} {size:>10} {write_time*1000:>8.1f}ms {read_time*1000:>8.1f}ms")

    print("""
    经验法则:
      - 数值数据（qpos/qvel/action）：gzip-1 或 lzf 足矣
      - 图像数据：gzip-4 是个好的折中
      - 如果读取速度是第一优先级：用 lzf 或不压缩
      - 如果存储成本是第一优先级：用 gzip-9
    """)


# ============================================================
# 第 5 节：分块策略（Chunking）
# ============================================================

def section_5_chunking(work_dir):
    print(DIVIDER)
    print("第 5 节：分块策略（Chunking）")
    print(DIVIDER)

    print("""
    什么是 Chunking？

    HDF5 默认把数据连续存储（contiguous）。
    启用 chunking 后，数据被切分成固定大小的「块」独立存储。

    为什么需要 chunking？
      1. 启用压缩必须先启用 chunking
      2. 不同的 chunk 大小影响读取模式的性能
      3. 支持数据追加（resize）

    chunk 大小选择策略:
    ┌───────────────────────────────────────────────────┐
    │  访问模式            推荐 chunk shape              │
    │  ─────────           ────────────────              │
    │  顺序读整条轨迹      (T, nq) 或 (T, nv)           │
    │  随机读单个时间步     (1, nq)                      │
    │  读取时间窗口         (window_size, nq)            │
    │  图像顺序读          (1, H, W, C) 一帧一 chunk    │
    │  图像批量读          (batch, H, W, C)              │
    └───────────────────────────────────────────────────┘
    """)

    T = 1000
    nq = 7
    data = np.random.randn(T, nq).astype(np.float64)

    chunk_configs = [
        ("无 chunking (contiguous)", None),
        ("chunk=(1, 7) 单步", (1, nq)),
        ("chunk=(10, 7) 10步窗口", (10, nq)),
        ("chunk=(100, 7) 100步窗口", (100, nq)),
        ("chunk=(1000, 7) 整条轨迹", (T, nq)),
    ]

    # 测试不同 chunk 大小对「读取中间 50 步」的性能影响
    print(f"  基准测试: 从 T={T} 的轨迹中读取第 450-500 步 (50 步)")
    print(f"  数据 shape: {data.shape}, dtype: {data.dtype}\n")

    print(f"  {'chunk 策略':<30} {'文件大小':>10} {'顺序读全部':>12} {'随机读50步':>12}")
    print(f"  {'-' * 68}")

    for name, chunks in chunk_configs:
        fpath = os.path.join(work_dir, f"chunk_test.hdf5")
        kwargs = {}
        if chunks is not None:
            kwargs["chunks"] = chunks
            kwargs["compression"] = "gzip"

        with h5py.File(fpath, "w") as f:
            f.create_dataset("data", data=data, **kwargs)

        # 顺序读全部
        t0 = time.perf_counter()
        for _ in range(10):
            with h5py.File(fpath, "r") as f:
                _ = f["data"][:]
        seq_time = (time.perf_counter() - t0) / 10

        # 随机读中间 50 步
        t0 = time.perf_counter()
        for _ in range(10):
            with h5py.File(fpath, "r") as f:
                _ = f["data"][450:500]
        rand_time = (time.perf_counter() - t0) / 10

        size = file_size_str(fpath)
        print(f"  {name:<30} {size:>10} {seq_time*1000:>10.2f}ms {rand_time*1000:>10.2f}ms")

    print("""
    观察:
      - chunk=(1, 7) 随机读最快，但文件大小和顺序读性能会变差
      - chunk=(100, 7) 通常是个好折中
      - 图像数据建议 chunk=(1, H, W, C)，因为通常按帧访问
    """)


# ============================================================
# 第 6 节：切片读取 — 只读你需要的数据
# ============================================================

def section_6_slicing(work_dir):
    print(DIVIDER)
    print("第 6 节：切片读取 — 只读你需要的数据")
    print(DIVIDER)

    # 创建一个较大的测试文件
    T = 2000
    episode = generate_robot_episode(T, nq=14, nv=14, nu=14, include_images=False)
    fpath = os.path.join(work_dir, "slicing_demo.hdf5")

    with h5py.File(fpath, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=episode["observations"]["qpos"],
                           chunks=(50, 14), compression="gzip")
        obs.create_dataset("qvel", data=episode["observations"]["qvel"],
                           chunks=(50, 14), compression="gzip")
        f.create_dataset("action", data=episode["action"],
                         chunks=(50, 14), compression="gzip")

    print(f"  测试文件: {fpath}")
    print(f"  轨迹长度: T={T}, nq=14\n")

    with h5py.File(fpath, "r") as f:
        qpos = f["observations/qpos"]  # 注意：这是一个 Dataset 对象，不是数组！

        # --- 各种切片方式 ---

        # 1. 读取前 10 步
        first_10 = qpos[:10]
        print(f"  qpos[:10]           → shape = {first_10.shape}")

        # 2. 读取最后 5 步
        last_5 = qpos[-5:]
        print(f"  qpos[-5:]           → shape = {last_5.shape}")

        # 3. 读取第 100-200 步
        window = qpos[100:200]
        print(f"  qpos[100:200]       → shape = {window.shape}")

        # 4. 每隔 10 步采样
        subsampled = qpos[::10]
        print(f"  qpos[::10]          → shape = {subsampled.shape}")

        # 5. 只读前 3 个关节
        first_3_joints = qpos[:, :3]
        print(f"  qpos[:, :3]         → shape = {first_3_joints.shape}")

        # 6. 特定时间步 + 特定关节
        specific = qpos[100:200, 2:5]
        print(f"  qpos[100:200, 2:5]  → shape = {specific.shape}")

        # 7. 使用 fancy indexing（高级索引）
        indices = [0, 50, 100, 150, 199]
        selected = qpos[indices]
        print(f"  qpos[[0,50,...,199]] → shape = {selected.shape}")

    # --- 性能对比：全量读 vs 切片读 ---
    print(f"\n  性能对比: 全量读 vs 切片读 (T={T})")
    print(f"  {'-' * 50}")

    with h5py.File(fpath, "r") as f:
        # 全量读取
        t0 = time.perf_counter()
        for _ in range(20):
            _ = f["observations/qpos"][:]
        full_time = (time.perf_counter() - t0) / 20

        # 切片读取 (10%)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = f["observations/qpos"][900:1100]
        slice_time = (time.perf_counter() - t0) / 20

        print(f"  全量读取 (T={T}):  {full_time*1000:.2f}ms")
        print(f"  切片读取 (200步):  {slice_time*1000:.2f}ms")
        print(f"  加速比:            {full_time/slice_time:.1f}x")


# ============================================================
# 第 7 节：追加数据到现有文件
# ============================================================

def section_7_appending(work_dir):
    print(DIVIDER)
    print("第 7 节：追加数据到现有文件")
    print(DIVIDER)

    print("""
    HDF5 支持在创建 Dataset 时指定 maxshape，之后可以 resize 来追加数据。
    这在在线数据采集场景中很有用（边采集边写入）。
    """)

    fpath = os.path.join(work_dir, "append_demo.hdf5")
    nq = 7

    # 第一步：创建文件，设置 maxshape 允许时间轴扩展
    with h5py.File(fpath, "w") as f:
        f.create_dataset(
            "qpos",
            shape=(0, nq),           # 初始为空
            maxshape=(None, nq),     # None 表示该维度可以无限扩展
            dtype=np.float64,
            chunks=(100, nq),
            compression="gzip",
        )
        f.attrs["num_timesteps"] = 0
        print(f"  创建空文件，qpos shape = (0, {nq})")

    # 第二步：模拟在线采集，分批追加数据
    batch_size = 50
    num_batches = 5

    for i in range(num_batches):
        new_data = np.random.randn(batch_size, nq).astype(np.float64)

        with h5py.File(fpath, "a") as f:  # "a" = append 模式
            ds = f["qpos"]
            old_len = ds.shape[0]
            new_len = old_len + batch_size

            # 扩展 dataset
            ds.resize(new_len, axis=0)
            ds[old_len:new_len] = new_data

            # 更新元数据
            f.attrs["num_timesteps"] = new_len

        print(f"  批次 {i+1}: 追加 {batch_size} 步 → 总计 {(i+1)*batch_size} 步")

    # 验证
    with h5py.File(fpath, "r") as f:
        print(f"\n  最终 qpos shape: {f['qpos'].shape}")
        print(f"  元数据 num_timesteps: {f.attrs['num_timesteps']}")
        assert f["qpos"].shape == (num_batches * batch_size, nq)
        print("  ✓ 追加写入验证通过")


# ============================================================
# 第 8 节：遍历 episode 目录
# ============================================================

def section_8_iterate_episodes(work_dir):
    print(DIVIDER)
    print("第 8 节：遍历 episode 目录")
    print(DIVIDER)

    # 先创建一批 episode 文件
    episode_dir = os.path.join(work_dir, "episodes")
    os.makedirs(episode_dir, exist_ok=True)

    num_episodes = 5
    for ep_idx in range(num_episodes):
        T = np.random.randint(100, 300)
        ep = generate_robot_episode(T, nq=7, nv=7, nu=7, include_images=False)
        fpath = os.path.join(episode_dir, f"episode_{ep_idx:04d}.hdf5")

        with h5py.File(fpath, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=ep["observations"]["qpos"],
                               compression="gzip")
            obs.create_dataset("qvel", data=ep["observations"]["qvel"],
                               compression="gzip")
            f.create_dataset("action", data=ep["action"], compression="gzip")
            for k, v in ep["metadata"].items():
                f.attrs[k] = v
            f.attrs["num_timesteps"] = T

    print(f"  已创建 {num_episodes} 个 episode 文件\n")

    # --- 方法 1: 简单遍历，收集元信息 ---
    print("  方法 1: 遍历收集元信息")
    episode_files = sorted([
        f for f in os.listdir(episode_dir) if f.endswith(".hdf5")
    ])

    total_steps = 0
    for fname in episode_files:
        fpath = os.path.join(episode_dir, fname)
        with h5py.File(fpath, "r") as f:
            T = f.attrs["num_timesteps"]
            qpos_shape = f["observations/qpos"].shape
            total_steps += T
            print(f"    {fname}: T={T}, qpos={qpos_shape}")

    print(f"    合计: {total_steps} 步, {len(episode_files)} 集\n")

    # --- 方法 2: 构建数据集索引 ---
    print("  方法 2: 构建数据集索引 (用于训练时的随机采样)")

    index = []
    cumulative = 0
    for fname in episode_files:
        fpath = os.path.join(episode_dir, fname)
        with h5py.File(fpath, "r") as f:
            T = int(f.attrs["num_timesteps"])
            index.append({
                "file": fpath,
                "episode_idx": len(index),
                "num_timesteps": T,
                "global_start": cumulative,
                "global_end": cumulative + T,
            })
            cumulative += T

    print(f"    索引条目数: {len(index)}")
    print(f"    全局时间步范围: 0 ~ {cumulative}")

    # 通过全局索引定位到具体文件和局部位置
    global_idx = cumulative // 2  # 取中间位置
    for entry in index:
        if entry["global_start"] <= global_idx < entry["global_end"]:
            local_idx = global_idx - entry["global_start"]
            print(f"    全局索引 {global_idx} → {os.path.basename(entry['file'])} 的第 {local_idx} 步")
            break


# ============================================================
# 第 9 节：实用技巧汇总
# ============================================================

def section_9_tips():
    print(DIVIDER)
    print("第 9 节：实用技巧汇总")
    print(DIVIDER)

    print("""
    ┌──────────────────── HDF5 最佳实践 ────────────────────┐
    │                                                       │
    │  1. 文件打开模式                                       │
    │     "r"  只读                                         │
    │     "w"  写入（覆盖已有文件）                           │
    │     "a"  追加（读写，不覆盖）                           │
    │     "r+" 读写（文件必须已存在）                         │
    │                                                       │
    │  2. 始终用 with 语句                                   │
    │     with h5py.File("data.hdf5", "r") as f:            │
    │         data = f["qpos"][:]                            │
    │     # 自动关闭文件句柄                                 │
    │                                                       │
    │  3. Dataset vs 数组                                    │
    │     f["qpos"]     → h5py.Dataset（懒加载引用）         │
    │     f["qpos"][:]  → numpy.ndarray（实际加载到内存）     │
    │     f["qpos"][0]  → numpy.ndarray（只加载第 0 步）      │
    │                                                       │
    │  4. 字符串属性                                         │
    │     f.attrs["name"] = "robot"         # 写入           │
    │     name = f.attrs["name"]            # 读取           │
    │     # 注意: h5py 返回的字符串可能是 bytes              │
    │     # 安全做法: str(f.attrs["name"])                   │
    │                                                       │
    │  5. 检查键是否存在                                     │
    │     if "observations/qpos" in f:                       │
    │         data = f["observations/qpos"][:]               │
    │                                                       │
    │  6. 并发注意事项                                       │
    │     - HDF5 不支持多进程同时写入同一文件                 │
    │     - 多进程只读是安全的                                │
    │     - 需要并发写时，考虑 SWMR 模式或每进程一个文件     │
    │                                                       │
    │  7. 大文件策略                                         │
    │     - 图像数据一定要开压缩（gzip-4 + chunk per frame） │
    │     - 数值数据压缩收益有限，但不会有明显负面影响       │
    │     - maxshape=None 预留扩展性                         │
    └───────────────────────────────────────────────────────┘
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("  第 5 章 · 01 — HDF5 深度实战")
    print(DIVIDER)

    # 创建工作目录
    work_dir = tempfile.mkdtemp(prefix="ch05_hdf5_")
    print(f"\n  工作目录: {work_dir}\n")

    try:
        section_1_what_is_hdf5()
        section_2_create_hdf5(work_dir)
        section_3_robot_data_structure(work_dir)
        section_4_compression(work_dir)
        section_5_chunking(work_dir)
        section_6_slicing(work_dir)
        section_7_appending(work_dir)
        section_8_iterate_episodes(work_dir)
        section_9_tips()

        print(DIVIDER)
        print("  所有 HDF5 示例运行完毕！")
        print(DIVIDER)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"\n  已清理工作目录: {work_dir}")


if __name__ == "__main__":
    main()
