"""
第 5 章 · 05 - 动手练习

目标: 通过实际编码练习，巩固本章学到的数据格式知识。
     每个练习都有 assert 验证，确保实现正确。

练习:
  1. 创建符合标准的 HDF5 episode 文件
  2. 合并多个 episode 文件为一个大文件
  3. 构建数据集索引（从多文件中提取元信息）
  4. 格式转换 pipeline（PKL → HDF5 → NPZ 三步转换）

运行: python 05_exercises.py
依赖: pip install numpy h5py
"""

import numpy as np
import pickle
import os
import time
import tempfile
import shutil

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("错误: h5py 未安装。请运行 pip install h5py")

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
# 练习 1：创建标准 HDF5 Episode 文件
# ============================================================

def exercise_1_create_episode(work_dir):
    """
    练习 1: 创建一个符合 ALOHA 标准的 HDF5 episode 文件。

    要求:
      - 文件名: episode_0000.hdf5
      - /observations/qpos: shape (T, 14), dtype float64
      - /observations/qvel: shape (T, 14), dtype float64
      - /observations/images/cam_high: shape (T, H, W, 3), dtype uint8, 带压缩
      - /action: shape (T, 14), dtype float64
      - 属性: sim=True, robot_type="aloha", num_timesteps=T
      - T=300, H=60, W=80 (用小图像节省内存)
      - 所有数值数据需要有物理意义（使用正弦波生成）
    """
    print(DIVIDER)
    print("练习 1：创建标准 HDF5 Episode 文件")
    print(DIVIDER)

    T = 300
    nq = 14
    img_h, img_w = 60, 80
    filepath = os.path.join(work_dir, "episode_0000.hdf5")

    # ==================== 你的代码 ====================
    t = np.linspace(0, 4 * np.pi, T)

    # 关节位置：正弦波组合，模拟双臂运动
    qpos = np.column_stack([
        np.sin(t * (i + 1) * 0.3) * (0.5 - 0.02 * i)
        for i in range(nq)
    ]).astype(np.float64)

    # 关节速度：qpos 的导数近似
    qvel = np.column_stack([
        np.cos(t * (i + 1) * 0.3) * (i + 1) * 0.3 * (0.5 - 0.02 * i)
        for i in range(nq)
    ]).astype(np.float64)

    # 动作：略微偏移的正弦波
    action = np.column_stack([
        np.sin(t * (i + 1) * 0.3 + 0.05) * (0.5 - 0.02 * i)
        for i in range(nq)
    ]).astype(np.float64)

    # 伪图像：渐变 + 噪声
    gradient_h = np.linspace(50, 200, img_h).reshape(-1, 1, 1).astype(np.float32)
    gradient_w = np.linspace(30, 180, img_w).reshape(1, -1, 1).astype(np.float32)
    base_img = (gradient_h + gradient_w) / 2
    base_img = np.broadcast_to(base_img, (img_h, img_w, 3)).copy()

    images = np.zeros((T, img_h, img_w, 3), dtype=np.uint8)
    for i in range(T):
        noise = np.random.randint(-10, 10, (img_h, img_w, 3))
        offset = int(20 * np.sin(2 * np.pi * i / T))
        frame = np.clip(base_img + noise + offset, 0, 255)
        images[i] = frame.astype(np.uint8)

    with h5py.File(filepath, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=qvel)

        img_group = obs.create_group("images")
        img_group.create_dataset(
            "cam_high", data=images,
            chunks=(1, img_h, img_w, 3),
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset("action", data=action)

        f.attrs["sim"] = True
        f.attrs["robot_type"] = "aloha"
        f.attrs["num_timesteps"] = T
    # ==================== 代码结束 ====================

    # --- 验证 ---
    print(f"\n  验证中...\n")

    assert os.path.exists(filepath), "文件不存在"

    with h5py.File(filepath, "r") as f:
        # 结构检查
        assert "observations" in f, "缺少 /observations"
        assert "observations/qpos" in f, "缺少 /observations/qpos"
        assert "observations/qvel" in f, "缺少 /observations/qvel"
        assert "observations/images/cam_high" in f, "缺少 /observations/images/cam_high"
        assert "action" in f, "缺少 /action"

        # 形状检查
        assert f["observations/qpos"].shape == (T, nq), \
            f"qpos shape 错误: {f['observations/qpos'].shape}"
        assert f["observations/qvel"].shape == (T, nq), \
            f"qvel shape 错误: {f['observations/qvel'].shape}"
        assert f["observations/images/cam_high"].shape == (T, img_h, img_w, 3), \
            f"cam_high shape 错误: {f['observations/images/cam_high'].shape}"
        assert f["action"].shape == (T, nq), \
            f"action shape 错误: {f['action'].shape}"

        # 类型检查
        assert f["observations/qpos"].dtype == np.float64
        assert f["observations/images/cam_high"].dtype == np.uint8

        # 压缩检查
        assert f["observations/images/cam_high"].compression is not None, \
            "cam_high 未启用压缩"

        # 属性检查
        assert f.attrs["sim"] == True
        assert f.attrs["robot_type"] == "aloha"
        assert f.attrs["num_timesteps"] == T

        # 数据合理性检查
        qpos_data = f["observations/qpos"][:]
        assert np.all(np.isfinite(qpos_data)), "qpos 包含 nan/inf"
        assert np.abs(qpos_data).max() < 10, "qpos 数值范围异常"

        print(f"  文件: {filepath}")
        print(f"  大小: {file_size_str(filepath)}")
        print(f"  qpos shape:     {f['observations/qpos'].shape}")
        print(f"  cam_high shape: {f['observations/images/cam_high'].shape}")
        print(f"  压缩方式:       {f['observations/images/cam_high'].compression}")

    print(f"\n  ✓ 练习 1 全部验证通过！")
    return filepath


# ============================================================
# 练习 2：合并多个 Episode 文件
# ============================================================

def exercise_2_merge_episodes(work_dir):
    """
    练习 2: 编写一个函数，把多个 episode 文件合并为一个大文件。

    要求:
      - 输入: 一个目录路径，包含多个 episode_XXXX.hdf5 文件
      - 输出: 一个合并后的文件，结构如下:
          /episode_0000/observations/qpos  (T0, nq)
          /episode_0000/action             (T0, nu)
          /episode_0001/observations/qpos  (T1, nq)
          /episode_0001/action             (T1, nu)
          ...
          [attrs] total_episodes, total_timesteps
      - 保留每个 episode 的元数据
    """
    print(DIVIDER)
    print("练习 2：合并多个 Episode 文件")
    print(DIVIDER)

    # 创建测试数据：5 个不同长度的 episode
    episodes_dir = os.path.join(work_dir, "episodes_to_merge")
    os.makedirs(episodes_dir, exist_ok=True)

    episode_lengths = [150, 200, 180, 250, 170]
    nq = 7

    for i, T in enumerate(episode_lengths):
        t = np.linspace(0, 2 * np.pi, T)
        fpath = os.path.join(episodes_dir, f"episode_{i:04d}.hdf5")
        with h5py.File(fpath, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.column_stack(
                [np.sin(t * (j+1)) for j in range(nq)]).astype(np.float64))
            obs.create_dataset("qvel", data=np.column_stack(
                [np.cos(t * (j+1)) for j in range(nq)]).astype(np.float64))
            f.create_dataset("action", data=np.column_stack(
                [np.sin(t * (j+1) + 0.1) for j in range(nq)]).astype(np.float64))
            f.attrs["num_timesteps"] = T
            f.attrs["robot_type"] = "franka_panda"
            f.attrs["episode_idx"] = i

    print(f"  已创建 {len(episode_lengths)} 个 episode 文件")

    # ==================== 你的代码 ====================
    def merge_episodes(input_dir: str, output_path: str):
        """合并目录中所有 episode 文件"""
        files = sorted([
            f for f in os.listdir(input_dir) if f.endswith(".hdf5")
        ])

        total_episodes = 0
        total_timesteps = 0

        with h5py.File(output_path, "w") as out_f:
            for fname in files:
                ep_name = os.path.splitext(fname)[0]
                fpath = os.path.join(input_dir, fname)

                with h5py.File(fpath, "r") as in_f:
                    ep_group = out_f.create_group(ep_name)

                    # 递归复制所有 group 和 dataset
                    def copy_item(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            ep_group.create_dataset(name, data=obj[:], compression="gzip")
                        elif isinstance(obj, h5py.Group):
                            if name not in ep_group:
                                ep_group.create_group(name)
                    in_f.visititems(copy_item)

                    # 复制 episode 级别的属性
                    for key, val in in_f.attrs.items():
                        ep_group.attrs[key] = val

                    T = int(in_f.attrs.get("num_timesteps", 0))
                    total_timesteps += T
                    total_episodes += 1

            out_f.attrs["total_episodes"] = total_episodes
            out_f.attrs["total_timesteps"] = total_timesteps

        return total_episodes, total_timesteps
    # ==================== 代码结束 ====================

    merged_path = os.path.join(work_dir, "merged_dataset.hdf5")
    num_eps, num_steps = merge_episodes(episodes_dir, merged_path)

    # --- 验证 ---
    print(f"\n  验证中...\n")

    with h5py.File(merged_path, "r") as f:
        assert f.attrs["total_episodes"] == len(episode_lengths), \
            f"total_episodes 错误: {f.attrs['total_episodes']}"
        assert f.attrs["total_timesteps"] == sum(episode_lengths), \
            f"total_timesteps 错误: {f.attrs['total_timesteps']}"

        for i, T in enumerate(episode_lengths):
            ep_name = f"episode_{i:04d}"
            assert ep_name in f, f"缺少 {ep_name}"
            assert f[ep_name]["observations/qpos"].shape == (T, nq), \
                f"{ep_name} qpos shape 错误"
            assert f[ep_name]["action"].shape == (T, nq), \
                f"{ep_name} action shape 错误"
            assert f[ep_name].attrs["num_timesteps"] == T

            # 验证数据正确性（与原文件比较）
            orig_path = os.path.join(episodes_dir, f"episode_{i:04d}.hdf5")
            with h5py.File(orig_path, "r") as orig:
                assert np.allclose(
                    f[ep_name]["observations/qpos"][:],
                    orig["observations/qpos"][:]
                ), f"{ep_name} qpos 数据不一致"

    print(f"  合并文件: {merged_path}")
    print(f"  文件大小: {file_size_str(merged_path)}")
    print(f"  总 episode 数: {num_eps}")
    print(f"  总时间步数: {num_steps}")
    print(f"\n  ✓ 练习 2 全部验证通过！")


# ============================================================
# 练习 3：构建数据集索引
# ============================================================

def exercise_3_build_index(work_dir):
    """
    练习 3: 从一个包含多个 episode 文件的目录构建数据集索引。

    要求:
      - 扫描目录中所有 .hdf5 文件
      - 提取每个文件的: 文件名, 时间步数, qpos 维度, 是否有图像, 文件大小
      - 计算全局时间步映射 (global_start, global_end)
      - 返回 index 列表和汇总统计

    索引用途:
      - 训练时通过全局索引快速定位到具体 episode 和局部位置
      - 数据集概览统计
    """
    print(DIVIDER)
    print("练习 3：构建数据集索引")
    print(DIVIDER)

    # 创建测试数据
    dataset_dir = os.path.join(work_dir, "indexed_dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    expected_info = []
    for i in range(10):
        T = np.random.randint(80, 400)
        nq = 7 if i < 7 else 14
        has_images = i % 3 == 0

        t = np.linspace(0, 2 * np.pi, T)
        fpath = os.path.join(dataset_dir, f"episode_{i:04d}.hdf5")

        with h5py.File(fpath, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.random.randn(T, nq).astype(np.float64),
                               compression="gzip")
            obs.create_dataset("qvel", data=np.random.randn(T, nq).astype(np.float64),
                               compression="gzip")
            f.create_dataset("action", data=np.random.randn(T, nq).astype(np.float64),
                             compression="gzip")
            if has_images:
                imgs = obs.create_group("images")
                imgs.create_dataset("cam_high",
                                    data=np.random.randint(0, 256, (T, 30, 40, 3), dtype=np.uint8),
                                    compression="gzip")
            f.attrs["num_timesteps"] = T
            f.attrs["robot_type"] = "franka_panda" if nq == 7 else "aloha"

        expected_info.append({
            "T": T, "nq": nq, "has_images": has_images,
            "file_size": os.path.getsize(fpath),
        })

    print(f"  已创建 {len(expected_info)} 个 episode 文件\n")

    # ==================== 你的代码 ====================
    def build_dataset_index(dataset_dir: str) -> tuple:
        """
        构建数据集索引。

        返回:
            index: list of dict，每个 episode 的信息
            summary: dict，汇总统计
        """
        files = sorted([
            f for f in os.listdir(dataset_dir) if f.endswith((".hdf5", ".h5"))
        ])

        index = []
        cumulative_steps = 0

        for fname in files:
            fpath = os.path.join(dataset_dir, fname)

            with h5py.File(fpath, "r") as f:
                T = int(f.attrs.get("num_timesteps", f["action"].shape[0]))
                nq = f["observations/qpos"].shape[1]
                has_images = "images" in f["observations"]
                robot_type = str(f.attrs.get("robot_type", "unknown"))

            entry = {
                "filename": fname,
                "filepath": fpath,
                "episode_idx": len(index),
                "num_timesteps": T,
                "nq": nq,
                "has_images": has_images,
                "robot_type": robot_type,
                "file_size_bytes": os.path.getsize(fpath),
                "global_start": cumulative_steps,
                "global_end": cumulative_steps + T,
            }
            index.append(entry)
            cumulative_steps += T

        summary = {
            "total_episodes": len(index),
            "total_timesteps": cumulative_steps,
            "total_size_bytes": sum(e["file_size_bytes"] for e in index),
            "avg_episode_length": cumulative_steps / len(index) if index else 0,
            "min_episode_length": min(e["num_timesteps"] for e in index) if index else 0,
            "max_episode_length": max(e["num_timesteps"] for e in index) if index else 0,
            "has_images_count": sum(1 for e in index if e["has_images"]),
            "robot_types": list(set(e["robot_type"] for e in index)),
        }

        return index, summary

    def lookup_global_index(index: list, global_idx: int) -> tuple:
        """通过全局时间步索引定位到具体 episode 和局部位置"""
        for entry in index:
            if entry["global_start"] <= global_idx < entry["global_end"]:
                local_idx = global_idx - entry["global_start"]
                return entry, local_idx
        raise IndexError(f"全局索引 {global_idx} 超出范围")
    # ==================== 代码结束 ====================

    index, summary = build_dataset_index(dataset_dir)

    # --- 验证 ---
    print(f"  验证中...\n")

    assert len(index) == 10, f"索引条目数错误: {len(index)}"
    assert summary["total_episodes"] == 10

    total_T = sum(info["T"] for info in expected_info)
    assert summary["total_timesteps"] == total_T, \
        f"总步数错误: {summary['total_timesteps']} vs {total_T}"

    # 验证全局索引的连续性
    for i in range(len(index) - 1):
        assert index[i]["global_end"] == index[i+1]["global_start"], \
            f"全局索引不连续: episode {i}"

    # 验证全局索引查找
    for i, entry in enumerate(index):
        # 检查每个 episode 的第一步
        found_entry, local_idx = lookup_global_index(index, entry["global_start"])
        assert found_entry["episode_idx"] == i
        assert local_idx == 0

        # 检查每个 episode 的最后一步
        found_entry, local_idx = lookup_global_index(index, entry["global_end"] - 1)
        assert found_entry["episode_idx"] == i
        assert local_idx == entry["num_timesteps"] - 1

    # 验证有图像的 episode 数量
    actual_img_count = sum(1 for info in expected_info if info["has_images"])
    assert summary["has_images_count"] == actual_img_count

    # 打印索引摘要
    print(f"  数据集索引摘要:")
    print(f"    总 episode 数:    {summary['total_episodes']}")
    print(f"    总时间步:         {summary['total_timesteps']}")
    print(f"    平均 episode 长度: {summary['avg_episode_length']:.0f}")
    print(f"    最短/最长:        {summary['min_episode_length']}/{summary['max_episode_length']}")
    print(f"    有图像的 episode: {summary['has_images_count']}")
    print(f"    机器人类型:       {summary['robot_types']}")
    print(f"    总文件大小:       {summary['total_size_bytes']/1024:.1f} KB")

    # 演示全局索引查找
    mid_idx = total_T // 2
    entry, local = lookup_global_index(index, mid_idx)
    print(f"\n  全局索引 {mid_idx} → {entry['filename']} 第 {local} 步")

    print(f"\n  ✓ 练习 3 全部验证通过！")


# ============================================================
# 练习 4：格式转换 Pipeline
# ============================================================

def exercise_4_conversion_pipeline(work_dir):
    """
    练习 4: 构建一个完整的格式转换 pipeline。

    流程: PKL → HDF5 → NPZ
    要求:
      - 从 PKL 文件读取轨迹数据
      - 转换为标准 HDF5 格式（带压缩和元数据）
      - 再转换为 NPZ 格式
      - 每步都要验证数据一致性
      - 记录并输出每步的耗时和文件大小
    """
    print(DIVIDER)
    print("练习 4：格式转换 Pipeline (PKL → HDF5 → NPZ)")
    print(DIVIDER)

    # 创建测试 PKL 数据
    T = 500
    nq = 14
    t = np.linspace(0, 4 * np.pi, T)

    original_data = {
        "observations": {
            "qpos": np.column_stack([np.sin(t * (i+1) * 0.3) for i in range(nq)]).astype(np.float64),
            "qvel": np.column_stack([np.cos(t * (i+1) * 0.3) for i in range(nq)]).astype(np.float64),
        },
        "action": np.column_stack([np.sin(t * (i+1) * 0.3 + 0.05) for i in range(nq)]).astype(np.float64),
        "metadata": {
            "robot_type": "aloha",
            "num_timesteps": T,
            "hz": 50,
            "task": "bimanual_insertion",
            "sim": True,
        }
    }

    pkl_path = os.path.join(work_dir, "raw_episode.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(original_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  源文件: {pkl_path} ({file_size_str(pkl_path)})")
    print(f"  数据: T={T}, nq={nq}\n")

    # ==================== 你的代码 ====================
    pipeline_log = []

    def log_step(step_name, input_path, output_path, elapsed):
        entry = {
            "step": step_name,
            "input_size": file_size_str(input_path),
            "output_size": file_size_str(output_path),
            "elapsed_ms": elapsed * 1000,
        }
        pipeline_log.append(entry)
        print(f"  [{step_name}]")
        print(f"    输入: {entry['input_size']}")
        print(f"    输出: {entry['output_size']}")
        print(f"    耗时: {entry['elapsed_ms']:.1f}ms")

    # 步骤 1: PKL → HDF5
    h5_path = os.path.join(work_dir, "converted.hdf5")

    t0 = time.perf_counter()
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    with h5py.File(h5_path, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=pkl_data["observations"]["qpos"], compression="gzip")
        obs.create_dataset("qvel", data=pkl_data["observations"]["qvel"], compression="gzip")
        f.create_dataset("action", data=pkl_data["action"], compression="gzip")
        for key, val in pkl_data.get("metadata", {}).items():
            f.attrs[key] = val

    elapsed = time.perf_counter() - t0
    log_step("PKL → HDF5", pkl_path, h5_path, elapsed)

    # 验证步骤 1
    with h5py.File(h5_path, "r") as f:
        h5_qpos = f["observations/qpos"][:]
        assert np.allclose(original_data["observations"]["qpos"], h5_qpos), \
            "PKL → HDF5 数据不一致"
    print("    ✓ 数据一致性验证通过\n")

    # 步骤 2: HDF5 → NPZ
    npz_path = os.path.join(work_dir, "converted.npz")

    t0 = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        arrays = {}
        arrays["observations.qpos"] = f["observations/qpos"][:]
        arrays["observations.qvel"] = f["observations/qvel"][:]
        arrays["action"] = f["action"][:]
        # 元数据编码为数组
        for key, val in f.attrs.items():
            if isinstance(val, str):
                arrays[f"__str__metadata.{key}"] = np.array(val)
            else:
                arrays[f"metadata.{key}"] = np.array(val)

    np.savez_compressed(npz_path, **arrays)
    elapsed = time.perf_counter() - t0
    log_step("HDF5 → NPZ", h5_path, npz_path, elapsed)

    # 验证步骤 2
    with np.load(npz_path, allow_pickle=False) as npz_data:
        npz_qpos = npz_data["observations.qpos"]
        assert np.allclose(original_data["observations"]["qpos"], npz_qpos), \
            "HDF5 → NPZ 数据不一致"
    print("    ✓ 数据一致性验证通过\n")
    # ==================== 代码结束 ====================

    # --- 最终验证 ---
    print(f"  ────── Pipeline 汇总 ──────")
    for entry in pipeline_log:
        print(f"  {entry['step']}: {entry['input_size']} → {entry['output_size']} ({entry['elapsed_ms']:.1f}ms)")

    # 端到端一致性验证
    with np.load(npz_path, allow_pickle=False) as final_data:
        final_qpos = final_data["observations.qpos"]
        final_qvel = final_data["observations.qvel"]
        final_action = final_data["action"]

    assert np.allclose(original_data["observations"]["qpos"], final_qpos), \
        "端到端 qpos 不一致"
    assert np.allclose(original_data["observations"]["qvel"], final_qvel), \
        "端到端 qvel 不一致"
    assert np.allclose(original_data["action"], final_action), \
        "端到端 action 不一致"

    assert len(pipeline_log) == 2, "Pipeline 应该有 2 个步骤"

    print(f"\n  ✓ 练习 4 全部验证通过！（端到端数据一致性确认）")


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("  第 5 章 · 05 — 动手练习")
    print(DIVIDER)

    if not HAS_H5PY:
        print("\n  请先安装 h5py: pip install h5py\n")
        return

    work_dir = tempfile.mkdtemp(prefix="ch05_exercises_")
    print(f"\n  工作目录: {work_dir}\n")

    passed = 0
    total = 4

    try:
        exercise_1_create_episode(work_dir)
        passed += 1
        print()

        exercise_2_merge_episodes(work_dir)
        passed += 1
        print()

        exercise_3_build_index(work_dir)
        passed += 1
        print()

        exercise_4_conversion_pipeline(work_dir)
        passed += 1
        print()

    except AssertionError as e:
        print(f"\n  ✗ 验证失败: {e}")
    except Exception as e:
        print(f"\n  ✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"  已清理工作目录: {work_dir}")

    print(DIVIDER)
    print(f"  练习完成: {passed}/{total} 通过")
    if passed == total:
        print("  🎉 全部通过！你已掌握机器人数据格式的核心技能。")
    print(DIVIDER)


if __name__ == "__main__":
    main()
