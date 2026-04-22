"""
第 5 章 · 03 - 数据格式互转

目标: 构建一个通用的 DatasetConverter 类，实现 HDF5、PKL、NPZ 之间的无损转换，
     包括批量转换、元数据保留、性能对比。

核心知识点:
  1. HDF5 → PKL / NPZ 转换
  2. PKL → HDF5 转换
  3. NPZ → HDF5 转换
  4. DatasetConverter 统一接口
  5. 批量目录转换
  6. 元数据保留策略
  7. 读写性能对比

运行: python 03_data_conversion.py
依赖: pip install numpy h5py
"""

import numpy as np
import pickle
import os
import time
import tempfile
import shutil
from typing import Any

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("警告: h5py 未安装，HDF5 相关功能不可用。pip install h5py")

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


def file_size_str(path):
    """返回人类可读的文件大小"""
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def generate_episode(T=200, nq=7, nv=7, nu=7):
    """生成一条标准机器人轨迹"""
    t = np.linspace(0, 2 * np.pi, T)
    return {
        "observations": {
            "qpos": np.column_stack([np.sin(t * (i+1)) * 0.5 for i in range(nq)]).astype(np.float64),
            "qvel": np.column_stack([np.cos(t * (i+1)) * 0.3 for i in range(nv)]).astype(np.float64),
        },
        "action": np.column_stack([np.sin(t * (i+1) + 0.1) * 0.5 for i in range(nu)]).astype(np.float64),
        "metadata": {
            "robot_type": "franka_panda",
            "num_timesteps": T,
            "hz": 50,
            "task": "pick_and_place",
            "sim": True,
        }
    }


# ============================================================
# HDF5 读写辅助函数
# ============================================================

def _hdf5_group_to_dict(group) -> dict:
    """递归地把 HDF5 Group 转换为嵌套 dict"""
    result = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            result[key] = _hdf5_group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[:]
    return result


def _dict_to_hdf5_group(group, data: dict, compression="gzip"):
    """递归地把嵌套 dict 写入 HDF5 Group"""
    for key, val in data.items():
        if isinstance(val, dict):
            sub_group = group.create_group(key)
            _dict_to_hdf5_group(sub_group, val, compression)
        elif isinstance(val, np.ndarray):
            group.create_dataset(key, data=val, compression=compression)
        else:
            # 标量或字符串 → 存为 attribute
            group.attrs[key] = val


def read_hdf5(filepath: str) -> dict:
    """读取 HDF5 文件为嵌套 dict，包括元数据"""
    with h5py.File(filepath, "r") as f:
        data = _hdf5_group_to_dict(f)
        metadata = dict(f.attrs)
        if metadata:
            data["metadata"] = metadata
    return data


def write_hdf5(filepath: str, data: dict, compression="gzip"):
    """把嵌套 dict 写入 HDF5 文件"""
    with h5py.File(filepath, "w") as f:
        metadata = data.get("metadata", {})
        for key, val in data.items():
            if key == "metadata":
                for mk, mv in val.items():
                    f.attrs[mk] = mv
            elif isinstance(val, dict):
                grp = f.create_group(key)
                _dict_to_hdf5_group(grp, val, compression)
            elif isinstance(val, np.ndarray):
                f.create_dataset(key, data=val, compression=compression)


# ============================================================
# PKL 读写辅助函数
# ============================================================

def read_pkl(filepath: str) -> dict:
    """读取 PKL 文件"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pkl(filepath: str, data: dict):
    """写入 PKL 文件"""
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================
# NPZ 读写辅助函数
# ============================================================

def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """
    把嵌套 dict 展平为 key1.key2.key3 → ndarray 的形式。
    NPZ 不支持层级结构，所以需要展平。
    非数组值转为小数组存储。
    """
    flat = {}
    for key, val in d.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(val, dict):
            flat.update(_flatten_dict(val, full_key))
        elif isinstance(val, np.ndarray):
            flat[full_key] = val
        elif isinstance(val, (int, float, bool)):
            flat[full_key] = np.array(val)
        elif isinstance(val, str):
            flat[f"__str__{full_key}"] = np.array(val)
        else:
            flat[full_key] = np.array(val)
    return flat


def _unflatten_dict(flat: dict) -> dict:
    """把展平的 dict 还原为嵌套 dict"""
    result = {}
    for key, val in flat.items():
        # 还原字符串
        actual_key = key
        is_string = False
        if key.startswith("__str__"):
            actual_key = key[7:]
            is_string = True

        parts = actual_key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]

        if is_string:
            d[parts[-1]] = str(val)
        elif val.ndim == 0:
            d[parts[-1]] = val.item()
        else:
            d[parts[-1]] = val

    return result


def read_npz(filepath: str) -> dict:
    """读取 NPZ 文件并还原为嵌套 dict"""
    with np.load(filepath, allow_pickle=False) as data:
        flat = {key: data[key] for key in data.files}
    return _unflatten_dict(flat)


def write_npz(filepath: str, data: dict, compressed: bool = True):
    """把嵌套 dict 写入 NPZ 文件"""
    flat = _flatten_dict(data)
    if compressed:
        np.savez_compressed(filepath, **flat)
    else:
        np.savez(filepath, **flat)


# ============================================================
# 第 1 节：基础转换演示
# ============================================================

def section_1_basic_conversion(work_dir):
    print(DIVIDER)
    print("第 1 节：基础格式转换")
    print(DIVIDER)

    episode = generate_episode(T=200)

    # --- HDF5 → PKL ---
    print("\n  [HDF5 → PKL]")
    h5_path = os.path.join(work_dir, "episode.hdf5")
    pkl_path = os.path.join(work_dir, "episode.pkl")

    write_hdf5(h5_path, episode)
    h5_data = read_hdf5(h5_path)
    write_pkl(pkl_path, h5_data)

    print(f"    HDF5: {file_size_str(h5_path)}")
    print(f"    PKL:  {file_size_str(pkl_path)}")

    # 验证一致性
    pkl_data = read_pkl(pkl_path)
    assert np.allclose(h5_data["observations"]["qpos"], pkl_data["observations"]["qpos"])
    print("    ✓ 数据一致性验证通过")

    # --- HDF5 → NPZ ---
    print("\n  [HDF5 → NPZ]")
    npz_path = os.path.join(work_dir, "episode.npz")
    write_npz(npz_path, h5_data)
    print(f"    NPZ: {file_size_str(npz_path)}")

    npz_data = read_npz(npz_path)
    assert np.allclose(h5_data["observations"]["qpos"], npz_data["observations"]["qpos"])
    print("    ✓ 数据一致性验证通过")

    # --- PKL → HDF5 ---
    print("\n  [PKL → HDF5]")
    h5_from_pkl = os.path.join(work_dir, "from_pkl.hdf5")
    write_hdf5(h5_from_pkl, pkl_data)
    print(f"    HDF5: {file_size_str(h5_from_pkl)}")

    roundtrip = read_hdf5(h5_from_pkl)
    assert np.allclose(episode["observations"]["qpos"], roundtrip["observations"]["qpos"])
    print("    ✓ 往返转换验证通过")

    # --- NPZ → HDF5 ---
    print("\n  [NPZ → HDF5]")
    h5_from_npz = os.path.join(work_dir, "from_npz.hdf5")
    write_hdf5(h5_from_npz, npz_data)
    print(f"    HDF5: {file_size_str(h5_from_npz)}")

    roundtrip2 = read_hdf5(h5_from_npz)
    assert np.allclose(episode["observations"]["qpos"], roundtrip2["observations"]["qpos"])
    print("    ✓ 往返转换验证通过")


# ============================================================
# 第 2 节：DatasetConverter 类
# ============================================================

class DatasetConverter:
    """
    通用数据集格式转换器。

    支持的格式: hdf5, pkl, npz

    用法:
        converter = DatasetConverter()
        converter.convert("input.hdf5", "output.pkl")
        converter.convert("input.pkl", "output.npz")
        converter.batch_convert("input_dir/", "output_dir/", target_format="hdf5")
    """

    SUPPORTED_FORMATS = {"hdf5", "h5", "pkl", "pickle", "npz"}

    READERS = {
        "hdf5": read_hdf5,
        "h5": read_hdf5,
        "pkl": read_pkl,
        "pickle": read_pkl,
        "npz": read_npz,
    }

    WRITERS = {
        "hdf5": write_hdf5,
        "h5": write_hdf5,
        "pkl": write_pkl,
        "pickle": write_pkl,
        "npz": write_npz,
    }

    @staticmethod
    def _get_format(filepath: str) -> str:
        """从文件扩展名推断格式"""
        ext = os.path.splitext(filepath)[1].lstrip(".")
        if ext in ("hdf5", "h5"):
            return "hdf5"
        elif ext in ("pkl", "pickle"):
            return "pkl"
        elif ext == "npz":
            return "npz"
        else:
            raise ValueError(f"不支持的文件格式: .{ext}")

    def read(self, filepath: str) -> dict:
        """读取任意支持格式的文件"""
        fmt = self._get_format(filepath)
        reader = self.READERS[fmt]
        return reader(filepath)

    def write(self, filepath: str, data: dict):
        """写入任意支持格式的文件"""
        fmt = self._get_format(filepath)
        writer = self.WRITERS[fmt]
        writer(filepath, data)

    def convert(self, input_path: str, output_path: str, verbose: bool = True) -> dict:
        """
        转换单个文件。

        返回: {"input_size": int, "output_size": int, "read_time": float, "write_time": float}
        """
        src_fmt = self._get_format(input_path)
        dst_fmt = self._get_format(output_path)

        t0 = time.perf_counter()
        data = self.read(input_path)
        read_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.write(output_path, data)
        write_time = time.perf_counter() - t0

        stats = {
            "input_size": os.path.getsize(input_path),
            "output_size": os.path.getsize(output_path),
            "read_time": read_time,
            "write_time": write_time,
        }

        if verbose:
            print(f"    {src_fmt} → {dst_fmt}: "
                  f"{file_size_str(input_path)} → {file_size_str(output_path)} "
                  f"(读 {read_time*1000:.1f}ms, 写 {write_time*1000:.1f}ms)")

        return stats

    def batch_convert(self, input_dir: str, output_dir: str,
                      target_format: str = "hdf5",
                      source_extensions: tuple = (".pkl", ".npz", ".hdf5", ".h5"),
                      verbose: bool = True) -> list:
        """
        批量转换目录中的所有文件。

        参数:
            input_dir: 输入目录
            output_dir: 输出目录
            target_format: 目标格式 ("hdf5", "pkl", "npz")
            source_extensions: 要转换的源文件扩展名

        返回: 转换统计列表
        """
        os.makedirs(output_dir, exist_ok=True)
        ext_map = {"hdf5": ".hdf5", "pkl": ".pkl", "npz": ".npz"}
        target_ext = ext_map[target_format]

        files = sorted([
            f for f in os.listdir(input_dir)
            if any(f.endswith(ext) for ext in source_extensions)
        ])

        if verbose:
            print(f"    找到 {len(files)} 个文件待转换 → {target_format}")

        stats_list = []
        for fname in files:
            src = os.path.join(input_dir, fname)
            dst_name = os.path.splitext(fname)[0] + target_ext
            dst = os.path.join(output_dir, dst_name)

            # 跳过已经是目标格式的文件
            if fname.endswith(target_ext):
                if verbose:
                    print(f"    跳过 {fname} (已是目标格式)")
                continue

            stats = self.convert(src, dst, verbose=verbose)
            stats["source_file"] = fname
            stats_list.append(stats)

        return stats_list

    def verify_conversion(self, original_path: str, converted_path: str,
                          rtol: float = 1e-10) -> bool:
        """
        验证转换后数据与原始数据的一致性。
        """
        orig = self.read(original_path)
        conv = self.read(converted_path)

        def compare_nested(a, b, path=""):
            if isinstance(a, dict) and isinstance(b, dict):
                for key in a:
                    if key not in b:
                        print(f"    ✗ 键缺失: {path}.{key}")
                        return False
                    if not compare_nested(a[key], b[key], f"{path}.{key}"):
                        return False
                return True
            elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                if not np.allclose(a, b, rtol=rtol):
                    print(f"    ✗ 数据不一致: {path}")
                    return False
                return True
            else:
                if a != b:
                    print(f"    ✗ 值不一致: {path}: {a} vs {b}")
                    return False
                return True

        return compare_nested(orig, conv)


def section_2_converter_class(work_dir):
    print(DIVIDER)
    print("第 2 节：DatasetConverter 统一转换器")
    print(DIVIDER)

    converter = DatasetConverter()
    episode = generate_episode(T=300)

    # 先写入 HDF5 作为源文件
    src_h5 = os.path.join(work_dir, "source.hdf5")
    write_hdf5(src_h5, episode)

    # 转换为各种格式
    print("\n  单文件转换:")
    out_pkl = os.path.join(work_dir, "converted.pkl")
    out_npz = os.path.join(work_dir, "converted.npz")
    out_h5 = os.path.join(work_dir, "roundtrip.hdf5")

    converter.convert(src_h5, out_pkl)
    converter.convert(src_h5, out_npz)
    converter.convert(out_pkl, out_h5)

    # 验证往返一致性
    print("\n  往返验证:")
    ok = converter.verify_conversion(src_h5, out_h5)
    print(f"    HDF5 → PKL → HDF5: {'✓ 通过' if ok else '✗ 失败'}")


# ============================================================
# 第 3 节：批量目录转换
# ============================================================

def section_3_batch_conversion(work_dir):
    print(DIVIDER)
    print("第 3 节：批量目录转换")
    print(DIVIDER)

    # 创建一批 PKL episode 文件
    src_dir = os.path.join(work_dir, "pkl_episodes")
    os.makedirs(src_dir, exist_ok=True)

    num_episodes = 8
    for i in range(num_episodes):
        T = np.random.randint(100, 400)
        ep = generate_episode(T=T)
        fpath = os.path.join(src_dir, f"episode_{i:04d}.pkl")
        write_pkl(fpath, ep)

    print(f"  源目录: {src_dir} ({num_episodes} 个 PKL 文件)\n")

    converter = DatasetConverter()

    # --- PKL → HDF5 ---
    print("  批量转换: PKL → HDF5")
    dst_h5 = os.path.join(work_dir, "hdf5_episodes")
    stats = converter.batch_convert(src_dir, dst_h5, target_format="hdf5")

    total_in = sum(s["input_size"] for s in stats)
    total_out = sum(s["output_size"] for s in stats)
    print(f"    总输入: {total_in/1024:.1f} KB → 总输出: {total_out/1024:.1f} KB")
    print(f"    压缩比: {total_out/total_in:.2%}\n")

    # --- PKL → NPZ ---
    print("  批量转换: PKL → NPZ")
    dst_npz = os.path.join(work_dir, "npz_episodes")
    stats = converter.batch_convert(src_dir, dst_npz, target_format="npz")

    total_in = sum(s["input_size"] for s in stats)
    total_out = sum(s["output_size"] for s in stats)
    print(f"    总输入: {total_in/1024:.1f} KB → 总输出: {total_out/1024:.1f} KB")

    # 验证所有转换结果
    print("\n  验证所有转换结果:")
    all_ok = True
    for i in range(num_episodes):
        src = os.path.join(src_dir, f"episode_{i:04d}.pkl")
        dst = os.path.join(dst_h5, f"episode_{i:04d}.hdf5")
        if os.path.exists(dst):
            ok = converter.verify_conversion(src, dst)
            if not ok:
                all_ok = False
    print(f"    {'✓ 全部通过' if all_ok else '✗ 存在差异'}")


# ============================================================
# 第 4 节：元数据保留策略
# ============================================================

def section_4_metadata_preservation(work_dir):
    print(DIVIDER)
    print("第 4 节：元数据保留策略")
    print(DIVIDER)

    print("""
    不同格式对元数据的支持程度不同:

    ┌──────────── 元数据支持对比 ────────────┐
    │  格式    层级结构  属性/元数据  类型保真  │
    │  HDF5      ✓         ✓           ✓      │
    │  PKL       ✓         ✓           ✓      │
    │  NPZ       ✗         △           △      │
    │  (△ = 通过编码方式间接支持)              │
    └──────────────────────────────────────────┘

    NPZ 的元数据编码方案:
      1. 把嵌套 dict 的 key 用 "." 连接展平
         observations.qpos → ndarray
      2. 字符串值加 "__str__" 前缀
      3. 标量值转为 0-d 数组
    """)

    # 演示：带丰富元数据的 episode
    episode = generate_episode(T=100)
    episode["metadata"]["calibration_date"] = "2025-01-15"
    episode["metadata"]["operator"] = "zhang_san"
    episode["metadata"]["gripper_force_limit"] = 10.5

    converter = DatasetConverter()

    # HDF5 → NPZ → HDF5 往返
    h5_orig = os.path.join(work_dir, "meta_orig.hdf5")
    npz_mid = os.path.join(work_dir, "meta_mid.npz")
    h5_back = os.path.join(work_dir, "meta_back.hdf5")

    write_hdf5(h5_orig, episode)
    converter.convert(h5_orig, npz_mid, verbose=False)
    converter.convert(npz_mid, h5_back, verbose=False)

    # 比较元数据
    print("  元数据往返测试 (HDF5 → NPZ → HDF5):\n")
    orig_data = read_hdf5(h5_orig)
    back_data = read_hdf5(h5_back)

    orig_meta = orig_data.get("metadata", {})
    back_meta = back_data.get("metadata", {})

    for key in orig_meta:
        orig_val = orig_meta[key]
        back_val = back_meta.get(key, "<缺失>")
        match = "✓" if str(orig_val) == str(back_val) else "✗"
        print(f"    {match} {key}: {orig_val} → {back_val}")


# ============================================================
# 第 5 节：性能对比
# ============================================================

def section_5_performance(work_dir):
    print(DIVIDER)
    print("第 5 节：读写性能全面对比")
    print(DIVIDER)

    sizes = [
        ("小 (T=100)",  100,  7),
        ("中 (T=1000)", 1000, 14),
        ("大 (T=5000)", 5000, 14),
    ]

    converter = DatasetConverter()

    for size_name, T, nq in sizes:
        episode = generate_episode(T=T, nq=nq, nv=nq, nu=nq)
        raw_bytes = T * nq * 8 * 3  # qpos + qvel + action

        print(f"\n  数据规模: {size_name}, 原始大小 ≈ {raw_bytes/1024:.0f} KB")
        print(f"  {'格式':<15} {'写入':>10} {'读取':>10} {'文件大小':>10}")
        print(f"  {'-' * 50}")

        trials = 3

        for fmt, ext in [("hdf5", ".hdf5"), ("pkl", ".pkl"), ("npz", ".npz")]:
            fpath = os.path.join(work_dir, f"perf_{fmt}{ext}")

            writer = converter.WRITERS[fmt]
            reader = converter.READERS[fmt]

            # 写入基准
            t0 = time.perf_counter()
            for _ in range(trials):
                writer(fpath, episode)
            w_time = (time.perf_counter() - t0) / trials

            # 读取基准
            t0 = time.perf_counter()
            for _ in range(trials):
                reader(fpath)
            r_time = (time.perf_counter() - t0) / trials

            size = file_size_str(fpath)
            print(f"  {fmt:<15} {w_time*1000:>8.1f}ms {r_time*1000:>8.1f}ms {size:>10}")

    print("""
    总结:
      - PKL 读写最快（几乎零开销的内存序列化）
      - NPZ 压缩版文件最小（zip 压缩）
      - HDF5 是速度和功能的最佳平衡（压缩 + 切片读取）
      - 数据量越大，HDF5 的压缩优势越明显
    """)


# ============================================================
# 第 6 节：转换 pipeline 实战
# ============================================================

def section_6_pipeline(work_dir):
    print(DIVIDER)
    print("第 6 节：转换 Pipeline 实战")
    print(DIVIDER)

    print("""
    实际场景：数据采集系统产出 PKL 文件 → 需要转换为标准 HDF5 格式。

    Pipeline 步骤:
      1. 扫描源目录
      2. 验证每个文件的数据完整性
      3. 转换格式
      4. 验证转换结果
      5. 生成转换报告
    """)

    # 模拟数据采集产出
    raw_dir = os.path.join(work_dir, "raw_data")
    os.makedirs(raw_dir, exist_ok=True)
    num_episodes = 6

    for i in range(num_episodes):
        T = np.random.randint(100, 500)
        ep = generate_episode(T=T)
        fpath = os.path.join(raw_dir, f"episode_{i:04d}.pkl")
        write_pkl(fpath, ep)

    # 运行 pipeline
    print(f"  源目录: {raw_dir}")
    print(f"  episode 数量: {num_episodes}\n")

    converter = DatasetConverter()
    output_dir = os.path.join(work_dir, "standardized")
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "total_files": 0,
        "success": 0,
        "failed": 0,
        "total_input_bytes": 0,
        "total_output_bytes": 0,
        "total_timesteps": 0,
    }

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".pkl"))

    for fname in files:
        src = os.path.join(raw_dir, fname)
        dst = os.path.join(output_dir, fname.replace(".pkl", ".hdf5"))
        report["total_files"] += 1

        try:
            # 读取并验证
            data = read_pkl(src)
            assert "observations" in data, "缺少 observations"
            assert "action" in data, "缺少 action"
            T = data["action"].shape[0]

            # 转换
            stats = converter.convert(src, dst, verbose=False)

            # 验证
            ok = converter.verify_conversion(src, dst)
            if ok:
                report["success"] += 1
                report["total_timesteps"] += T
            else:
                report["failed"] += 1

            report["total_input_bytes"] += stats["input_size"]
            report["total_output_bytes"] += stats["output_size"]

            status = "✓" if ok else "✗"
            print(f"    {status} {fname} → {os.path.basename(dst)} (T={T})")

        except Exception as e:
            report["failed"] += 1
            print(f"    ✗ {fname}: {e}")

    # 转换报告
    print(f"\n  ────── 转换报告 ──────")
    print(f"  总文件数:  {report['total_files']}")
    print(f"  成功:      {report['success']}")
    print(f"  失败:      {report['failed']}")
    print(f"  总时间步:  {report['total_timesteps']}")
    print(f"  输入总量:  {report['total_input_bytes']/1024:.1f} KB")
    print(f"  输出总量:  {report['total_output_bytes']/1024:.1f} KB")
    if report['total_input_bytes'] > 0:
        ratio = report['total_output_bytes'] / report['total_input_bytes']
        print(f"  压缩比:    {ratio:.2%}")


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("  第 5 章 · 03 — 数据格式互转")
    print(DIVIDER)

    if not HAS_H5PY:
        print("\n  ⚠️  h5py 未安装，部分功能不可用")
        print("  请运行: pip install h5py\n")
        return

    work_dir = tempfile.mkdtemp(prefix="ch05_convert_")
    print(f"\n  工作目录: {work_dir}\n")

    try:
        section_1_basic_conversion(work_dir)
        section_2_converter_class(work_dir)
        section_3_batch_conversion(work_dir)
        section_4_metadata_preservation(work_dir)
        section_5_performance(work_dir)
        section_6_pipeline(work_dir)

        print(DIVIDER)
        print("  所有格式转换示例运行完毕！")
        print(DIVIDER)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"\n  已清理工作目录: {work_dir}")


if __name__ == "__main__":
    main()
