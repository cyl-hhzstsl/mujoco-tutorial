"""
第 5 章 · 02 - PKL 与 NPZ 格式详解

目标: 掌握 pickle 和 NumPy 原生格式在机器人数据场景中的用法，
     理解各自的优劣、安全风险、以及与 HDF5 的对比。

核心知识点:
  1. pickle.dump / pickle.load —— 任意 Python 对象的序列化
  2. 安全警告 —— 为什么不应该加载不信任来源的 pkl 文件
  3. np.save / np.load / np.savez / np.savez_compressed
  4. 格式对比 —— 文件大小、读写速度、灵活性
  5. 格式选择指南

运行: python 02_pkl_and_npz.py
依赖: pip install numpy
"""

import numpy as np
import pickle
import os
import time
import tempfile
import shutil
import sys

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


def generate_trajectory(T=200, nq=7, nv=7, nu=7):
    """生成模拟轨迹数据"""
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
# 第 1 节：Pickle 基础
# ============================================================

def section_1_pickle_basics(work_dir):
    print(DIVIDER)
    print("第 1 节：Pickle —— Python 对象的万能序列化")
    print(DIVIDER)

    print("""
    pickle 是 Python 内置的序列化模块，可以把几乎任意 Python 对象
    保存到磁盘，再原样恢复。

    Java 类比:
      pickle ≈ Java Serializable + ObjectOutputStream
      但比 Java 序列化灵活得多（不需要实现特定接口）

    优势:
      ✓ 几乎支持任意 Python 对象（dict, list, class instance, ...）
      ✓ 使用极其简单：dump / load 两个函数搞定
      ✓ 快速原型开发时非常方便

    劣势:
      ✗ 不安全（恶意 pkl 文件可以执行任意代码！）
      ✗ 仅限 Python（不跨语言）
      ✗ 版本敏感（不同 Python 版本可能不兼容）
      ✗ 没有切片读取（必须加载整个文件到内存）
    """)

    traj = generate_trajectory(T=200)

    # --- 基本用法: dump 和 load ---
    fpath = os.path.join(work_dir, "trajectory.pkl")

    # 写入
    with open(fpath, "wb") as f:
        pickle.dump(traj, f)
    print(f"  pickle.dump → {file_size_str(fpath)}")

    # 读取
    with open(fpath, "rb") as f:
        loaded = pickle.load(f)

    # 验证
    assert np.allclose(traj["observations"]["qpos"], loaded["observations"]["qpos"])
    assert loaded["metadata"]["robot_type"] == "franka_panda"
    print("  pickle.load → 验证通过 ✓")

    # --- 不同 protocol 版本 ---
    print(f"\n  Pickle Protocol 版本对比:")
    print(f"  {'Protocol':>10} {'文件大小':>10} {'写入耗时':>10}")
    print(f"  {'-' * 35}")

    for protocol in range(3, pickle.HIGHEST_PROTOCOL + 1):
        fp = os.path.join(work_dir, f"traj_proto{protocol}.pkl")
        t0 = time.perf_counter()
        with open(fp, "wb") as f:
            pickle.dump(traj, f, protocol=protocol)
        elapsed = time.perf_counter() - t0
        print(f"  {protocol:>10} {file_size_str(fp):>10} {elapsed*1000:>8.1f}ms")

    print(f"\n  当前默认 protocol: {pickle.DEFAULT_PROTOCOL}")
    print(f"  最高可用 protocol: {pickle.HIGHEST_PROTOCOL}")
    print("  建议：使用 pickle.HIGHEST_PROTOCOL 以获得最佳性能")

    return fpath


# ============================================================
# 第 2 节：Pickle 安全警告
# ============================================================

def section_2_security_warning():
    print(DIVIDER)
    print("第 2 节：⚠️  Pickle 安全警告 ⚠️")
    print(DIVIDER)

    print("""
    ┌─────────────────── 严重安全风险 ───────────────────┐
    │                                                     │
    │  pickle.load() 可以执行任意代码！                    │
    │                                                     │
    │  恶意 .pkl 文件可以:                                 │
    │    - 删除你的文件                                    │
    │    - 安装后门                                        │
    │    - 窃取密钥/密码                                   │
    │    - 下载并运行恶意软件                               │
    │                                                     │
    │  原理: pickle 反序列化时会调用 __reduce__ 方法，     │
    │  攻击者可以构造恶意对象让 __reduce__ 执行任意命令。  │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    安全准则:
      1. 绝对不要加载来自不信任来源的 .pkl 文件
      2. 从网上下载的数据集，如果是 .pkl 格式，要格外小心
      3. 团队内部传输 .pkl 文件时，确保来源可信
      4. 优先使用 HDF5/NPZ 等更安全的格式分享数据
      5. 如果必须用 pkl，考虑用 fickling 库扫描恶意内容
    """)

    # 演示恶意 pkl 的原理（仅展示机制，不实际执行危险操作）
    print("  演示: pickle 如何执行任意代码（安全的示例）\n")

    class SafeDemo:
        """演示 __reduce__ 机制（这里只是打印一条消息）"""
        def __reduce__(self):
            return (print, ("  [!] 这条消息是 pickle.load 时执行的 print() 产生的！",))

    demo_bytes = pickle.dumps(SafeDemo())
    print("  调用 pickle.loads() ...")
    pickle.loads(demo_bytes)  # 会打印上面的消息
    print("  ↑ 上面的消息证明了 pickle.load 可以执行函数调用\n")

    print("  如果 __reduce__ 返回 (os.system, ('rm -rf /',)) 就会删除整个磁盘。")
    print("  所以：永远不要 pickle.load 不信任的文件！")


# ============================================================
# 第 3 节：NumPy 的 .npy 和 .npz 格式
# ============================================================

def section_3_numpy_formats(work_dir):
    print(DIVIDER)
    print("第 3 节：NumPy 的 .npy 和 .npz 格式")
    print(DIVIDER)

    print("""
    NumPy 提供了自己的序列化格式:

    .npy  —— 单个数组
      np.save("data.npy", array)
      array = np.load("data.npy")

    .npz  —— 多个数组的归档（类似 zip 打包多个 .npy）
      np.savez("data.npz", qpos=qpos, qvel=qvel)
      data = np.load("data.npz")
      qpos = data["qpos"]

    .npz (compressed) —— 压缩版
      np.savez_compressed("data.npz", qpos=qpos, qvel=qvel)

    优势:
      ✓ 轻量、快速
      ✓ NumPy 原生，无需额外依赖
      ✓ 比 pkl 更安全（只能存数组，不能执行代码）

    劣势:
      ✗ 只能存 NumPy 数组（不能存嵌套 dict 等结构）
      ✗ 没有层级结构（只有扁平的 key-value）
      ✗ 没有元数据支持
      ✗ 不支持切片读取（默认全部加载到内存）
    """)

    T, nq = 500, 7
    qpos = np.random.randn(T, nq).astype(np.float64)
    qvel = np.random.randn(T, nq).astype(np.float64)
    action = np.random.randn(T, nq).astype(np.float64)

    # --- .npy: 单个数组 ---
    npy_path = os.path.join(work_dir, "qpos.npy")
    np.save(npy_path, qpos)
    loaded_qpos = np.load(npy_path)
    assert np.allclose(qpos, loaded_qpos)
    print(f"  np.save('qpos.npy')        → {file_size_str(npy_path)}")

    # --- .npz: 多个数组（未压缩） ---
    npz_path = os.path.join(work_dir, "trajectory.npz")
    np.savez(npz_path, qpos=qpos, qvel=qvel, action=action)
    print(f"  np.savez('traj.npz')       → {file_size_str(npz_path)}")

    # --- .npz: 多个数组（压缩） ---
    npzc_path = os.path.join(work_dir, "trajectory_compressed.npz")
    np.savez_compressed(npzc_path, qpos=qpos, qvel=qvel, action=action)
    print(f"  np.savez_compressed(...)   → {file_size_str(npzc_path)}")

    # --- 读取 .npz ---
    print(f"\n  读取 .npz 文件:")
    data = np.load(npz_path)
    print(f"    可用的 key: {list(data.keys())}")
    print(f"    data['qpos'].shape = {data['qpos'].shape}")
    data.close()

    # --- 懒加载特性 ---
    print(f"\n  .npz 的懒加载（mmap_mode）:")
    data = np.load(npz_path, allow_pickle=False)
    print(f"    np.load 返回 NpzFile 对象，数据按需加载")
    print(f"    访问 data['qpos'] 时才真正读取磁盘")
    data.close()

    # --- .npy 的 mmap 模式 ---
    print(f"\n  .npy 支持 memory-mapped 读取:")
    mmap_data = np.load(npy_path, mmap_mode="r")
    print(f"    mmap_mode='r' → 数据不全部加载到内存")
    print(f"    mmap_data[0:10].shape = {mmap_data[0:10].shape} (按需从磁盘读取)")

    return npz_path


# ============================================================
# 第 4 节：存储多集轨迹数据
# ============================================================

def section_4_storing_episodes(work_dir):
    print(DIVIDER)
    print("第 4 节：用 PKL 和 NPZ 存储多集轨迹")
    print(DIVIDER)

    trajectories = []
    for ep in range(3):
        T = np.random.randint(100, 300)
        traj = generate_trajectory(T=T)
        trajectories.append(traj)
        print(f"  Episode {ep}: T={T}")

    # --- 方式 1: 整体存为一个 pkl ---
    pkl_all = os.path.join(work_dir, "all_episodes.pkl")
    with open(pkl_all, "wb") as f:
        pickle.dump(trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  方式 1 (一个 pkl 存所有 episode): {file_size_str(pkl_all)}")
    print("    优点: 简单直接")
    print("    缺点: 必须全部加载到内存，无法按 episode 随机读取")

    # --- 方式 2: 每集一个 pkl ---
    ep_dir = os.path.join(work_dir, "episodes_pkl")
    os.makedirs(ep_dir, exist_ok=True)
    for i, traj in enumerate(trajectories):
        fpath = os.path.join(ep_dir, f"episode_{i:04d}.pkl")
        with open(fpath, "wb") as f:
            pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)
    total = sum(os.path.getsize(os.path.join(ep_dir, f)) for f in os.listdir(ep_dir))
    print(f"\n  方式 2 (每集一个 pkl): 总大小 {total/1024:.1f} KB")
    print("    优点: 可以按需加载单集")
    print("    缺点: 文件数量多，管理复杂")

    # --- 方式 3: 每集的数组用 npz 存 ---
    ep_dir_npz = os.path.join(work_dir, "episodes_npz")
    os.makedirs(ep_dir_npz, exist_ok=True)
    for i, traj in enumerate(trajectories):
        fpath = os.path.join(ep_dir_npz, f"episode_{i:04d}.npz")
        np.savez_compressed(
            fpath,
            qpos=traj["observations"]["qpos"],
            qvel=traj["observations"]["qvel"],
            action=traj["action"],
        )
    total = sum(os.path.getsize(os.path.join(ep_dir_npz, f)) for f in os.listdir(ep_dir_npz))
    print(f"\n  方式 3 (每集一个 npz): 总大小 {total/1024:.1f} KB")
    print("    优点: 安全、高效、跨版本兼容")
    print("    缺点: 不能存元数据（需要额外的 metadata 文件）")


# ============================================================
# 第 5 节：格式对比 —— 文件大小与读写速度
# ============================================================

def section_5_format_comparison(work_dir):
    print(DIVIDER)
    print("第 5 节：PKL vs NPZ vs NPZ_compressed 性能对比")
    print(DIVIDER)

    T = 1000
    nq = 14
    traj = generate_trajectory(T=T, nq=nq, nv=nq, nu=nq)

    formats = {
        "pkl": {
            "ext": ".pkl",
            "write": lambda p, d: pickle.dump(d, open(p, "wb"), protocol=pickle.HIGHEST_PROTOCOL),
            "read": lambda p: pickle.load(open(p, "rb")),
        },
        "npz": {
            "ext": ".npz",
            "write": lambda p, d: np.savez(p, qpos=d["observations"]["qpos"],
                                           qvel=d["observations"]["qvel"],
                                           action=d["action"]),
            "read": lambda p: dict(np.load(p)),
        },
        "npz_compressed": {
            "ext": ".npz",
            "write": lambda p, d: np.savez_compressed(p, qpos=d["observations"]["qpos"],
                                                      qvel=d["observations"]["qvel"],
                                                      action=d["action"]),
            "read": lambda p: dict(np.load(p)),
        },
    }

    # 加入 HDF5 对比
    try:
        import h5py
        def write_hdf5(path, data):
            with h5py.File(path, "w") as f:
                obs = f.create_group("observations")
                obs.create_dataset("qpos", data=data["observations"]["qpos"], compression="gzip")
                obs.create_dataset("qvel", data=data["observations"]["qvel"], compression="gzip")
                f.create_dataset("action", data=data["action"], compression="gzip")

        def read_hdf5(path):
            with h5py.File(path, "r") as f:
                return {
                    "qpos": f["observations/qpos"][:],
                    "qvel": f["observations/qvel"][:],
                    "action": f["action"][:],
                }

        formats["hdf5_gzip"] = {
            "ext": ".hdf5",
            "write": write_hdf5,
            "read": read_hdf5,
        }
    except ImportError:
        pass

    print(f"  数据规模: T={T}, nq={nq} (约 {T*nq*8*3/1024:.0f} KB 原始数据)")
    print(f"\n  {'格式':<20} {'文件大小':>10} {'写入耗时':>10} {'读取耗时':>10}")
    print(f"  {'-' * 55}")

    num_trials = 5
    for name, fmt in formats.items():
        fpath = os.path.join(work_dir, f"bench_{name}{fmt['ext']}")

        # 写入基准
        t0 = time.perf_counter()
        for _ in range(num_trials):
            fmt["write"](fpath, traj)
        write_time = (time.perf_counter() - t0) / num_trials

        # 读取基准
        t0 = time.perf_counter()
        for _ in range(num_trials):
            fmt["read"](fpath)
        read_time = (time.perf_counter() - t0) / num_trials

        size = file_size_str(fpath)
        print(f"  {name:<20} {size:>10} {write_time*1000:>8.1f}ms {read_time*1000:>8.1f}ms")

    print("""
    ┌──────────────────── 格式选择指南 ────────────────────┐
    │                                                       │
    │  场景                          推荐格式                │
    │  ────                          ────────                │
    │  快速原型 / 临时缓存           pkl                     │
    │  纯数组小规模实验               npz_compressed          │
    │  正式数据集 / 需要元数据        HDF5                    │
    │  需要跨语言访问                 HDF5                    │
    │  需要切片读取大文件             HDF5                    │
    │  分享给他人                     HDF5 或 npz（更安全）   │
    │  包含复杂 Python 对象           pkl（注意安全风险）     │
    │                                                       │
    └───────────────────────────────────────────────────────┘
    """)


# ============================================================
# 第 6 节：常见陷阱与最佳实践
# ============================================================

def section_6_pitfalls():
    print(DIVIDER)
    print("第 6 节：常见陷阱与最佳实践")
    print(DIVIDER)

    print("""
    1. pkl 版本兼容性问题
       - Python 3.8 的 pkl 在 3.12 中可能无法加载
       - 如果使用了 protocol=5，低版本 Python 无法读取
       - 建议：长期存储不要用 pkl

    2. npz 的 allow_pickle 参数
       - np.load 默认 allow_pickle=False（安全）
       - 如果 npz 中包含对象数组，需要 allow_pickle=True
       - 建议：避免在 npz 中存储对象数组

    3. pkl 文件大小膨胀
       - pickle 的开销对小数据很可观
       - 存 1000 个小 dict，pkl 比 json 还大
       - 建议：大批量小对象用 json 或 parquet

    4. 内存管理
       - pickle.load 会把整个对象加载到内存
       - 大数据集（>1GB）可能导致 OOM
       - 建议：大数据集用 HDF5 + 切片读取

    5. 文件描述符泄漏
    """)

    # 演示: npz 文件需要手动关闭
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    np.savez(tmp_path, x=np.zeros(10))

    data = np.load(tmp_path)
    print(f"    NpzFile 类型: {type(data)}")
    data.close()  # 记得关闭！
    print("    ✓ 使用后记得 data.close() 或用 with 语句")
    os.unlink(tmp_path)

    print("""
    6. 序列化自定义类
       - pkl 可以序列化自定义类的实例
       - 但加载时需要该类的定义在作用域内
       - 如果类定义改了，旧 pkl 可能无法加载
       - 建议：只序列化基础类型 (dict, list, ndarray)
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    print(DIVIDER)
    print("  第 5 章 · 02 — PKL 与 NPZ 格式详解")
    print(DIVIDER)

    work_dir = tempfile.mkdtemp(prefix="ch05_pkl_npz_")
    print(f"\n  工作目录: {work_dir}\n")

    try:
        section_1_pickle_basics(work_dir)
        section_2_security_warning()
        section_3_numpy_formats(work_dir)
        section_4_storing_episodes(work_dir)
        section_5_format_comparison(work_dir)
        section_6_pitfalls()

        print(DIVIDER)
        print("  所有 PKL / NPZ 示例运行完毕！")
        print(DIVIDER)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"\n  已清理工作目录: {work_dir}")


if __name__ == "__main__":
    main()
