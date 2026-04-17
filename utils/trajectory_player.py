"""
轨迹回放器 (Trajectory Player)

功能: 加载各种格式的轨迹数据，支持回放（有/无 viewer）、
     帧查询、轨迹统计等功能。

支持格式: .pkl, .npy, .npz, .h5/.hdf5

使用:
  player = TrajectoryPlayer()
  player.load("episode.h5")
  player.play(speed=1.0)              # 在 viewer 中回放
  player.play_headless()              # 无 viewer 迭代
  frame = player.get_frame(0.5)       # 获取 t=0.5s 的 qpos
  stats = player.get_statistics()     # 轨迹统计信息
"""

import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union


@dataclass
class TrajectoryData:
    """轨迹数据的标准内部表示。"""
    qpos: np.ndarray                         # (T, nq)
    qvel: Optional[np.ndarray] = None        # (T, nv)
    ctrl: Optional[np.ndarray] = None        # (T, nu)
    timestamps: Optional[np.ndarray] = None  # (T,)
    dt: float = 0.001
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.qpos)

    @property
    def duration(self) -> float:
        """轨迹总时长（秒）。"""
        if self.timestamps is not None and len(self.timestamps) > 1:
            return float(self.timestamps[-1] - self.timestamps[0])
        return self.n_frames * self.dt

    @property
    def nq(self) -> int:
        return self.qpos.shape[1] if self.qpos.ndim > 1 else 1


class TrajectoryPlayer:
    """
    轨迹回放器：加载、查询、回放机器人轨迹数据。

    支持的文件格式:
      - .pkl: pickle 序列化的字典
      - .npy: 单个 numpy 数组（视为 qpos）
      - .npz: 多数组压缩包
      - .h5 / .hdf5: HDF5 分层数据

    回放模式:
      - play(speed): 使用 MuJoCo viewer 可视化回放
      - play_headless(): 无 viewer 逐帧迭代（用于数据处理管道）

    查询接口:
      - get_frame(t): 按时间查询 qpos（支持插值）
      - get_statistics(): 轨迹统计信息
    """

    def __init__(self):
        self._data: Optional[TrajectoryData] = None
        self._source_path: Optional[str] = None

    # ================================================================
    # 加载
    # ================================================================

    def load(self, filepath: Union[str, Path]) -> "TrajectoryPlayer":
        """
        加载轨迹文件，自动检测格式。

        参数:
          filepath: 轨迹文件路径（支持 pkl/npy/npz/h5/hdf5）

        返回:
          self（链式调用）
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        suffix = filepath.suffix.lower()
        loaders = {
            ".pkl": self._load_pkl,
            ".npy": self._load_npy,
            ".npz": self._load_npz,
            ".h5": self._load_hdf5,
            ".hdf5": self._load_hdf5,
        }

        loader = loaders.get(suffix)
        if loader is None:
            raise ValueError(
                f"不支持的格式: {suffix}  "
                f"(支持: {', '.join(loaders.keys())})"
            )

        self._data = loader(filepath)
        self._source_path = str(filepath)
        return self

    def load_from_arrays(self,
                         qpos: np.ndarray,
                         qvel: Optional[np.ndarray] = None,
                         ctrl: Optional[np.ndarray] = None,
                         dt: float = 0.001,
                         timestamps: Optional[np.ndarray] = None) -> "TrajectoryPlayer":
        """直接从 numpy 数组加载。"""
        self._data = TrajectoryData(
            qpos=np.atleast_2d(qpos),
            qvel=np.atleast_2d(qvel) if qvel is not None else None,
            ctrl=np.atleast_2d(ctrl) if ctrl is not None else None,
            timestamps=timestamps,
            dt=dt,
        )
        self._source_path = "<arrays>"
        return self

    # ---- 格式特化加载器 ----

    def _load_pkl(self, filepath: Path) -> TrajectoryData:
        """加载 pickle 文件。"""
        with open(filepath, "rb") as f:
            raw = pickle.load(f)

        if isinstance(raw, dict):
            return self._dict_to_trajectory(raw)
        elif isinstance(raw, np.ndarray):
            return TrajectoryData(qpos=np.atleast_2d(raw))
        elif isinstance(raw, list):
            arr = np.array(raw)
            return TrajectoryData(qpos=np.atleast_2d(arr))
        else:
            raise ValueError(f"无法解析 pkl 内容: type={type(raw)}")

    def _load_npy(self, filepath: Path) -> TrajectoryData:
        """加载 .npy 文件（单数组，视为 qpos）。"""
        arr = np.load(filepath)
        return TrajectoryData(qpos=np.atleast_2d(arr))

    def _load_npz(self, filepath: Path) -> TrajectoryData:
        """加载 .npz 压缩包。"""
        npz = np.load(filepath, allow_pickle=True)
        return self._dict_to_trajectory(dict(npz))

    def _load_hdf5(self, filepath: Path) -> TrajectoryData:
        """加载 HDF5 文件。"""
        try:
            import h5py
        except ImportError:
            raise ImportError("加载 HDF5 需要 h5py: pip install h5py")

        raw: Dict[str, Any] = {}
        with h5py.File(filepath, "r") as f:
            self._h5_read_recursive(f, raw)

        return self._dict_to_trajectory(raw)

    @staticmethod
    def _h5_read_recursive(group, target: dict, prefix: str = "") -> None:
        """递归读取 HDF5 group 中的所有 dataset。"""
        import h5py
        for key in group.keys():
            full_key = f"{prefix}/{key}" if prefix else key
            item = group[key]
            if isinstance(item, h5py.Dataset):
                target[full_key] = item[:]
            elif isinstance(item, h5py.Group):
                # 也在顶层键上放一份，方便后续匹配
                TrajectoryPlayer._h5_read_recursive(item, target, full_key)
                if "data" in item:
                    target[key] = item["data"][:]

    def _dict_to_trajectory(self, d: Dict[str, Any]) -> TrajectoryData:
        """将字典统一转换为 TrajectoryData。"""
        qpos = self._find_array(d, ["qpos", "joint_positions", "positions", "q"])
        if qpos is None:
            for v in d.values():
                if isinstance(v, np.ndarray) and v.ndim >= 1:
                    qpos = v
                    break
        if qpos is None:
            raise ValueError("数据中未找到 qpos 或类似字段")

        qpos = np.atleast_2d(qpos)

        qvel = self._find_array(d, ["qvel", "joint_velocities", "velocities", "qdot"])
        ctrl = self._find_array(d, ["ctrl", "control", "actions", "torques"])
        timestamps = self._find_array(d, ["timestamps", "time", "t"])

        dt = 0.001
        if "dt" in d:
            dt_val = d["dt"]
            dt = float(dt_val.item()) if isinstance(dt_val, np.ndarray) else float(dt_val)
        elif "sim_dt" in d:
            sim_dt_val = d["sim_dt"]
            dt = float(sim_dt_val.item()) if isinstance(sim_dt_val, np.ndarray) else float(sim_dt_val)

        metadata = {}
        for k, v in d.items():
            if k not in ("qpos", "qvel", "ctrl", "timestamps", "time", "dt", "sim_dt"):
                if isinstance(v, np.ndarray) and v.size <= 10:
                    metadata[k] = v.tolist()
                elif not isinstance(v, np.ndarray):
                    metadata[k] = v

        return TrajectoryData(
            qpos=qpos,
            qvel=np.atleast_2d(qvel) if qvel is not None else None,
            ctrl=np.atleast_2d(ctrl) if ctrl is not None else None,
            timestamps=timestamps,
            dt=dt,
            metadata=metadata,
        )

    @staticmethod
    def _find_array(d: Dict[str, Any],
                    candidates: List[str]) -> Optional[np.ndarray]:
        """在字典中查找第一个匹配的 numpy 数组。"""
        for key in candidates:
            if key in d and isinstance(d[key], np.ndarray):
                return d[key]
        return None

    # ================================================================
    # 回放
    # ================================================================

    def play(self, speed: float = 1.0,
             model_xml: Optional[str] = None) -> None:
        """
        使用 MuJoCo viewer 回放轨迹。

        参数:
          speed: 回放速度倍率（1.0 = 原速，2.0 = 两倍速）
          model_xml: 可选的 MJCF 模型（用于可视化）

        如果 viewer 不可用，自动降级为 play_headless()。
        """
        self._ensure_loaded()
        data = self._data

        import mujoco

        if model_xml is None:
            model_xml = self._generate_minimal_model(data.nq)

        model = mujoco.MjModel.from_xml_string(model_xml)
        mj_data = mujoco.MjData(model)

        try:
            import mujoco.viewer
            viewer_available = True
        except ImportError:
            viewer_available = False

        if not viewer_available:
            print("  viewer 不可用，降级为 headless 模式")
            self.play_headless(speed=speed)
            return

        import time

        print(f"  开始回放: {data.n_frames} 帧, "
              f"时长 {data.duration:.2f}s, 速度 {speed}x")
        print("  (关闭 viewer 窗口结束回放)")

        try:
            with mujoco.viewer.launch_passive(model, mj_data) as viewer:
                frame_dt = data.dt / speed
                for i in range(data.n_frames):
                    if not viewer.is_running():
                        break

                    nq = min(model.nq, data.qpos.shape[1])
                    mj_data.qpos[:nq] = data.qpos[i, :nq]
                    if data.qvel is not None:
                        nv = min(model.nv, data.qvel.shape[1])
                        mj_data.qvel[:nv] = data.qvel[i, :nv]

                    mujoco.mj_forward(model, mj_data)
                    viewer.sync()
                    time.sleep(frame_dt)

            print(f"  回放完成 ({i + 1}/{data.n_frames} 帧)")
        except Exception as e:
            print(f"  viewer 回放出错 ({e})，降级为 headless")
            self.play_headless(speed=speed)

    def play_headless(self, speed: float = 1.0,
                      callback=None) -> List[Dict[str, Any]]:
        """
        无 viewer 回放（逐帧迭代），适用于数据处理管道。

        参数:
          speed: 回放速度（影响报告中的时间戳，不影响实际执行速度）
          callback: 可选的每帧回调 fn(frame_idx, qpos, t) -> None

        返回:
          每帧的摘要列表
        """
        self._ensure_loaded()
        data = self._data

        print(f"  Headless 回放: {data.n_frames} 帧, "
              f"时长 {data.duration:.2f}s")

        summaries: List[Dict[str, Any]] = []
        report_interval = max(1, data.n_frames // 5)

        for i in range(data.n_frames):
            t = (data.timestamps[i] if data.timestamps is not None
                 else i * data.dt)

            frame_info = {
                "frame": i,
                "time": float(t),
                "qpos": data.qpos[i].copy(),
            }
            if data.qvel is not None:
                frame_info["qvel"] = data.qvel[i].copy()
            if data.ctrl is not None:
                frame_info["ctrl"] = data.ctrl[i].copy()

            summaries.append(frame_info)

            if callback is not None:
                callback(i, data.qpos[i], t)

            if (i + 1) % report_interval == 0:
                print(f"    帧 {i + 1}/{data.n_frames} "
                      f"(t={t:.3f}s)")

        print(f"  Headless 回放完成: {data.n_frames} 帧")
        return summaries

    # ================================================================
    # 帧查询
    # ================================================================

    def get_frame(self, t: float,
                  interpolate: bool = True) -> np.ndarray:
        """
        获取时间 t 处的 qpos（支持线性插值）。

        参数:
          t: 查询时间（秒）
          interpolate: 是否在帧间线性插值

        返回:
          qpos 数组 (nq,)
        """
        self._ensure_loaded()
        data = self._data

        if data.timestamps is not None:
            times = data.timestamps
        else:
            times = np.arange(data.n_frames) * data.dt

        t = np.clip(t, times[0], times[-1])

        if not interpolate:
            idx = int(np.argmin(np.abs(times - t)))
            return data.qpos[idx].copy()

        # 线性插值
        idx = np.searchsorted(times, t, side="right") - 1
        idx = np.clip(idx, 0, data.n_frames - 2)

        t0, t1 = times[idx], times[idx + 1]
        dt_local = t1 - t0
        if dt_local < 1e-12:
            return data.qpos[idx].copy()

        alpha = (t - t0) / dt_local
        return (1 - alpha) * data.qpos[idx] + alpha * data.qpos[idx + 1]

    def get_frame_index(self, idx: int) -> Dict[str, np.ndarray]:
        """按帧索引获取完整数据。"""
        self._ensure_loaded()
        data = self._data

        if idx < 0:
            idx = data.n_frames + idx
        if idx < 0 or idx >= data.n_frames:
            raise IndexError(f"帧索引越界: {idx} (总帧数: {data.n_frames})")

        result: Dict[str, np.ndarray] = {"qpos": data.qpos[idx].copy()}
        if data.qvel is not None:
            result["qvel"] = data.qvel[idx].copy()
        if data.ctrl is not None:
            result["ctrl"] = data.ctrl[idx].copy()
        if data.timestamps is not None:
            result["time"] = np.array(data.timestamps[idx])
        return result

    # ================================================================
    # 统计
    # ================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        计算轨迹的统计信息。

        返回内容:
          - 基本信息: 帧数、时长、关节数
          - qpos 统计: 均值、标准差、极值、范围
          - 运动统计: 帧间最大变化、平均速度
          - 数据完整性: NaN 数量、零帧比例
        """
        self._ensure_loaded()
        data = self._data
        qpos = data.qpos

        stats: Dict[str, Any] = {
            "source": self._source_path,
            "n_frames": data.n_frames,
            "nq": data.nq,
            "duration_s": data.duration,
            "dt": data.dt,
        }

        # qpos 统计
        stats["qpos"] = {
            "mean": qpos.mean(axis=0).tolist(),
            "std": qpos.std(axis=0).tolist(),
            "min": qpos.min(axis=0).tolist(),
            "max": qpos.max(axis=0).tolist(),
            "range": (qpos.max(axis=0) - qpos.min(axis=0)).tolist(),
        }

        # 帧间变化
        if data.n_frames > 1:
            diffs = np.diff(qpos, axis=0)
            stats["motion"] = {
                "max_frame_diff": float(np.abs(diffs).max()),
                "mean_frame_diff": float(np.abs(diffs).mean()),
                "max_abs_velocity_est": float(np.abs(diffs).max() / data.dt),
            }
        else:
            stats["motion"] = {
                "max_frame_diff": 0.0,
                "mean_frame_diff": 0.0,
                "max_abs_velocity_est": 0.0,
            }

        # qvel 统计（如果有）
        if data.qvel is not None:
            stats["qvel"] = {
                "mean": data.qvel.mean(axis=0).tolist(),
                "std": data.qvel.std(axis=0).tolist(),
                "max_abs": float(np.abs(data.qvel).max()),
            }

        # 数据完整性
        n_nan = int(np.isnan(qpos).sum())
        zero_frames = int((np.abs(qpos).sum(axis=1) == 0).sum())
        stats["integrity"] = {
            "nan_count": n_nan,
            "zero_frame_count": zero_frames,
            "zero_frame_ratio": zero_frames / max(data.n_frames, 1),
        }

        return stats

    # ================================================================
    # 内部工具
    # ================================================================

    def _ensure_loaded(self) -> None:
        if self._data is None:
            raise RuntimeError("未加载轨迹数据，请先调用 load() 或 load_from_arrays()")

    @staticmethod
    def _generate_minimal_model(nq: int) -> str:
        """生成与 qpos 维度匹配的最小 MJCF 模型。"""
        joints_xml = ""
        geoms_xml = ""
        actuators_xml = ""
        for i in range(nq):
            joints_xml += (
                f'        <joint name="j{i}" type="hinge" axis="0 1 0" '
                f'limited="true" range="-180 180"/>\n'
            )
            length = max(0.1, 0.3 - i * 0.03)
            geoms_xml += (
                f'        <geom type="capsule" '
                f'fromto="0 0 0 {length} 0 0" size="0.02"/>\n'
            )
            actuators_xml += (
                f'    <motor joint="j{i}" ctrlrange="-50 50"/>\n'
            )

        return f"""<mujoco model="trajectory_player">
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <worldbody>
    <body name="arm" pos="0 0 0">
{geoms_xml}{joints_xml}    </body>
  </worldbody>
  <actuator>
{actuators_xml}  </actuator>
</mujoco>"""

    @property
    def data(self) -> Optional[TrajectoryData]:
        """访问底层轨迹数据。"""
        return self._data

    @property
    def source(self) -> Optional[str]:
        return self._source_path

    def __repr__(self) -> str:
        if self._data is None:
            return "TrajectoryPlayer(empty)"
        d = self._data
        return (f"TrajectoryPlayer(frames={d.n_frames}, nq={d.nq}, "
                f"duration={d.duration:.2f}s, source={self._source_path})")


# ============================================================
# 独立运行时的演示
# ============================================================

def main():
    print("=" * 60)
    print("TrajectoryPlayer 演示")
    print("=" * 60)

    np.random.seed(42)

    # --- 1. 从数组直接加载 ---
    print("\n--- 1. 从 numpy 数组加载 ---")
    n_frames = 500
    dt = 0.002
    t = np.arange(n_frames) * dt
    qpos = np.column_stack([
        0.8 * np.sin(2 * np.pi * 0.5 * t),
        0.5 * np.cos(2 * np.pi * 0.3 * t),
    ])
    qvel = np.column_stack([
        0.8 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t),
        -0.5 * 2 * np.pi * 0.3 * np.sin(2 * np.pi * 0.3 * t),
    ])

    player = TrajectoryPlayer()
    player.load_from_arrays(qpos, qvel=qvel, dt=dt, timestamps=t)
    print(f"  {player}")

    # --- 2. 帧查询 ---
    print("\n--- 2. 帧查询 ---")
    test_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    for query_t in test_times:
        frame = player.get_frame(query_t)
        print(f"  t={query_t:.2f}s → qpos=[{frame[0]:>7.4f}, {frame[1]:>7.4f}]")

    # --- 3. 按索引查询 ---
    print("\n--- 3. 按索引查询 ---")
    for idx in [0, 100, -1]:
        frame_data = player.get_frame_index(idx)
        q = frame_data["qpos"]
        print(f"  frame[{idx:>4}] → qpos=[{q[0]:>7.4f}, {q[1]:>7.4f}]")

    # --- 4. 统计信息 ---
    print("\n--- 4. 轨迹统计 ---")
    stats = player.get_statistics()
    print(f"  帧数: {stats['n_frames']}")
    print(f"  时长: {stats['duration_s']:.3f}s")
    print(f"  关节数: {stats['nq']}")
    print(f"  qpos 均值: {[f'{v:.4f}' for v in stats['qpos']['mean']]}")
    print(f"  qpos 标准差: {[f'{v:.4f}' for v in stats['qpos']['std']]}")
    print(f"  qpos 范围: {[f'{v:.4f}' for v in stats['qpos']['range']]}")
    print(f"  最大帧间变化: {stats['motion']['max_frame_diff']:.6f}")
    print(f"  NaN 数量: {stats['integrity']['nan_count']}")
    print(f"  零帧比例: {stats['integrity']['zero_frame_ratio']:.2%}")

    # --- 5. 保存测试文件并重新加载 ---
    print("\n--- 5. 文件加载测试 ---")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # npz 格式
        npz_path = os.path.join(tmpdir, "test_traj.npz")
        np.savez(npz_path, qpos=qpos, qvel=qvel, dt=np.array(dt))
        player2 = TrajectoryPlayer().load(npz_path)
        print(f"  NPZ: {player2}")
        assert player2.data.n_frames == n_frames

        # npy 格式
        npy_path = os.path.join(tmpdir, "test_qpos.npy")
        np.save(npy_path, qpos)
        player3 = TrajectoryPlayer().load(npy_path)
        print(f"  NPY: {player3}")
        assert player3.data.n_frames == n_frames

        # pkl 格式
        pkl_path = os.path.join(tmpdir, "test_traj.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"qpos": qpos, "qvel": qvel, "dt": dt}, f)
        player4 = TrajectoryPlayer().load(pkl_path)
        print(f"  PKL: {player4}")
        assert player4.data.n_frames == n_frames

        # HDF5 格式（如果 h5py 可用）
        try:
            import h5py
            h5_path = os.path.join(tmpdir, "test_traj.h5")
            with h5py.File(h5_path, "w") as hf:
                hf.create_dataset("qpos", data=qpos)
                hf.create_dataset("qvel", data=qvel)
                hf.attrs["dt"] = dt
            player5 = TrajectoryPlayer().load(h5_path)
            print(f"  HDF5: {player5}")
            assert player5.data.n_frames == n_frames
        except ImportError:
            print("  HDF5: 跳过 (h5py 未安装)")

    # --- 6. Headless 回放 ---
    print("\n--- 6. Headless 回放 ---")
    short_player = TrajectoryPlayer().load_from_arrays(qpos[:50], dt=dt)
    summaries = short_player.play_headless()
    print(f"  回放帧数: {len(summaries)}")
    print(f"  首帧: t={summaries[0]['time']:.3f}s, "
          f"qpos=[{summaries[0]['qpos'][0]:.4f}, {summaries[0]['qpos'][1]:.4f}]")

    print("\n" + "=" * 60)
    print("TrajectoryPlayer 演示完成!")
    print("  ✓ 支持 pkl / npy / npz / hdf5 格式")
    print("  ✓ 帧查询支持时间插值")
    print("  ✓ 轨迹统计覆盖位置/速度/完整性")
    print("  ✓ Viewer / Headless 双模式回放")
    print("=" * 60)


if __name__ == "__main__":
    main()
