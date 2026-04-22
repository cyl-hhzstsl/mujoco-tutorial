"""
第 8 章 · 04 - 多模态机器人数据处理 (Multi-Modal Robot Data)

目标: 理解如何同步采集、对齐、存储来自不同传感器的数据，
     包括关节状态 (qpos)、渲染图像、力/力矩传感器。

核心知识点:
  1. MuJoCo 离屏渲染: 从仿真中获取相机图像
  2. 多模态数据对齐: 不同频率的数据流如何同步
  3. HDF5 多模态存储: 结构化存储异构数据
  4. MultiModalRecorder: 可复用的多模态数据采集类
  5. 数据完整性校验: 确保各模态数据对齐

数据工程视角:
  - 机器人数据不只是 qpos，还有图像、力、IMU 等
  - 不同模态的采样频率不同（qpos: 1kHz, 图像: 30Hz）
  - 存储设计要考虑读取效率和数据对齐

运行: python 04_multimodal_data.py
输出: multimodal_episode.h5 (多模态数据文件)
依赖: pip install numpy matplotlib mujoco h5py
"""

import numpy as np
import mujoco
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70

# ============================================================
# 带传感器的机械臂 MJCF
# ============================================================

SENSOR_ARM_XML = """
<mujoco model="multimodal_arm">
  <option gravity="0 0 -9.81" timestep="0.001"/>

  <visual>
    <global offwidth="320" offheight="240"/>
  </visual>

  <worldbody>
    <light pos="0 -1 2" dir="0 1 -1" diffuse="1 1 1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>

    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.06 0.05" rgba="0.3 0.3 0.3 1" mass="0"/>

      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 0 1" limited="true"
               range="-180 180" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025"
              rgba="0.4 0.6 0.8 1" mass="1.0"/>

        <body name="link2" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" limited="true"
                 range="-120 120" damping="0.3"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02"
                rgba="0.6 0.8 0.4 1" mass="0.7"/>

          <!-- 末端执行器 + 力传感器 -->
          <body name="end_effector" pos="0.25 0 0">
            <site name="ee_site" pos="0 0 0" size="0.02" rgba="1 0.2 0.2 1"/>
            <site name="force_sensor" pos="0 0 0" size="0.01"/>
            <geom type="sphere" size="0.03" rgba="1 0.3 0.3 1" mass="0.2"/>
          </body>
        </body>
      </body>
    </body>

    <!-- 相机 -->
    <camera name="overhead" pos="0.2 -0.8 0.8" xyaxes="1 0 0 0 0.7 0.7"/>
    <camera name="side" pos="0.8 0 0.4" xyaxes="0 1 0 -0.5 0 0.87"/>
  </worldbody>

  <actuator>
    <motor joint="j1" ctrlrange="-20 20"/>
    <motor joint="j2" ctrlrange="-15 15"/>
  </actuator>

  <sensor>
    <framepos name="ee_pos" objtype="site" objname="ee_site"/>
    <framequat name="ee_quat" objtype="site" objname="ee_site"/>
    <jointpos name="j1_pos" joint="j1"/>
    <jointpos name="j2_pos" joint="j2"/>
    <jointvel name="j1_vel" joint="j1"/>
    <jointvel name="j2_vel" joint="j2"/>
    <actuatorfrc name="j1_force" actuator="motor1" noise="0.01"/>
    <actuatorfrc name="j2_force" actuator="motor2" noise="0.01"/>
  </sensor>

  <actuator>
    <!-- 重新定义以匹配 sensor 中的名称 -->
  </actuator>
</mujoco>
"""

# 修正后的 XML（actuator 名称需与 sensor 一致）
SENSOR_ARM_XML_FIXED = """
<mujoco model="multimodal_arm">
  <option gravity="0 0 -9.81" timestep="0.001"/>

  <visual>
    <global offwidth="320" offheight="240"/>
  </visual>

  <worldbody>
    <light pos="0 -1 2" dir="0 1 -1" diffuse="1 1 1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>

    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.06 0.05" rgba="0.3 0.3 0.3 1" mass="0"/>

      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 0 1" limited="true"
               range="-180 180" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025"
              rgba="0.4 0.6 0.8 1" mass="1.0"/>

        <body name="link2" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" limited="true"
                 range="-120 120" damping="0.3"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02"
                rgba="0.6 0.8 0.4 1" mass="0.7"/>

          <body name="end_effector" pos="0.25 0 0">
            <site name="ee_site" pos="0 0 0" size="0.02" rgba="1 0.2 0.2 1"/>
            <geom type="sphere" size="0.03" rgba="1 0.3 0.3 1" mass="0.2"/>
          </body>
        </body>
      </body>
    </body>

    <camera name="overhead" pos="0.2 -0.8 0.8" xyaxes="1 0 0 0 0.7 0.7"/>
    <camera name="side" pos="0.8 0 0.4" xyaxes="0 1 0 -0.5 0 0.87"/>
  </worldbody>

  <actuator>
    <motor name="motor1" joint="j1" ctrlrange="-20 20"/>
    <motor name="motor2" joint="j2" ctrlrange="-15 15"/>
  </actuator>

  <sensor>
    <framepos name="ee_pos" objtype="site" objname="ee_site"/>
    <framequat name="ee_quat" objtype="site" objname="ee_site"/>
    <jointpos name="j1_pos" joint="j1"/>
    <jointpos name="j2_pos" joint="j2"/>
    <jointvel name="j1_vel" joint="j1"/>
    <jointvel name="j2_vel" joint="j2"/>
    <actuatorfrc name="j1_force" actuator="motor1"/>
    <actuatorfrc name="j2_force" actuator="motor2"/>
  </sensor>
</mujoco>
"""


# ============================================================
# 第 1 节：离屏渲染
# ============================================================

class ImageRenderer:
    """
    MuJoCo 离屏图像渲染器。

    支持两种模式:
      1. 真实渲染: 使用 MuJoCo 的 OpenGL 离屏渲染（需要 GPU/EGL）
      2. Mock 渲染: 生成随机图像（无 GPU 时的降级方案）
    """

    def __init__(self, model: mujoco.MjModel, width: int = 320, height: int = 240):
        self.model = model
        self.width = width
        self.height = height
        self.renderer = None
        self._init_renderer()

    def _init_renderer(self) -> None:
        """尝试初始化渲染器，失败时使用 mock 模式。"""
        try:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
            self.mode = "real"
        except Exception as e:
            print(f"  ⚠️ 渲染器初始化失败 ({e}), 使用 mock 模式")
            self.mode = "mock"

    def render(self, data: mujoco.MjData,
               camera_name: Optional[str] = None) -> np.ndarray:
        """
        渲染当前场景为 RGB 图像。

        返回: uint8 数组 (height, width, 3)
        """
        if self.mode == "real" and self.renderer is not None:
            self.renderer.update_scene(data, camera=camera_name)
            img = self.renderer.render()
            return img.copy()
        else:
            return self._mock_render(data)

    def _mock_render(self, data: mujoco.MjData) -> np.ndarray:
        """
        Mock 渲染：基于 qpos 生成伪图像。
        在没有 GPU 的服务器上也能运行，保持数据管道完整。
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = 200  # 灰色背景

        # 根据关节角度绘制简单的色块
        for j in range(min(2, len(data.qpos))):
            val = data.qpos[j]
            normalized = (np.sin(val) + 1) / 2
            color_val = int(normalized * 200) + 55
            y_start = j * (self.height // 2)
            y_end = (j + 1) * (self.height // 2)
            img[y_start:y_end, :, j] = color_val

        return img

    def close(self) -> None:
        """释放渲染器资源。"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# ============================================================
# 第 2 节：多模态数据记录器
# ============================================================

@dataclass
class ModalityConfig:
    """单个数据模态的配置。"""
    name: str
    frequency: float         # 采样频率 (Hz)
    shape: Tuple[int, ...]   # 单帧数据形状
    dtype: str = "float32"   # 数据类型


class MultiModalRecorder:
    """
    多模态数据记录器：同步采集不同频率的传感器数据。

    设计要点:
      1. 不同模态可以有不同的采样频率
      2. 自动计算每个模态的采样间隔
      3. 记录精确的时间戳用于后续对齐
      4. 支持导出为 HDF5 格式

    使用示例:
      recorder = MultiModalRecorder(sim_dt=0.001)
      recorder.add_modality("qpos", frequency=1000, shape=(2,))
      recorder.add_modality("image", frequency=30, shape=(240, 320, 3), dtype="uint8")
      recorder.start_episode()
      for step in range(n_steps):
          recorder.record("qpos", step, data.qpos)
          if recorder.should_record("image", step):
              recorder.record("image", step, image)
      episode = recorder.end_episode()
    """

    def __init__(self, sim_dt: float = 0.001):
        self.sim_dt = sim_dt
        self.modalities: Dict[str, ModalityConfig] = {}
        self._buffers: Dict[str, List] = {}
        self._timestamps: Dict[str, List[float]] = {}
        self._step_intervals: Dict[str, int] = {}
        self._recording = False
        self._episode_count = 0

    def add_modality(self, name: str, frequency: float,
                     shape: Tuple[int, ...],
                     dtype: str = "float32") -> "MultiModalRecorder":
        """注册一个数据模态。"""
        self.modalities[name] = ModalityConfig(
            name=name, frequency=frequency, shape=shape, dtype=dtype
        )
        # 计算采样间隔（每隔多少仿真步采样一次）
        self._step_intervals[name] = max(1, int(1.0 / (frequency * self.sim_dt)))
        return self

    def should_record(self, modality: str, step: int) -> bool:
        """判断当前步是否应该采样该模态。"""
        return step % self._step_intervals[modality] == 0

    def start_episode(self) -> None:
        """开始新的 episode 录制。"""
        self._buffers = {name: [] for name in self.modalities}
        self._timestamps = {name: [] for name in self.modalities}
        self._recording = True

    def record(self, modality: str, step: int, data: np.ndarray) -> None:
        """记录一帧数据。"""
        if not self._recording:
            raise RuntimeError("未在录制中，请先调用 start_episode()")
        if modality not in self.modalities:
            raise ValueError(f"未注册的模态: {modality}")

        self._buffers[modality].append(np.array(data, copy=True))
        self._timestamps[modality].append(step * self.sim_dt)

    def end_episode(self) -> Dict[str, Any]:
        """
        结束录制，返回结构化的 episode 数据。
        """
        self._recording = False
        self._episode_count += 1

        episode = {
            "episode_id": self._episode_count,
            "sim_dt": self.sim_dt,
            "modalities": {},
        }

        for name, config in self.modalities.items():
            buf = self._buffers[name]
            if buf:
                episode["modalities"][name] = {
                    "data": np.array(buf, dtype=config.dtype),
                    "timestamps": np.array(self._timestamps[name]),
                    "frequency": config.frequency,
                    "shape": config.shape,
                    "n_frames": len(buf),
                }

        return episode

    def get_statistics(self) -> Dict[str, Dict]:
        """获取当前录制的统计信息。"""
        stats = {}
        for name in self.modalities:
            buf = self._buffers.get(name, [])
            stats[name] = {
                "n_frames": len(buf),
                "interval_steps": self._step_intervals[name],
                "expected_frequency": self.modalities[name].frequency,
            }
        return stats


# ============================================================
# 第 3 节：HDF5 存储
# ============================================================

def save_episode_hdf5(episode: Dict[str, Any],
                      filepath: str) -> None:
    """
    将 episode 数据保存为 HDF5 文件。

    HDF5 结构:
      /metadata
        episode_id, sim_dt, timestamp
      /qpos
        data, timestamps
      /image
        data, timestamps
      /force
        data, timestamps
    """
    try:
        import h5py
    except ImportError:
        print("  ⚠️ h5py 未安装，使用 npz 格式替代")
        save_episode_npz(episode, filepath.replace(".h5", ".npz"))
        return

    with h5py.File(filepath, "w") as f:
        # 元数据
        meta = f.create_group("metadata")
        meta.attrs["episode_id"] = episode["episode_id"]
        meta.attrs["sim_dt"] = episode["sim_dt"]
        meta.attrs["save_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # 各模态数据
        for name, mod_data in episode["modalities"].items():
            grp = f.create_group(name)

            # 图像数据使用压缩（减小文件大小）
            if mod_data["data"].dtype == np.uint8:
                grp.create_dataset("data", data=mod_data["data"],
                                   compression="gzip", compression_opts=4,
                                   chunks=True)
            else:
                grp.create_dataset("data", data=mod_data["data"])

            grp.create_dataset("timestamps", data=mod_data["timestamps"])
            grp.attrs["frequency"] = mod_data["frequency"]
            grp.attrs["shape"] = mod_data["shape"]
            grp.attrs["n_frames"] = mod_data["n_frames"]

    print(f"  ✓ HDF5 已保存: {filepath}")
    print(f"    文件大小: {os.path.getsize(filepath) / 1024:.1f} KB")


def save_episode_npz(episode: Dict[str, Any], filepath: str) -> None:
    """HDF5 不可用时的 npz 降级方案。"""
    save_dict = {
        "episode_id": np.array(episode["episode_id"]),
        "sim_dt": np.array(episode["sim_dt"]),
    }
    for name, mod_data in episode["modalities"].items():
        save_dict[f"{name}_data"] = mod_data["data"]
        save_dict[f"{name}_timestamps"] = mod_data["timestamps"]

    np.savez_compressed(filepath, **save_dict)
    print(f"  ✓ NPZ 已保存: {filepath}")
    print(f"    文件大小: {os.path.getsize(filepath) / 1024:.1f} KB")


def load_episode_hdf5(filepath: str) -> Dict[str, Any]:
    """从 HDF5 加载 episode 数据。"""
    try:
        import h5py
    except ImportError:
        raise ImportError("需要 h5py: pip install h5py")

    episode = {"modalities": {}}

    with h5py.File(filepath, "r") as f:
        # 元数据
        meta = f["metadata"]
        episode["episode_id"] = int(meta.attrs["episode_id"])
        episode["sim_dt"] = float(meta.attrs["sim_dt"])

        # 各模态
        for key in f.keys():
            if key == "metadata":
                continue
            grp = f[key]
            episode["modalities"][key] = {
                "data": grp["data"][:],
                "timestamps": grp["timestamps"][:],
                "frequency": float(grp.attrs["frequency"]),
                "n_frames": int(grp.attrs["n_frames"]),
            }

    return episode


# ============================================================
# 第 4 节：数据对齐
# ============================================================

class DataAligner:
    """
    多模态数据对齐器。

    不同模态的采样频率不同，需要对齐到统一的时间轴:
      - qpos: 1000 Hz → 每步都有
      - image: 30 Hz → 每 33 步一帧
      - force: 200 Hz → 每 5 步一次

    对齐策略:
      1. 最近邻: 找最近的时间戳
      2. 线性插值: 在两个时间戳之间插值
      3. 零阶保持: 使用最近的较早的值
    """

    @staticmethod
    def align_nearest(target_timestamps: np.ndarray,
                      source_timestamps: np.ndarray,
                      source_data: np.ndarray) -> np.ndarray:
        """最近邻对齐。"""
        indices = np.searchsorted(source_timestamps, target_timestamps)
        indices = np.clip(indices, 0, len(source_timestamps) - 1)

        # 检查前一个时间戳是否更近
        for i in range(len(indices)):
            idx = indices[i]
            if idx > 0:
                d_after = abs(source_timestamps[idx] - target_timestamps[i])
                d_before = abs(source_timestamps[idx - 1] - target_timestamps[i])
                if d_before < d_after:
                    indices[i] = idx - 1

        return source_data[indices]

    @staticmethod
    def align_interpolate(target_timestamps: np.ndarray,
                          source_timestamps: np.ndarray,
                          source_data: np.ndarray) -> np.ndarray:
        """线性插值对齐（仅适用于连续数值数据）。"""
        if source_data.ndim == 1:
            return np.interp(target_timestamps, source_timestamps, source_data)

        result = np.zeros((len(target_timestamps), source_data.shape[1]),
                          dtype=source_data.dtype)
        for col in range(source_data.shape[1]):
            result[:, col] = np.interp(
                target_timestamps, source_timestamps, source_data[:, col]
            )
        return result

    @staticmethod
    def align_zero_order_hold(target_timestamps: np.ndarray,
                              source_timestamps: np.ndarray,
                              source_data: np.ndarray) -> np.ndarray:
        """零阶保持：使用最近的较早值。"""
        indices = np.searchsorted(source_timestamps, target_timestamps, side="right") - 1
        indices = np.clip(indices, 0, len(source_timestamps) - 1)
        return source_data[indices]


# ============================================================
# 第 5 节：完整演示
# ============================================================

def demo_multimodal_recording() -> Dict[str, Any]:
    """
    完整的多模态数据录制演示。
    """
    print(f"\n{DIVIDER}")
    print("第 5 节：多模态数据录制演示")
    print(DIVIDER)

    # --- 加载模型 ---
    model = mujoco.MjModel.from_xml_string(SENSOR_ARM_XML_FIXED)
    data = mujoco.MjData(model)

    print(f"\n  模型信息:")
    print(f"    关节: {model.njnt}, 传感器: {model.nsensor}")
    print(f"    相机数: {model.ncam}")

    # --- 初始化渲染器 ---
    renderer = ImageRenderer(model, width=320, height=240)
    print(f"    渲染模式: {renderer.mode}")

    # --- 配置录制器 ---
    recorder = MultiModalRecorder(sim_dt=model.opt.timestep)
    recorder \
        .add_modality("qpos", frequency=1000.0, shape=(model.nq,)) \
        .add_modality("qvel", frequency=1000.0, shape=(model.nv,)) \
        .add_modality("sensor", frequency=1000.0, shape=(model.nsensordata,)) \
        .add_modality("ctrl", frequency=1000.0, shape=(model.nu,)) \
        .add_modality("image_overhead", frequency=30.0,
                      shape=(240, 320, 3), dtype="uint8") \
        .add_modality("ee_pos", frequency=200.0, shape=(3,))

    print(f"\n  录制配置:")
    for name, config in recorder.modalities.items():
        interval = recorder._step_intervals[name]
        print(f"    {name}: {config.frequency} Hz "
              f"(每 {interval} 步采样), shape={config.shape}")

    # --- 录制一个 episode ---
    duration = 3.0
    n_steps = int(duration / model.opt.timestep)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    recorder.start_episode()
    print(f"\n  开始录制: {duration}s, {n_steps} 步...")

    for step in range(n_steps):
        t = step * model.opt.timestep

        # 控制信号: 正弦波
        data.ctrl[0] = 8 * np.sin(2 * np.pi * 0.3 * t)
        data.ctrl[1] = 5 * np.sin(2 * np.pi * 0.5 * t + np.pi / 3)

        mujoco.mj_step(model, data)

        # 录制各模态
        if recorder.should_record("qpos", step):
            recorder.record("qpos", step, data.qpos[:model.nq])

        if recorder.should_record("qvel", step):
            recorder.record("qvel", step, data.qvel[:model.nv])

        if recorder.should_record("sensor", step):
            recorder.record("sensor", step, data.sensordata[:model.nsensordata])

        if recorder.should_record("ctrl", step):
            recorder.record("ctrl", step, data.ctrl[:model.nu])

        if recorder.should_record("image_overhead", step):
            img = renderer.render(data, camera_name="overhead")
            recorder.record("image_overhead", step, img)

        if recorder.should_record("ee_pos", step):
            recorder.record("ee_pos", step, data.site_xpos[site_id])

        if (step + 1) % 500 == 0:
            print(f"    步骤 {step + 1}/{n_steps}")

    episode = recorder.end_episode()
    renderer.close()

    # --- 打印统计 ---
    print(f"\n  录制完成! Episode #{episode['episode_id']}")
    print(f"\n  {'模态':<20} {'帧数':<10} {'频率':<12} {'数据形状':<20} {'大小 (KB)'}")
    print(f"  {SUB_DIVIDER}")

    total_size = 0
    for name, mod_data in episode["modalities"].items():
        data_arr = mod_data["data"]
        size_kb = data_arr.nbytes / 1024
        total_size += size_kb
        print(f"  {name:<20} {mod_data['n_frames']:<10} "
              f"{mod_data['frequency']:<12.0f} "
              f"{str(data_arr.shape):<20} {size_kb:.1f}")

    print(f"  {'合计':<20} {'':<10} {'':<12} {'':<20} {total_size:.1f}")

    return episode


def demo_data_alignment(episode: Dict[str, Any]) -> None:
    """数据对齐演示。"""
    print(f"\n{DIVIDER}")
    print("数据对齐演示")
    print(DIVIDER)

    aligner = DataAligner()

    # 将图像时间戳对齐到 qpos 时间轴
    qpos_ts = episode["modalities"]["qpos"]["timestamps"]
    img_ts = episode["modalities"]["image_overhead"]["timestamps"]
    ee_ts = episode["modalities"]["ee_pos"]["timestamps"]
    ee_data = episode["modalities"]["ee_pos"]["data"]

    print(f"\n  qpos 时间戳: {len(qpos_ts)} 帧, "
          f"范围 [{qpos_ts[0]:.3f}, {qpos_ts[-1]:.3f}]s")
    print(f"  image 时间戳: {len(img_ts)} 帧, "
          f"范围 [{img_ts[0]:.3f}, {img_ts[-1]:.3f}]s")
    print(f"  ee_pos 时间戳: {len(ee_ts)} 帧, "
          f"范围 [{ee_ts[0]:.3f}, {ee_ts[-1]:.3f}]s")

    # 将 ee_pos 插值到每个图像帧的时间
    ee_at_img_times = aligner.align_interpolate(img_ts, ee_ts, ee_data)
    print(f"\n  ee_pos 对齐到 image 时间: {ee_at_img_times.shape}")
    print(f"  对齐后第一帧 ee_pos: [{ee_at_img_times[0, 0]:.4f}, "
          f"{ee_at_img_times[0, 1]:.4f}, {ee_at_img_times[0, 2]:.4f}]")


# ============================================================
# 主程序
# ============================================================

def main():
    print(DIVIDER)
    print("第 8 章 · 04 - 多模态机器人数据")
    print("图像渲染 · HDF5 存储 · 数据对齐")
    print(DIVIDER)

    # 多模态录制
    episode = demo_multimodal_recording()

    # 数据对齐
    demo_data_alignment(episode)

    # 保存为 HDF5
    print(f"\n{SUB_DIVIDER}")
    save_path = "multimodal_episode.h5"
    save_episode_hdf5(episode, save_path)

    # --- 验证读取 ---
    try:
        import h5py
        loaded = load_episode_hdf5(save_path)
        print(f"\n  ✓ HDF5 读取验证成功:")
        for name, mod_data in loaded["modalities"].items():
            print(f"    {name}: {mod_data['data'].shape}")
    except ImportError:
        print("\n  (跳过 HDF5 读取验证，h5py 未安装)")

    # --- 总结 ---
    print(f"\n{DIVIDER}")
    print("关键收获:")
    print("  1. 多模态数据: qpos + 图像 + 力，来自不同传感器")
    print("  2. 采样频率差异: qpos=1kHz, image=30Hz → 需要对齐策略")
    print("  3. HDF5 适合存储异构数据: 压缩图像 + 浮点轨迹")
    print("  4. 数据对齐有三种策略: 最近邻、插值、零阶保持")
    print("  5. MultiModalRecorder 是可复用的数据采集基础设施")
    print(DIVIDER)


if __name__ == "__main__":
    main()
