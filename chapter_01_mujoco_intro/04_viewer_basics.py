"""
第 1 章 · 04 - MuJoCo Viewer 可视化

目标: 学会使用 MuJoCo 内置的 3D 查看器。

⚠️  本脚本会弹出 3D 窗口，需要图形界面环境。
    如果在远程服务器上运行，请跳过此节。

Viewer 快捷键:
  空格       暂停/继续仿真
  Backspace  重置
  Tab        切换相机视角
  鼠标左键   旋转视角
  鼠标右键   平移视角
  滚轮       缩放
  双击物体   选中并跟踪
  Ctrl+A     显示/隐藏所有
  F1         显示帮助

运行: python 04_viewer_basics.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pendulum.xml")

# ============================================================
# 示例 1: 最简启动 (交互式)
# ============================================================
print("=" * 60)
print("示例 1: 交互式 Viewer")
print("=" * 60)
print("弹出窗口后可以用鼠标旋转/缩放/平移")
print("按空格暂停, 关闭窗口继续下一个示例\n")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 给一个初始角度，否则摆不动
data.qpos[0] = np.deg2rad(60)

# launch() 会阻塞，直到用户关闭窗口
# 内部自动调用 mj_step
mujoco.viewer.launch(model, data)

# ============================================================
# 示例 2: 被动式 Viewer (代码控制仿真)
# ============================================================
print("\n" + "=" * 60)
print("示例 2: 被动式 Viewer (代码控制)")
print("=" * 60)
print("这次由代码控制仿真循环，施加正弦控制力")
print("关闭窗口退出\n")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
data.qpos[0] = np.deg2rad(30)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()

    while viewer.is_running():
        step_start = time.time()

        # 正弦控制力
        data.ctrl[0] = 3.0 * np.sin(2 * np.pi * 0.5 * data.time)

        # 仿真一步
        mujoco.mj_step(model, data)

        # 同步渲染
        viewer.sync()

        # 保持实时速度
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        # 每 2 秒打印状态
        if int(data.time * 10) % 20 == 0 and data.time > 0:
            angle_deg = np.rad2deg(data.qpos[0])
            vel_deg = np.rad2deg(data.qvel[0])
            print(f"  t={data.time:>5.1f}s  角度={angle_deg:>7.1f}°  "
                  f"角速度={vel_deg:>8.1f}°/s  ctrl={data.ctrl[0]:>5.2f}")

print("\n✅ 第 04 节完成！")
print("你已经学会了 MuJoCo 的两种 Viewer 模式。")
