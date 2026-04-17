# 第 1 章 · 04 — MuJoCo Viewer 可视化

> **目标**: 学会使用 MuJoCo 内置的 3D 查看器。

> 需要图形界面环境，远程服务器请跳过。

---

## Viewer 快捷键

| 快捷键 | 功能 |
| :--- | :--- |
| 空格 | 暂停/继续仿真 |
| Backspace | 重置 |
| Tab | 切换相机视角 |
| 鼠标左键 | 旋转视角 |
| 鼠标右键 | 平移视角 |
| 滚轮 | 缩放 |
| 双击物体 | 选中并跟踪 |
| Ctrl+A | 显示/隐藏所有 |
| F1 | 显示帮助 |

---

## 示例 1: 交互式 Viewer

```python
model = mujoco.MjModel.from_xml_path("models/pendulum.xml")
data = mujoco.MjData(model)
data.qpos[0] = np.deg2rad(60)

mujoco.viewer.launch(model, data)   # 阻塞，内部自动调用 mj_step
```

`launch()` 打开窗口后，MuJoCo 自动运行仿真。关闭窗口才会继续执行后续代码。

---

## 示例 2: 被动式 Viewer（代码控制仿真）

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[0] = 3.0 * np.sin(2 * np.pi * 0.5 * data.time)
        mujoco.mj_step(model, data)
        viewer.sync()           # 同步渲染

        # 保持实时速度
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
```

### 两种模式对比

| | `launch` | `launch_passive` |
| :--- | :--- | :--- |
| 谁控制仿真 | Viewer 内部 | 你的代码 |
| 能设置 ctrl | 有限 | 完全自由 |
| 阻塞 | 是 | 否（循环内） |
| 适合 | 快速查看模型 | 自定义控制逻辑 |
