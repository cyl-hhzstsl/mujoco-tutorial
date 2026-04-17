# 第 4 章：真实机器人模型实战

> **从教学模型到工业级机器人 —— 学会加载、分析、录制、回放真实机器人数据**

---

## 本章目标

| # | 目标 | 对应脚本 |
|---|------|---------|
| 1 | 从 `robot_descriptions` 库加载真实机器人模型 | `01_load_robots.py` |
| 2 | 理解不同机器人的 qpos 结构差异 | `02_compare_robots.py` |
| 3 | 录制仿真轨迹并保存为多种格式 | `03_trajectory_recording.py` |
| 4 | 加载并回放已录制的轨迹 | `04_trajectory_replay.py` |

## 涉及的机器人

| 机器人 | 类型 | 制造商 | 自由度 |
|--------|------|--------|--------|
| UR5e | 6 轴协作机械臂 | Universal Robots | 6 |
| Franka Panda | 7 轴协作机械臂 | Franka Emika | 7 (+2 手指) |
| Unitree Go2 | 四足机器人 | 宇树科技 | 12 (+6 free joint) |
| Unitree H1 | 人形机器人 | 宇树科技 | 19+ (+7 free joint) |
| ALOHA | 双臂遥操作系统 | Google DeepMind | 14 (2×7) |

## 前置知识

- 第 2 章：MJCF 模型格式（关节、执行器、传感器）
- 第 3 章：qpos/qvel 结构与关节类型

## 环境准备

```bash
# 安装真实机器人模型描述库（可选，脚本有内置回退模型）
pip install robot_descriptions

# 本章还需要
pip install matplotlib numpy
```

> **注意**: `robot_descriptions` 是可选依赖。所有脚本在该库不可用时
> 会自动使用内置的简化 MJCF 模型，确保在任何环境下都能运行。

## 学习路线

```
01_load_robots.py          加载多种真实机器人，打印关节结构
        │
        ▼
02_compare_robots.py       对比不同类型机器人的 qpos 差异
        │
        ▼
03_trajectory_recording.py 运行仿真并录制轨迹数据
        │
        ▼
04_trajectory_replay.py    加载并回放已录制的轨迹
```

## 核心概念

### 固定基座 vs 浮动基座

- **固定基座 (Fixed Base)**: 机械臂（UR5e、Franka）—— 基座固定在桌面，qpos 只包含关节角
- **浮动基座 (Floating Base)**: 移动机器人（Go2、H1）—— 基座可自由移动，qpos 前 7 位是 `[x, y, z, qw, qx, qy, qz]`

### 轨迹数据结构

```
trajectory = {
    "time":     shape (T,),         # 时间戳
    "qpos":     shape (T, nq),      # 广义坐标
    "qvel":     shape (T, nv),      # 广义速度
    "ctrl":     shape (T, nu),      # 控制信号
    "ee_pos":   shape (T, 3),       # 末端执行器位置（如适用）
    "metadata": { ... }             # 机器人信息、采样率等
}
```

---

**开始学习 → [01_load_robots.py](01_load_robots.py)**
