# 第 8 章：进阶主题

> 运动学、控制、Sim-to-Real —— 当你理解了数据的来源，就能更好地设计数据系统。

## 本章目标

1. 理解**正运动学与逆运动学**（FK / IK），掌握关节空间与笛卡尔空间的映射
2. 实现**基础控制器**（PD / PID），理解控制信号如何变成机器人数据
3. 理解**Sim-to-Real 差距**对数据分布的影响，掌握域随机化技术
4. 实现**多模态数据采集**，同步记录关节状态、图像、力传感器数据
5. 使用**mink 库**实现专业级 IK，掌握约束优化在机器人控制中的应用

## 为什么后端工程师需要了解这些

| 场景 | 你会遇到的问题 |
|------|---------------|
| 数据清洗 | 为什么这条轨迹的末端执行器位置不合理？→ 需要理解 FK |
| 数据校验 | 控制器输出的力矩范围应该是多少？→ 需要理解 PD 控制 |
| 数据管道 | 仿真数据和真实数据分布为什么不同？→ 需要理解 Sim-to-Real |
| 存储设计 | 图像 + 关节 + 力如何对齐存储？→ 需要理解多模态数据 |

## 文件结构

| 文件 | 内容 |
|------|------|
| `01_kinematics.py` | 正运动学 / 逆运动学：3 连杆机械臂的 FK/IK、Jacobian 计算、工作空间分析 |
| `02_control_basics.py` | 控制基础：PD/PID 控制器实现、重力补偿、轨迹跟踪、控制器性能对比 |
| `03_sim_to_real.py` | Sim-to-Real：域随机化、参数扰动、数据分布对比、数据增强技术 |
| `04_multimodal_data.py` | 多模态数据：图像渲染、HDF5 存储、数据流对齐、MultiModalRecorder |
| `05_mink_ik.py` | mink 逆运动学：Configuration/FrameTask/solve_ik、约束 IK、与手写 IK 对比、轨迹生成 |

## 核心概念

```
关节空间 (Joint Space)          笛卡尔空间 (Cartesian Space)
    qpos = [θ1, θ2, θ3]  ──FK──▶  xyz = [x, y, z]
                          ◀──IK──
                               │
                               ▼
                    控制器 (Controller)
                    PD: τ = Kp·e + Kd·ė
                               │
                               ▼
                    仿真器 (Simulator)
                    MuJoCo: qpos → 传感器数据
                               │
                         ┌─────┼─────┐
                         ▼     ▼     ▼
                       qpos  图像   力
                         │     │     │
                         └─────┼─────┘
                               ▼
                    多模态数据 (HDF5)
```

## 运行方式

```bash
# 按顺序运行每个文件
python 01_kinematics.py          # 生成 workspace.png
python 02_control_basics.py      # 生成 control_comparison.png
python 03_sim_to_real.py         # 生成 sim2real_distributions.png
python 04_multimodal_data.py     # 生成 multimodal_episode.h5
python 05_mink_ik.py             # 生成 mink_vs_manual_ik.png, mink_trajectory.png
```

## 依赖

```bash
pip install numpy matplotlib mujoco
# 可选
pip install h5py    # 04 多模态数据存储
pip install mink    # 05 专业级 IK 库
```

## 本章与前几章的关系

```
第 2 章 (MJCF 模型)  ───▶  本章用到的机械臂模型
第 3 章 (qpos 深度)  ───▶  本章操作的核心数据
第 5 章 (数据格式)   ───▶  本章输出的 HDF5 文件
第 6 章 (数据质量)   ───▶  本章产生的数据需要校验
第 7 章 (数据平台)   ───▶  本章的数据最终流入平台
```

## 学习建议

- **01 运动学**：重点理解 FK 和 Jacobian，这决定了你能不能「读懂」轨迹数据
- **02 控制**：重点看 PD 控制的参数如何影响数据特征（振荡、超调、稳态误差）
- **03 Sim-to-Real**：关注数据分布的变化，这是数据工程师最需要理解的
- **04 多模态**：动手修改采样频率和数据对齐方式，体会存储设计的复杂性
- **05 mink**：先理解 01 的手写 IK 原理，再用 mink 体会专业库的抽象和约束能力
