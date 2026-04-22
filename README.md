# MuJoCo 机器人仿真与数据工程实战教程

> **面向后端工程师的机器人数据开发完整指南**
>
> 从 Python 基础到企业级机器人数据平台设计，8 章循序渐进，全部可实操运行。

---

## 适合谁

- Java/Go/C++ 后端开发，转型或兼顾机器人数据平台开发
- 有编程基础，但不熟悉 Python 科学计算生态
- 想系统理解机器人仿真数据，而非成为算法研究员

## 你将学到

| 能力 | 对应章节 |
|------|---------|
| Python 科学计算基础 | 第 0 章 |
| MuJoCo 仿真引擎使用 | 第 1 章 |
| 机器人模型（MJCF）的阅读与编写 | 第 2 章 |
| qpos/qvel/ctrl 的完整理解 | 第 3 章 |
| 加载和分析真实机器人模型 | 第 4 章 |
| 机器人数据的存储格式（HDF5/PKL/LeRobot） | 第 5 章 |
| 数据质量校验与异常检测 | 第 6 章 |
| 企业级数据平台架构设计 | 第 7 章 |
| 运动学、控制、Sim2Real、mink IK 进阶 | 第 8 章 |

## 目录结构

```
mujoco-tutorial/
├── README.md                    ← 你在这里
├── requirements.txt             ← 依赖清单
├── chapter_00_python_basics/    ← Python 科学计算基础
├── chapter_01_mujoco_intro/     ← MuJoCo 安装与入门
├── chapter_02_mjcf_modeling/    ← MJCF 模型格式详解
├── chapter_03_qpos_deep_dive/   ← qpos 深度解析
├── chapter_04_real_robots/      ← 真实机器人模型实战
├── chapter_05_data_formats/     ← 机器人数据格式
├── chapter_06_data_quality/     ← 数据质量与校验
├── chapter_07_data_platform/    ← 数据平台设计
├── chapter_08_advanced/         ← 进阶主题
└── utils/                       ← 公共工具库
```

## 环境准备

```bash
# 1. 创建 Conda 环境
conda create -n mujoco-tutorial python=3.10 -y
conda activate mujoco-tutorial

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "import mujoco; print(f'MuJoCo {mujoco.__version__} 安装成功')"
```

## 学习建议

- **每章按顺序学习**，后面的章节依赖前面的知识
- **每个 .py 文件都可以直接运行**，建议先运行看效果，再阅读代码
- **动手修改参数**，观察变化，比纯看代码学得快 10 倍
- **完成每章末尾的练习**再进入下一章

## 预计时间

| 阶段 | 章节 | 时间 |
|------|------|------|
| 基础 | 第 0-1 章 | 1.5 周 |
| 核心 | 第 2-4 章 | 2.5 周 |
| 数据 | 第 5-7 章 | 3 周 |
| 进阶 | 第 8 章 | 持续 |

---

**开始学习 → [第 0 章：Python 科学计算基础](chapter_00_python_basics/README.md)**
