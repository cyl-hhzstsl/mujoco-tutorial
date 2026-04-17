# 第 1 章：MuJoCo 安装与入门

> 从零开始安装 MuJoCo，理解仿真引擎的核心概念。

## 本章目标

- 安装 MuJoCo 并验证环境
- 理解 Model / Data / Step 三大核心概念
- 学会用 Viewer 可视化模型
- 完成第一次仿真循环

## 文件列表

| 文件 | 内容 | 预计时间 |
|------|------|---------|
| `01_install_verify.py` | 安装验证与基本概念 | 20 min |
| `02_core_concepts.py` | Model / Data / Step 详解 | 45 min |
| `03_first_simulation.py` | 第一个完整仿真 | 30 min |
| `04_viewer_basics.py` | 可视化查看器使用 | 30 min |
| `models/ball_drop.xml` | 自由落体模型 | - |
| `models/pendulum.xml` | 单摆模型 | - |

## 核心概念预览

```
MjModel (模型)         MjData (数据)           mj_step (仿真)
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ 不变的结构信息 │      │ 运行时状态    │      │ 推进一个时间步│
│ - 关节定义    │      │ - qpos       │      │ - 碰撞检测   │
│ - 几何形状    │      │ - qvel       │      │ - 约束求解   │
│ - 物理参数    │  →   │ - ctrl       │  →   │ - 数值积分   │
│ - 执行器     │      │ - xpos       │      │ - 更新 data  │
│ (类比: Class) │      │ (类比: Object)│      │ (类比: tick())│
└──────────────┘      └──────────────┘      └──────────────┘
```

## 运行方式

```bash
cd chapter_01_mujoco_intro
python 01_install_verify.py
python 02_core_concepts.py
python 03_first_simulation.py
python 04_viewer_basics.py    # 会弹出 3D 窗口
```

## 通关标准

- [ ] MuJoCo 安装成功，能打印版本号
- [ ] 理解 model 和 data 的区别
- [ ] 能从 XML 加载模型并运行仿真
- [ ] 能使用 viewer 查看模型

---

**完成后 → [第 2 章：MJCF 建模详解](../chapter_02_mjcf_modeling/README.md)**
