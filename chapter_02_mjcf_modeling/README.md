# 第 2 章：MJCF 建模详解

> 深入理解 MuJoCo 的 MJCF 建模语言，从 XML 结构到完整机器人。

## 本章目标

- 掌握 MJCF XML 的完整结构与各标签含义
- 理解 body / joint / geom 层级关系
- 掌握执行器（actuator）与传感器（sensor）的配置
- 从零构建一个 3 自由度机械臂模型
- 学会用 Python 程序化构建和修改模型

## 文件列表

| 文件 | 内容 | 预计时间 |
|------|------|---------|
| `models/double_pendulum.xml` | 双摆模型 | - |
| `models/robot_arm.xml` | 3-DOF 机械臂模型 | - |
| `01_mjcf_structure.py` | MJCF 结构全解析 | 45 min |
| `02_build_model_programmatic.py` | 程序化构建模型 | 40 min |
| `03_geom_and_collision.py` | 几何体与碰撞详解 | 50 min |
| `04_actuator_and_sensor.py` | 执行器与传感器 | 50 min |

## MJCF 结构预览

```
<mujoco>
├── <compiler>        编译器设置（角度单位、坐标系）
├── <option>          仿真选项（时间步、重力、积分器）
├── <default>         默认参数（减少重复）
├── <asset>           资源（材质、网格、纹理）
├── <worldbody>       世界体（场景根节点）
│   ├── <body>        刚体
│   │   ├── <joint>   关节（连接父子刚体）
│   │   ├── <geom>    几何体（碰撞/可视化）
│   │   ├── <site>    站点（传感器挂载点）
│   │   └── <body>    子刚体（递归嵌套）
│   └── ...
├── <actuator>        执行器（电机、位置伺服）
├── <sensor>          传感器（关节角、力、加速度）
└── <keyframe>        关键帧（预设状态）
```

## 运行方式

```bash
cd chapter_02_mjcf_modeling
python 01_mjcf_structure.py
python 02_build_model_programmatic.py
python 03_geom_and_collision.py
python 04_actuator_and_sensor.py    # 会生成 actuator_comparison.png
```

## 通关标准

- [ ] 能读懂任意 MJCF XML 文件的每个标签
- [ ] 理解 body 树形层级与坐标系传递关系
- [ ] 理解 joint 如何定义自由度
- [ ] 能区分碰撞几何体与视觉几何体
- [ ] 能配置不同类型的执行器并理解差异
- [ ] 能用 Python 程序化构建 MJCF 模型

---

**完成后 → [第 3 章：qpos 深度解析](../chapter_03_qpos_deep_dive/README.md)**
