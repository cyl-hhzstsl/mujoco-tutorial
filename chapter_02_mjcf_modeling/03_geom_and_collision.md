# 第 2 章 · 03 — 几何体与碰撞详解

> **目标**: 深入理解 MuJoCo 中所有几何体类型、碰撞检测机制、摩擦参数以及接触力的读取方式。

---

## 1. 几何体类型

| 类型 | 参数 | 说明 |
| :--- | :--- | :--- |
| plane | `size="半长 半宽 网格间距"` | 无限平面 |
| sphere | `size="半径"` | 球体，最简碰撞体 |
| capsule | `size="半径" fromto="起点 终点"` | 圆柱 + 两端半球，连杆首选 |
| box | `size="半长x 半长y 半长z"` | 长方体，注意是**半尺寸** |
| cylinder | `size="半径 半高"` | 圆柱体 |
| ellipsoid | `size="半轴x 半轴y 半轴z"` | 椭球体 |

---

## 2. 碰撞几何体 vs 视觉几何体 (contype / conaffinity)

碰撞规则：两个几何体发生碰撞当且仅当

```
(geom1.contype & geom2.conaffinity) != 0
   或
(geom2.contype & geom1.conaffinity) != 0
```

| 设置 | 效果 |
| :--- | :--- |
| contype=1, conaffinity=1 | 正常碰撞 |
| contype=0, conaffinity=0 | 幽灵体，穿过一切 |
| contype=2, conaffinity=2 | 只与同类 (conaffinity=2) 碰撞 |

---

## 3. 摩擦参数

```xml
<geom friction="滑动 扭转 滚动"/>
```

| 参数 | 含义 | 典型值 |
| :--- | :--- | :--- |
| 滑动 (tangential) | 阻止物体滑动 | 冰面 0.05, 木材 0.5, 橡胶 1.5 |
| 扭转 (torsional) | 阻止绕法线旋转 | 0.001 ~ 0.1 |
| 滚动 (rolling) | 阻止物体滚动 | 0.0001 ~ 0.01 |

### 斜面实验

30° 斜面上不同摩擦系数的滑动距离对比：摩擦越小滑得越远。

---

## 4. 接触检测与接触力

```python
# 仿真后读取接触
for i in range(data.ncon):
    contact = data.contact[i]
    contact.geom1, contact.geom2   # 接触的两个几何体 ID
    contact.pos                     # 接触点位置
    contact.dist                    # 穿透深度

    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, force)
    normal_force = np.linalg.norm(force[:3])
```

---

## 5. 接触参数说明

```
solref = [timeconst, dampratio]
  timeconst: 约束恢复时间常数（越小越硬）
  dampratio: 阻尼比（1.0 = 临界阻尼）

solimp = [dmin, dmax, width, midpoint, power]
  定义阻抗函数的形状

condim: 接触维度
  1 = 仅法向（无摩擦）
  3 = 法向 + 2 切向（普通摩擦）
  4 = 法向 + 2 切向 + 扭转
  6 = 完整（含滚动）
```

---

## 6. condim 对比

球体给予水平初速 2 m/s，不同 condim 下 1 秒后的表现：

| condim | 说明 | 效果 |
| :---: | :--- | :--- |
| 1 | 仅法向 | 无摩擦，一直滑 |
| 3 | + 切向摩擦 | 逐渐减速 |
| 4 | + 扭转 | 旋转也被阻止 |
| 6 | 完整 | 滚动也被阻止 |
