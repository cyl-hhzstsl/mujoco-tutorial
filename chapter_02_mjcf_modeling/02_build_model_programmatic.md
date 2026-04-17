# 第 2 章 · 02 — 程序化构建模型

> **目标**: 学会用 Python 动态创建和修改 MJCF 模型，而不是手写 XML。

---

## 1. 从 XML 字符串创建模型

```python
xml_string = """
<mujoco model="from_string">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 2">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml_string)
```

---

## 2. 运行时修改参数

编译后的 `model.opt` 中部分参数可以直接修改：

```python
model.opt.gravity[:] = [0, 0, -1.62]   # 改为月球重力
model.opt.timestep = 0.001              # 改时间步长
```

### 重力对比实验

| 环境 | 重力 | 1s 后高度 |
| :--- | :--- | :--- |
| 地球 | -9.81 | ≈ -2.9m |
| 月球 | -1.62 | ≈ +1.2m |
| 火星 | -3.72 | ≈ +0.1m |
| 零重力 | 0 | +2.0m（不动） |

### 时间步长对比

dt 越小精度越高，但仿真越慢。推荐 0.001 ~ 0.005。

---

## 3. MjSpec API (MuJoCo 3.x)

结构化构建模型，比拼接 XML 字符串更清晰：

```python
spec = mujoco.MjSpec()
spec.modelname = "my_robot"
spec.option.gravity = (0, 0, -9.81)

body = spec.worldbody.add_body()
body.name = "link1"

joint = body.add_joint()
joint.type = mujoco.mjtJoint.mjJNT_HINGE
joint.axis = [0, 1, 0]

geom = body.add_geom()
geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
geom.fromto = [0, 0, 0, 0, 0, -0.5]

model = spec.compile()
xml = spec.to_xml()    # 导出为 XML
```

### MjSpec 不可用时的替代方案

用 Python f-string 拼接 XML：

```python
def build_chain_xml(n_links):
    xml = '<mujoco>...'
    for i in range(n_links):
        xml += f'<body name="link{i+1}">...'
    return xml
```

---

## 4. N 连杆链模型对比

| 连杆数 | nq | nv | 说明 |
| :---: | :---: | :---: | :--- |
| 1 | 1 | 1 | 单摆 |
| 2 | 2 | 2 | 双摆 |
| 3 | 3 | 3 | 类似 3DOF 机械臂 |
| 5 | 5 | 5 | 柔性链/蛇形 |
| 10 | 10 | 10 | 复杂链条 |

每个 hinge 关节贡献 1 个 qpos 和 1 个 qvel。

---

## 关键知识点

| 方法 | 说明 |
| :--- | :--- |
| `MjModel.from_xml_string()` | 从字符串创建模型 |
| `MjModel.from_xml_path()` | 从文件创建模型 |
| `model.opt.gravity/timestep` | 运行时可修改仿真参数 |
| `MjSpec` API (3.x) | 结构化构建和修改模型 |
| XML 字符串拼接 | 兼容所有版本的动态构建方式 |
