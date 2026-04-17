# 第 1 章 · 01 — 安装验证

> **目标**: 确认 MuJoCo 安装正确，了解版本和基本能力。

## 安装

```bash
pip install mujoco
```

---

## 检查项

### 1. Python 环境

```python
import sys
print(sys.version)
print(sys.executable)
```

### 2. MuJoCo

```python
import mujoco
print(mujoco.__version__)
```

### 3. 依赖库

| 库 | 用途 | 安装 |
| :--- | :--- | :--- |
| numpy | 数组运算（必需） | `pip install numpy` |
| matplotlib | 绘图 | `pip install matplotlib` |
| scipy | 四元数/旋转工具 | `pip install scipy` |
| h5py | HDF5 数据格式 | `pip install h5py` |

### 4. 最简仿真测试

```python
xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

for _ in range(100):
    mujoco.mj_step(model, data)
```

验证：球从 z=1.0 自由落体，100 步后高度下降，与理论值 h = ½gt² 吻合。

### 5. Viewer

```python
import mujoco.viewer   # 如果不报错，说明 3D 查看器可用
```

Linux 用户可能需要：`sudo apt install libgl1-mesa-glx libglew-dev`
