# 第 3 章 · 02 — 四元数深度教程

> **目标**: 从零理解四元数，掌握在 MuJoCo 中处理旋转的所有技巧。

## 为什么需要四元数？

| 旋转表示方式 | 参数量 | 万向锁 | 插值方便 | MuJoCo 默认 |
| :----------: | :----: | :----: | :------: | :---------: |
| 欧拉角       | 3      | 有     | 差       | 否          |
| 旋转矩阵     | 9      | 无     | 差       | 否          |
| 四元数       | 4      | 无     | 好       | **是**      |

- 欧拉角有**万向锁 (gimbal lock)** 问题
- 旋转矩阵有 9 个元素但只有 3 个自由度，冗余且难以约束正交性
- 四元数：4 个数、无万向锁、容易插值、MuJoCo 默认使用

### 格式约定

```
MuJoCo 四元数: [w, x, y, z]  ← w 在前
scipy  四元数: [x, y, z, w]  ← w 在后

⚠️ 两者顺序相反，转换时务必注意！
```

---

## 1. 四元数基础

四元数 `q = w + xi + yj + zk`，其中：

- w — 实部（标量）
- x, y, z — 虚部（向量）
- i, j, k — 虚数单位，满足 `i² = j² = k² = ijk = -1`

**单位四元数**（|q| = 1）表示三维旋转：

```
q = [ cos(θ/2),  sin(θ/2)·ax,  sin(θ/2)·ay,  sin(θ/2)·az ]
```

其中 (ax, ay, az) 是旋转轴（单位向量），θ 是旋转角度。

### 人话翻译

上面的公式说的是：**你只需要告诉我两件事，我就能用四元数表示任何旋转：**

1. **绕哪根轴转？** —— 用一个方向向量 (ax, ay, az) 表示
2. **转多少度？** —— 用角度 θ 表示

然后四元数的 4 个数字就这么算出来：

```
q = [ cos(θ/2),  sin(θ/2)·ax,  sin(θ/2)·ay,  sin(θ/2)·az ]
      ────────   ───────────   ───────────   ───────────
         w            x              y              z
```

| 四元数位置 | 值 | 人话 |
|:---:|:---:|:---|
| w（第 1 个数） | cos(θ/2) | 把角度砍一半，算 cos |
| x（第 2 个数） | sin(θ/2) × ax | 把角度砍一半，算 sin，乘上轴的 x |
| y（第 3 个数） | sin(θ/2) × ay | 同上，乘轴的 y |
| z（第 4 个数） | sin(θ/2) × az | 同上，乘轴的 z |

**举个具体例子 —— "绕 Z 轴转 90°"：**

```
已知：旋转轴 = (0, 0, 1)，角度 θ = 90°

w = cos(90°/2) = cos(45°) = 0.7071
x = sin(45°) × 0 = 0
y = sin(45°) × 0 = 0
z = sin(45°) × 1 = 0.7071

结果：q = [0.7071, 0, 0, 0.7071]  ← 和下方表格完全对得上
```

> **为什么角度要除以 2？** 四元数旋转向量时，向量被左右各乘了一次（q·v·q*），每次"贡献"半个角度，两次加起来刚好是完整的旋转角度。记住：**算四元数时，角度永远先除以 2**。

> **一句话总结：四元数 = `[cos(半角), sin(半角)×轴]`，就是把"绕哪根轴转多少度"编码成 4 个数字。**

### 角度-轴 → 四元数的转换代码

```python
def angle_axis_to_quat(angle_deg, axis):
    """角度-轴 → MuJoCo 格式四元数 [w, x, y, z]"""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)       # 归一化旋转轴
    half = np.radians(angle_deg) / 2.0       # 角度的一半（弧度）
    w = np.cos(half)
    xyz = np.sin(half) * axis
    return np.array([w, xyz[0], xyz[1], xyz[2]])
```

---

## 2. 常见旋转的四元数表示

| 旋转描述              | 四元数 [w, x, y, z]                    |
| :-------------------- | :-------------------------------------- |
| 单位旋转（无旋转）     | `[1.0000, 0.0000, 0.0000, 0.0000]`     |
| 绕 X 轴旋转 90°       | `[0.7071, 0.7071, 0.0000, 0.0000]`     |
| 绕 Y 轴旋转 90°       | `[0.7071, 0.0000, 0.7071, 0.0000]`     |
| 绕 Z 轴旋转 90°       | `[0.7071, 0.0000, 0.0000, 0.7071]`     |
| 绕 X 轴旋转 180°      | `[0.0000, 1.0000, 0.0000, 0.0000]`     |
| 绕 Y 轴旋转 180°      | `[0.0000, 0.0000, 1.0000, 0.0000]`     |
| 绕 Z 轴旋转 180°      | `[0.0000, 0.0000, 0.0000, 1.0000]`     |
| 绕 X 轴旋转 45°       | `[0.9239, 0.3827, 0.0000, 0.0000]`     |

### 记忆技巧

- **单位旋转** = `[1, 0, 0, 0]`（w=1，无虚部）
- **180° 旋转**：w=0，虚部就是旋转轴
- **90° 旋转**：w = cos(45°) ≈ 0.7071，虚部 = sin(45°)·轴 ≈ 0.7071·轴

---

## 3. 四元数乘法（组合旋转）

物理含义：**q1 ⊗ q2 表示先做 q2 旋转，再做 q1 旋转**。

### Hamilton 积公式

```python
def quat_multiply(q1, q2):
    """四元数乘法: q1 ⊗ q2，MuJoCo [w,x,y,z] 格式"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
```

### 示例

```
先绕 X 转 90°，再绕 Z 转 90°:
  q_z90 ⊗ q_x90 = [0.5, 0.5, 0.5, 0.5]

先绕 Z 转 90°，再绕 X 转 90°:
  q_x90 ⊗ q_z90 = [0.5, 0.5, -0.5, 0.5]

两者结果不同 → 旋转不可交换！
```

### MuJoCo API

```python
q_result = np.zeros(4)
mujoco.mju_mulQuat(q_result, q1, q2)  # q_result = q1 ⊗ q2
```

---

## 4. 四元数共轭与逆

### 共轭

```
q* = [w, -x, -y, -z]
```

将虚部取反。对于**单位四元数**，共轭 = 逆。

### 逆

```
q⁻¹ = q* / |q|²
```

### 性质

- `q ⊗ q⁻¹ = [1, 0, 0, 0]`（单位四元数）
- **q 和 -q 表示同一个旋转**（双覆盖性质）

```python
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_inverse(q):
    conj = quat_conjugate(q)
    return conj / np.dot(q, q)
```

### MuJoCo API

```python
mujoco.mju_negQuat(q_neg, q)  # 共轭 (对应反向旋转)
```

> `mju_negQuat` 名字虽然带 "neg"，但实际做的是**共轭**（conjugate）：`q = [w, x, y, z] → [w, -x, -y, -z]`，只有虚部取反，w 不变。MuJoCo 官方文档描述为 "Conjugate quaternion, corresponding to opposite rotation"。对于单位四元数，共轭 = 逆 = 反向旋转。

---

## 5. 用四元数旋转向量

将向量 v 用四元数 q 旋转：

```
v' = q ⊗ [0, v] ⊗ q*
```

步骤：把 3D 向量扩展为纯虚四元数 `[0, vx, vy, vz]`，左乘 q、右乘 q*，取结果的虚部。

```python
v = np.array([1.0, 0.0, 0.0])          # X 轴方向
q = angle_axis_to_quat(90, [0, 0, 1])  # 绕 Z 轴 90°

v_quat = np.array([0, v[0], v[1], v[2]])
v_rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
result = v_rotated[1:]  # → [0, 1, 0]  (X 轴变成 Y 轴)
```

### MuJoCo API

```python
v_result = np.zeros(3)
mujoco.mju_rotVecQuat(v_result, v, q)  # 一步完成
```

---

## 6. 四元数 ↔ 旋转矩阵

### 四元数 → 3×3 旋转矩阵

```
R = | 1-2(y²+z²)    2(xy-wz)      2(xz+wy)   |
    | 2(xy+wz)      1-2(x²+z²)    2(yz-wx)    |
    | 2(xz-wy)      2(yz+wx)      1-2(x²+y²)  |
```

### 示例：绕 Z 轴 90°

```
四元数: [0.7071, 0, 0, 0.7071]

旋转矩阵:
  [ 0.0000  -1.0000   0.0000]
  [ 1.0000   0.0000   0.0000]
  [ 0.0000   0.0000   1.0000]
```

### MuJoCo API

```python
# 四元数 → 矩阵
R = np.zeros(9)
mujoco.mju_quat2Mat(R, q)
R = R.reshape(3, 3)  # 行主序

# 矩阵 → 四元数
q_back = np.zeros(4)
mujoco.mju_mat2Quat(q_back, R.flatten())
```

---

## 7. 四元数 ↔ 欧拉角（需要 scipy）

由于 MuJoCo 和 scipy 的四元数顺序不同，转换时需要手动调整：

```python
# MuJoCo [w,x,y,z] → scipy [x,y,z,w]
q_scipy = [q_mj[1], q_mj[2], q_mj[3], q_mj[0]]

# scipy [x,y,z,w] → MuJoCo [w,x,y,z]
q_mj = [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]
```

### 转换函数

```python
from scipy.spatial.transform import Rotation

def mujoco_quat_to_euler(q_mj, seq="ZYX", degrees=True):
    """MuJoCo 四元数 → 欧拉角"""
    q_scipy = [q_mj[1], q_mj[2], q_mj[3], q_mj[0]]
    r = Rotation.from_quat(q_scipy)
    return r.as_euler(seq, degrees=degrees)

def euler_to_mujoco_quat(euler_deg, seq="ZYX"):
    """欧拉角 → MuJoCo 四元数"""
    r = Rotation.from_euler(seq, euler_deg, degrees=True)
    q_scipy = r.as_quat()
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
```

### 示例

| 描述           | 四元数 [w,x,y,z]                    | 欧拉角 ZYX (度)       |
| :------------- | :----------------------------------- | :-------------------- |
| 单位旋转       | `[ 1.000, 0.000, 0.000, 0.000]`     | `[  0.00,   0.00,   0.00]` |
| 绕 Z 轴 90°   | `[ 0.707, 0.000, 0.000, 0.707]`     | `[ 90.00,   0.00,   0.00]` |
| 绕 X 轴 45°   | `[ 0.924, 0.383, 0.000, 0.000]`     | `[  0.00,   0.00,  45.00]` |

---

## 8. 四元数球面线性插值 (SLERP)

**SLERP** = Spherical Linear intERPolation

### 为什么不能用线性插值 (LERP)？

- 线性插值后 |q| ≠ 1，不再是有效旋转
- 即使归一化 (NLERP)，角速度也不均匀
- **SLERP 保证等角速度插值**

### SLERP 公式

```
slerp(q1, q2, t) = sin((1-t)θ)/sin(θ) · q1 + sin(tθ)/sin(θ) · q2
```

其中 θ = arccos(q1·q2)，t ∈ [0, 1]。

### 实现

```python
def slerp_quat(q1, q2, t):
    """球面线性插值: q1 → q2，t ∈ [0, 1]"""
    dot = np.dot(q1, q2)

    # q 和 -q 表示同一旋转，选近的那条路径
    if dot < 0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # 几乎平行，退化为线性插值 + 归一化
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return w1 * q1 + w2 * q2
```

### 插值示例：无旋转 → 绕 Z 轴 180°

| t    | 四元数 [w,x,y,z]                      | &#124;q&#124; | 等效角度(°) |
| ---: | :------------------------------------- | ----: | ----------: |
| 0.00 | `[ 1.0000, 0.0000, 0.0000, 0.0000]`   | 1.0000 |        0.0 |
| 0.20 | `[ 0.9511, 0.0000, 0.0000, 0.3090]`   | 1.0000 |       36.0 |
| 0.50 | `[ 0.7071, 0.0000, 0.0000, 0.7071]`   | 1.0000 |       90.0 |
| 0.80 | `[ 0.3090, 0.0000, 0.0000, 0.9511]`   | 1.0000 |      144.0 |
| 1.00 | `[ 0.0000, 0.0000, 0.0000, 1.0000]`   | 1.0000 |      180.0 |

观察：|q| 始终 = 1，角度均匀增长（等角速度）。

---

## 9. 四元数归一化的重要性

### 问题

如果 |q| ≠ 1，四元数就不再表示有效旋转，会导致仿真异常。

### 实验

```python
# 正常
data.qpos[3:7] = [1, 0, 0, 0]     # |q| = 1.0

# 异常
data.qpos[3:7] = [2, 1, 1, 1]     # |q| = √7 ≈ 2.65
```

MuJoCo 在积分时会自动归一化，但：

1. **初始状态**的非归一化四元数会导致第一帧计算异常
2. **外部设置 qpos** 时一定要自己归一化
3. 数据管道中检测 |q| ≈ 1 是重要的质量校验

### 归一化方法

```python
# NumPy
q_normalized = q / np.linalg.norm(q)

# MuJoCo (in-place)
mujoco.mju_normalize4(q)

# 安全版本（处理零向量退化）
def normalize_quat(q):
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm
```

---

## 10. 两个旋转之间的距离

### 角度距离

利用内积衡量两个旋转的接近程度：

```
d(q1, q2) = 2·arccos(|q1·q2|)
```

返回 [0, π] 范围的角度（弧度）。|q1·q2| 取绝对值是因为 q 和 -q 表示同一旋转。

### 旋转差异

从 q1 到 q2 的旋转差异：

```
Δq = q2 ⊗ q1⁻¹
```

```python
def quat_distance(q1, q2):
    dot = np.clip(np.abs(np.dot(q1, q2)), 0, 1)
    return 2.0 * np.arccos(dot)

def quat_difference(q1, q2):
    return quat_multiply(q2, quat_inverse(q1))
```

### 示例

| 旋转对         | 角度距离(°) |
| :------------- | ----------: |
| 0° vs 30°     |       30.00 |
| 0° vs 90°     |       90.00 |
| 0° vs 180°    |      180.00 |
| 30° vs 90°    |       60.00 |
| 90° vs 180°   |       90.00 |

---

## 11. MuJoCo 内置四元数工具函数一览

| 函数                            | 说明                               |
| :------------------------------ | :--------------------------------- |
| `mju_mulQuat(res, q1, q2)`     | 四元数乘法 res = q1 ⊗ q2          |
| `mju_negQuat(res, q)`          | 四元数共轭 res = q*（反向旋转）     |
| `mju_rotVecQuat(res, v, q)`    | 用 q 旋转向量 v                    |
| `mju_quat2Mat(mat, q)`         | 四元数 → 3×3 旋转矩阵（行主序）    |
| `mju_mat2Quat(q, mat)`         | 3×3 旋转矩阵 → 四元数              |
| `mju_axisAngle2Quat(q, a, θ)` | 轴角 → 四元数                       |
| `mju_quat2Vel(vel, q, dt)`     | 四元数 → 角速度                     |
| `mju_subQuat(res, qa, qb)`     | 四元数差 res = qa ⊖ qb（3 维向量） |
| `mju_normalize4(q)`            | 归一化四元数（in-place）            |

### 常用示例

```python
# 轴角 → 四元数
q = np.zeros(4)
axis = np.array([0.0, 0.0, 1.0])
mujoco.mju_axisAngle2Quat(q, axis, np.radians(90))

# 四元数 → 角速度
vel = np.zeros(3)
mujoco.mju_quat2Vel(vel, q, 1.0)

# 两个旋转的差异（3 维向量）
sub = np.zeros(3)
mujoco.mju_subQuat(sub, q_a, q_b)
```

---

## 12. 在仿真中使用四元数

在 MuJoCo 中，free joint 的 `qpos[3:7]` 就是物体的朝向四元数。设置不同的初始四元数即可改变物体朝向：

```python
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 设置绕 Z 轴 45° 的朝向
data.qpos[3:7] = angle_axis_to_quat(45, [0, 0, 1])
mujoco.mj_forward(model, data)

# 读取世界坐标系中的旋转矩阵
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
xmat = data.xmat[body_id].reshape(3, 3)
xpos = data.xpos[body_id]
```

### `mj_forward` 的作用

调用 `mj_forward` 后，MuJoCo 会根据 `qpos` 计算出所有派生量（世界位置 `xpos`、旋转矩阵 `xmat` 等），这样就能看到四元数设置的实际效果。

---

## 总结

```
┌──────────────────────────────────────────────────────────────┐
│                    四元数速查                                  │
│                                                              │
│  MuJoCo 顺序: [w, x, y, z] ← 永远记住 w 在前                │
│  scipy  顺序: [x, y, z, w] ← w 在后                         │
│                                                              │
│  单位旋转: [1, 0, 0, 0]                                     │
│  绕轴 â 旋转 θ: [cos(θ/2), sin(θ/2)·â]                     │
│                                                              │
│  关键性质:                                                    │
│    q 和 -q 表示同一旋转 (双覆盖)                              │
│    必须保持 |q| = 1 (归一化)                                  │
│    旋转不可交换: q1⊗q2 ≠ q2⊗q1                               │
│                                                              │
│  核心操作:                                                    │
│    乘法   q1⊗q2         — 组合旋转 (先q2后q1)                │
│    共轭   q* = [w,-x,-y,-z] — 反转旋转                       │
│    旋转向量 v' = q⊗[0,v]⊗q*                                  │
│    SLERP                 — 等角速度球面插值                    │
│    距离   2·arccos(|q1·q2|)                                  │
│                                                              │
│  MuJoCo API:                                                 │
│    mju_mulQuat      — 四元数乘法                              │
│    mju_rotVecQuat   — 旋转向量                                │
│    mju_quat2Mat     — 转旋转矩阵                              │
│    mju_mat2Quat     — 矩阵转四元数                            │
│    mju_normalize4   — 归一化                                  │
│    mju_subQuat      — 旋转差异                                │
│    mju_axisAngle2Quat — 轴角转四元数                          │
└──────────────────────────────────────────────────────────────┘
```
