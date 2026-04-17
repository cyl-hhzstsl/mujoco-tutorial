"""
第 0 章 · 01 - NumPy 核心操作

目标: 掌握 NumPy 数组的创建、索引、运算，为后续处理 qpos 数据打基础。

Java 类比:
  - np.array  ≈ double[] 但支持向量化运算
  - np.zeros  ≈ new double[n]
  - 切片操作  ≈ Arrays.copyOfRange() 但更强大
  - 广播机制  ≈ 没有直接对应，这是 NumPy 的杀手级特性

运行: python 01_numpy_essentials.py
"""

import numpy as np

# ============================================================
# 1. 创建数组
# ============================================================
print("=" * 60)
print("1. 创建数组")
print("=" * 60)

# Java: double[] a = {1.0, 2.0, 3.0};
a = np.array([1.0, 2.0, 3.0])
print(f"一维数组: {a}")
print(f"  shape: {a.shape}")    # (3,)
print(f"  dtype: {a.dtype}")    # float64

# Java: double[][] m = new double[3][4];
m = np.zeros((3, 4))
print(f"\n零矩阵:\n{m}")
print(f"  shape: {m.shape}")    # (3, 4)

# 常用创建方式
print(f"\n全1数组:   {np.ones(5)}")
print(f"等差数列:  {np.arange(0, 1, 0.2)}")          # [0.0, 0.2, 0.4, 0.6, 0.8]
print(f"等分数列:  {np.linspace(0, 1, 5)}")           # [0.0, 0.25, 0.5, 0.75, 1.0]
print(f"随机数组:  {np.random.randn(5)}")              # 标准正态分布
print(f"单位四元数: {np.array([1.0, 0.0, 0.0, 0.0])}") # 后面会频繁用到

# ============================================================
# 2. 索引与切片
# ============================================================
print("\n" + "=" * 60)
print("2. 索引与切片")
print("=" * 60)

# 模拟一个 qpos 数组: 7(free joint) + 6(hinge joints) = 13 维
qpos = np.array([
    0.0, 0.0, 1.0,           # 位置 x, y, z
    1.0, 0.0, 0.0, 0.0,      # 四元数 w, x, y, z
    0.1, -0.3, 0.5, -0.2, 0.8, 0.0  # 6 个关节角度
])
print(f"qpos = {qpos}")
print(f"qpos 长度: {len(qpos)}")

# 基本索引 (Java: qpos[2])
print(f"\nz 坐标: qpos[2] = {qpos[2]}")

# 切片 (Java: Arrays.copyOfRange(qpos, 0, 3))
position = qpos[0:3]    # 或 qpos[:3]
quaternion = qpos[3:7]
joint_angles = qpos[7:]
print(f"位置:     qpos[:3]  = {position}")
print(f"四元数:   qpos[3:7] = {quaternion}")
print(f"关节角度: qpos[7:]  = {joint_angles}")

# 负索引 (Java 没有这个特性)
print(f"\n最后一个: qpos[-1] = {qpos[-1]}")
print(f"倒数三个: qpos[-3:] = {qpos[-3:]}")

# 二维数组索引
# 模拟轨迹数据: 100 帧，每帧 13 维 qpos
trajectory = np.random.randn(100, 13)
print(f"\n轨迹 shape: {trajectory.shape}")
print(f"第 0 帧:    {trajectory[0]}")           # 第一帧的完整 qpos
print(f"所有帧的 z: {trajectory[:, 2][:5]}...")  # 所有帧的第 3 个维度(z坐标)
print(f"前 5 帧的关节角度:\n{trajectory[:5, 7:]}")

# ============================================================
# 3. 向量化运算
# ============================================================
print("\n" + "=" * 60)
print("3. 向量化运算 (告别 for 循环)")
print("=" * 60)

# Java 写法 (慢):
# for (int i = 0; i < a.length; i++) { b[i] = a[i] * 2; }

# NumPy 写法 (快):
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"a = {a}")
print(f"a * 2 = {a * 2}")            # 每个元素 ×2
print(f"a + 10 = {a + 10}")          # 每个元素 +10
print(f"a ** 2 = {a ** 2}")          # 每个元素 平方
print(f"np.sin(a) = {np.sin(a)}")    # 每个元素 求 sin

# 弧度 ↔ 角度 转换 (机器人数据中极其常用)
angles_rad = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
angles_deg = np.rad2deg(angles_rad)
print(f"\n弧度: {angles_rad}")
print(f"角度: {angles_deg}")
print(f"转回: {np.deg2rad(angles_deg)}")

# 数组间运算
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"\na={a}, b={b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")            # 逐元素乘，不是矩阵乘
print(f"a @ b = {a @ b}")            # 点积 (1*4 + 2*5 + 3*6 = 32)
print(f"np.dot(a, b) = {np.dot(a, b)}")

# ============================================================
# 4. 统计函数
# ============================================================
print("\n" + "=" * 60)
print("4. 统计函数 (数据分析必备)")
print("=" * 60)

data = np.random.randn(1000, 6)  # 模拟 1000 帧 6 个关节的角度
print(f"data shape: {data.shape}")

# 全局统计
print(f"\n全局 mean: {data.mean():.4f}")
print(f"全局 std:  {data.std():.4f}")

# 按轴统计 (axis=0 表示沿时间轴, axis=1 表示沿关节轴)
means = data.mean(axis=0)  # 每个关节的平均值, shape=(6,)
stds = data.std(axis=0)    # 每个关节的标准差
mins = data.min(axis=0)
maxs = data.max(axis=0)

print(f"\n每个关节的统计:")
for i in range(6):
    print(f"  joint[{i}]: mean={means[i]:>7.3f}  "
          f"std={stds[i]:>6.3f}  "
          f"range=[{mins[i]:>7.3f}, {maxs[i]:>6.3f}]")

# ============================================================
# 5. Reshape 与转置
# ============================================================
print("\n" + "=" * 60)
print("5. Reshape 与转置")
print("=" * 60)

# 一维 → 二维
flat = np.arange(12)
print(f"flat: {flat}")
reshaped = flat.reshape(3, 4)
print(f"reshape(3,4):\n{reshaped}")

# -1 表示自动推断
auto = flat.reshape(4, -1)  # 4 行，列数自动算
print(f"reshape(4,-1):\n{auto}")

# 转置
print(f"\n转置:\n{reshaped.T}")
print(f"  原 shape: {reshaped.shape}")
print(f"  转 shape: {reshaped.T.shape}")

# ============================================================
# 6. 布尔索引 (数据过滤)
# ============================================================
print("\n" + "=" * 60)
print("6. 布尔索引 (数据过滤)")
print("=" * 60)

heights = np.array([0.95, 1.02, 0.30, 1.05, 0.88, 0.10, 1.01])
print(f"基座高度序列: {heights}")

# 找出高度 < 0.5 的帧 (机器人摔倒了)
fallen = heights < 0.5
print(f"是否摔倒:    {fallen}")
print(f"摔倒帧索引:  {np.where(fallen)[0]}")
print(f"摔倒帧高度:  {heights[fallen]}")
print(f"没摔倒的帧:  {heights[~fallen]}")

# ============================================================
# 7. 拼接与分割
# ============================================================
print("\n" + "=" * 60)
print("7. 拼接与分割")
print("=" * 60)

pos = np.array([0.0, 0.0, 1.0])
quat = np.array([1.0, 0.0, 0.0, 0.0])
joints = np.array([0.1, -0.3, 0.5])

# 拼接成 qpos
qpos = np.concatenate([pos, quat, joints])
print(f"拼接: {qpos}")
print(f"shape: {qpos.shape}")

# 多个轨迹拼接
traj1 = np.random.randn(50, 10)
traj2 = np.random.randn(30, 10)
combined = np.concatenate([traj1, traj2], axis=0)  # 沿时间轴拼接
print(f"\ntraj1 {traj1.shape} + traj2 {traj2.shape} = {combined.shape}")

# 分割
pos, quat, joints = np.split(qpos, [3, 7])
print(f"\n分割: pos={pos}, quat={quat}, joints={joints}")

# ============================================================
# 8. copy 的重要性
# ============================================================
print("\n" + "=" * 60)
print("8. copy 的重要性 (!!)")
print("=" * 60)

original = np.array([1.0, 2.0, 3.0])

# 切片是视图，不是拷贝！（和 Java 的 subList 类似）
view = original[:2]
view[0] = 999
print(f"修改 view 后, original = {original}")  # original 也变了!

# 要独立副本必须显式 copy
original = np.array([1.0, 2.0, 3.0])
safe_copy = original[:2].copy()
safe_copy[0] = 999
print(f"修改 copy 后, original = {original}")  # original 不变

print("\n⚠️  在 MuJoCo 中, data.qpos 是内部内存的引用。")
print("   保存数据时一定要 data.qpos.copy(), 否则后续 mj_step 会覆盖!")

print("\n✅ 第 01 节完成！")
