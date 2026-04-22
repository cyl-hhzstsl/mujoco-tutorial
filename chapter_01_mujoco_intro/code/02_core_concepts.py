"""
第 1 章 · 02 - MuJoCo 核心概念

目标: 深入理解 MjModel 和 MjData 的关系，掌握仿真循环。

Java 类比:
  MjModel  ≈ Class 定义 (编译后不可变的结构)
  MjData   ≈ Object 实例 (运行时可变的状态)
  mj_step  ≈ tick() / update() 方法
  mj_forward ≈ 只计算不推进时间 (用于渲染/查询)

运行: python 02_core_concepts.py
"""

import mujoco
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ball_drop.xml")

# ============================================================
# 1. MjModel — 不可变的模型结构
# ============================================================
print("=" * 60)
print("1. MjModel — 模型结构 (编译后不可变)")
print("=" * 60)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)

print("【尺寸信息】")
print(f"  nq   = {model.nq:<4}  (qpos 的维度)")
print(f"  nv   = {model.nv:<4}  (qvel 的维度)")
print(f"  nu   = {model.nu:<4}  (ctrl 控制输入的维度)")
print(f"  nbody = {model.nbody:<4} (刚体数量, 含 world)")
print(f"  njnt  = {model.njnt:<4} (关节数量)")
print(f"  ngeom = {model.ngeom:<4} (几何体数量)")
print(f"  nsite = {model.nsite:<4} (站点数量)")

print("\n【物理选项】")
print(f"  timestep = {model.opt.timestep}s")
print(f"  gravity  = {model.opt.gravity}")

print("\n【Body 列表】")
for i in range(model.nbody):
    name = model.body(i).name or "(world)"
    parent = model.body_parentid[i]
    print(f"  body[{i}]: '{name}', parent={parent}")

print("\n【Joint 列表】")
type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
for i in range(model.njnt):
    name = model.joint(i).name
    jtype = type_names[model.jnt_type[i]]
    adr = model.jnt_qposadr[i]
    print(f"  joint[{i}]: '{name}', type={jtype}, qpos_adr={adr}")

print("\n【Geom 列表】")
geom_types = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid",
              5: "cylinder", 6: "box", 7: "mesh"}
for i in range(model.ngeom):
    name = model.geom(i).name
    gtype = geom_types.get(model.geom_type[i], f"type_{model.geom_type[i]}")
    body = model.geom_bodyid[i]
    print(f"  geom[{i}]: '{name}', type={gtype}, body={body}")

# ============================================================
# 2. MjData — 可变的运行时状态
# ============================================================
print("\n" + "=" * 60)
print("2. MjData — 运行时状态 (每步都在变)")
print("=" * 60)

data = mujoco.MjData(model)

print("【核心状态量】")
print(f"  data.time  = {data.time:.4f}s  (仿真时间)")
print(f"  data.qpos  = {data.qpos}  (广义位置)")
print(f"  data.qvel  = {data.qvel}  (广义速度)")
print(f"  data.ctrl  = {data.ctrl}  (控制输入)")

# mj_forward: 根据当前 qpos 计算所有派生量（不推进时间）
mujoco.mj_forward(model, data)

print("\n【派生量 (mj_forward 计算)】")
for i in range(model.nbody):
    name = model.body(i).name or "world"
    pos = data.xpos[i]
    print(f"  body '{name}' 世界位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

# ============================================================
# 3. 仿真循环: mj_step
# ============================================================
print("\n" + "=" * 60)
print("3. 仿真循环 — mj_step")
print("=" * 60)

mujoco.mj_resetData(model, data)
print(f"重置后: time={data.time:.4f}, z={data.qpos[2]:.4f}")

print("\n逐步仿真 (每 50 步打印一次):")
print(f"{'步数':>6} {'时间(s)':>8} {'高度z(m)':>10} {'速度vz(m/s)':>12}")
print("-" * 40)

for step in range(501):
    if step % 50 == 0:
        print(f"{step:>6} {data.time:>8.3f} {data.qpos[2]:>10.4f} {data.qvel[2]:>12.4f}")
    mujoco.mj_step(model, data)

print("\n观察:")
print("  - 球从 z=2.0 开始自由落体")
print("  - 碰到地面 (z≈0.1, 球半径=0.1) 后弹起")
print("  - 每次弹跳高度递减 (能量损耗)")

# ============================================================
# 4. mj_step vs mj_forward
# ============================================================
print("\n" + "=" * 60)
print("4. mj_step vs mj_forward")
print("=" * 60)

mujoco.mj_resetData(model, data)
original_qpos = data.qpos.copy()

# mj_forward: 只计算，不推进
mujoco.mj_forward(model, data)
print(f"mj_forward 后:")
print(f"  time = {data.time:.4f} (没变)")
print(f"  qpos = {data.qpos}")
print(f"  qpos 变了吗: {not np.allclose(data.qpos, original_qpos)}")

# mj_step: 计算 + 推进
mujoco.mj_step(model, data)
print(f"\nmj_step 后:")
print(f"  time = {data.time:.4f} (推进了 {model.opt.timestep}s)")
print(f"  qpos = {data.qpos}")
print(f"  qpos 变了吗: {not np.allclose(data.qpos, original_qpos)}")

print("""
总结:
  mj_forward: 根据 qpos/qvel 计算所有派生量（位姿、接触等）
              适用于: 设置 qpos 后更新渲染、计算传感器值
              不推进时间，不做动力学积分

  mj_step:    完整仿真一步 = 碰撞检测 + 约束求解 + 数值积分
              适用于: 物理仿真
              推进 time += timestep
""")

# ============================================================
# 5. 重置状态
# ============================================================
print("=" * 60)
print("5. 重置状态的几种方式")
print("=" * 60)

# 方式 1: 重置到默认状态
mujoco.mj_resetData(model, data)
print(f"mj_resetData: qpos = {data.qpos}, time = {data.time}")

# 方式 2: 手动设置 qpos
data.qpos[2] = 5.0  # 设到 5 米高
mujoco.mj_forward(model, data)
print(f"手动设 z=5: qpos = {data.qpos}")
print(f"  ball 世界位置: {data.xpos[1]}")

# 方式 3: 从 qpos0 恢复
data.qpos[:] = model.qpos0
mujoco.mj_forward(model, data)
print(f"从 qpos0 恢复: qpos = {data.qpos}")

print("\n✅ 第 02 节完成！")
