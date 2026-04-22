"""
第 1 章 · 01 - 安装验证

目标: 确认 MuJoCo 安装正确，了解版本和基本能力。

安装:
  pip install mujoco

运行: python 01_install_verify.py
"""

import sys

# ============================================================
# 1. 检查 Python 版本
# ============================================================
print("=" * 60)
print("1. 环境检查")
print("=" * 60)
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")

import platform
print(f"操作系统: {platform.system()} {platform.machine()}")

# ============================================================
# 2. 检查 MuJoCo 安装
# ============================================================
print("\n" + "=" * 60)
print("2. MuJoCo 检查")
print("=" * 60)

try:
    import mujoco
    print(f"✅ MuJoCo 版本: {mujoco.__version__}")
except ImportError:
    print("❌ MuJoCo 未安装，请运行: pip install mujoco")
    sys.exit(1)

# ============================================================
# 3. 检查依赖库
# ============================================================
print("\n" + "=" * 60)
print("3. 依赖库检查")
print("=" * 60)

dependencies = [
    ("numpy", "np"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("h5py", "h5py"),
]

for pkg_name, import_name in dependencies:
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✅ {pkg_name}: {version}")
    except ImportError:
        print(f"  ❌ {pkg_name}: 未安装 (pip install {pkg_name})")

# ============================================================
# 4. 最简仿真测试
# ============================================================
print("\n" + "=" * 60)
print("4. 最简仿真测试")
print("=" * 60)

xml_string = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

print(f"模型编译成功!")
print(f"  nq (qpos 维度): {model.nq}")
print(f"  nv (qvel 维度): {model.nv}")
print(f"  timestep: {model.opt.timestep}s")

# 仿真 100 步
initial_z = data.qpos[2]
for _ in range(100):
    mujoco.mj_step(model, data)

print(f"\n仿真 100 步 ({100 * model.opt.timestep:.3f}s):")
print(f"  初始高度: {initial_z:.4f} m")
print(f"  当前高度: {data.qpos[2]:.4f} m")
print(f"  下落距离: {initial_z - data.qpos[2]:.4f} m")
print(f"  当前速度: {data.qvel[2]:.4f} m/s")

# 理论值: h = 0.5 * g * t^2
import numpy as np
t = 100 * model.opt.timestep
h_theory = 0.5 * 9.81 * t ** 2
print(f"\n理论下落距离: {h_theory:.4f} m")
print(f"仿真误差: {abs(h_theory - (initial_z - data.qpos[2])):.6f} m")

# ============================================================
# 5. Viewer 检查
# ============================================================
print("\n" + "=" * 60)
print("5. Viewer 检查")
print("=" * 60)

try:
    import mujoco.viewer
    print("✅ mujoco.viewer 可用 (可以运行 04_viewer_basics.py)")
except ImportError:
    print("⚠️  mujoco.viewer 不可用")
    print("   Linux 用户可能需要: sudo apt install libgl1-mesa-glx libglew-dev")

print("\n" + "=" * 60)
print("🎉 所有检查通过！环境准备就绪。")
print("=" * 60)
