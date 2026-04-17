"""
模型检查器 (Model Inspector)

功能: 加载 MuJoCo 模型并提取结构化信息，包括关节、执行器、Body 层级等。
用途: 快速了解一个 .xml 模型的结构，数据管道中的 Schema 提取。

使用:
  inspector = ModelInspector.load_from_string(xml_string)
  inspector.print_summary()
  inspector.print_joints()
  qpos_map = inspector.get_qpos_map()
"""

import mujoco
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# 关节类型名称映射
_JOINT_TYPE_NAMES = {
    0: "free",
    1: "ball",
    2: "slide",
    3: "hinge",
}

# 关节类型对应的 qpos 维度
_JOINT_QPOS_DIM = {
    "free": 7,   # 3 pos + 4 quat
    "ball": 4,   # 4 quat
    "slide": 1,
    "hinge": 1,
}


@dataclass
class JointInfo:
    """单个关节的详细信息。"""
    index: int
    name: str
    type_name: str
    qpos_start: int
    qpos_dim: int
    qvel_start: int
    qvel_dim: int
    limited: bool
    range_low: Optional[float]
    range_high: Optional[float]
    damping: float
    body_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "type": self.type_name,
            "qpos_start": self.qpos_start,
            "qpos_dim": self.qpos_dim,
            "qvel_start": self.qvel_start,
            "qvel_dim": self.qvel_dim,
            "limited": self.limited,
            "range": [self.range_low, self.range_high] if self.limited else None,
            "damping": self.damping,
            "body": self.body_name,
        }


@dataclass
class ActuatorInfo:
    """单个执行器的详细信息。"""
    index: int
    name: str
    joint_name: str
    ctrl_range_low: float
    ctrl_range_high: float
    gear: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "joint": self.joint_name,
            "ctrl_range": [self.ctrl_range_low, self.ctrl_range_high],
            "gear": self.gear,
        }


@dataclass
class BodyInfo:
    """单个 Body 的信息。"""
    index: int
    name: str
    parent_index: int
    parent_name: str
    mass: float
    n_geoms: int
    n_joints: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "parent": self.parent_name,
            "mass": self.mass,
            "n_geoms": self.n_geoms,
            "n_joints": self.n_joints,
        }


class ModelInspector:
    """
    MuJoCo 模型结构检查器。

    提供模型结构的完整视图:
      - 关节列表（类型、qpos 索引、范围）
      - 执行器列表（范围、增益）
      - Body 层级树
      - qpos 映射表
    """

    def __init__(self, model: mujoco.MjModel):
        self._model = model
        self._joints: List[JointInfo] = []
        self._actuators: List[ActuatorInfo] = []
        self._bodies: List[BodyInfo] = []
        self._parse()

    # ---- 工厂方法 ----

    @classmethod
    def load(cls, xml_path: str) -> "ModelInspector":
        """从文件路径加载模型。"""
        model = mujoco.MjModel.from_xml_path(xml_path)
        return cls(model)

    @classmethod
    def load_from_string(cls, xml_string: str) -> "ModelInspector":
        """从 XML 字符串加载模型。"""
        model = mujoco.MjModel.from_xml_string(xml_string)
        return cls(model)

    # ---- 解析 ----

    def _parse(self) -> None:
        """解析模型中的所有结构信息。"""
        self._parse_joints()
        self._parse_actuators()
        self._parse_bodies()

    def _parse_joints(self) -> None:
        m = self._model
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
            type_id = m.jnt_type[j]
            type_name = _JOINT_TYPE_NAMES.get(type_id, f"unknown({type_id})")
            qpos_start = m.jnt_qposadr[j]
            qpos_dim = _JOINT_QPOS_DIM.get(type_name, 1)
            qvel_start = m.jnt_dofadr[j]
            qvel_dim = 6 if type_name == "free" else (3 if type_name == "ball" else 1)
            limited = bool(m.jnt_limited[j])
            range_low = float(m.jnt_range[j, 0]) if limited else None
            range_high = float(m.jnt_range[j, 1]) if limited else None
            damping = float(m.dof_damping[qvel_start])
            body_id = m.jnt_bodyid[j]
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"

            self._joints.append(JointInfo(
                index=j, name=name, type_name=type_name,
                qpos_start=qpos_start, qpos_dim=qpos_dim,
                qvel_start=qvel_start, qvel_dim=qvel_dim,
                limited=limited, range_low=range_low, range_high=range_high,
                damping=damping, body_name=body_name,
            ))

    def _parse_actuators(self) -> None:
        m = self._model
        for a in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator_{a}"
            trntype = m.actuator_trntype[a]
            joint_name = ""
            if trntype == 0 and m.actuator_trnid[a, 0] >= 0:
                jid = m.actuator_trnid[a, 0]
                joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
            ctrl_low = float(m.actuator_ctrlrange[a, 0])
            ctrl_high = float(m.actuator_ctrlrange[a, 1])
            gear = float(m.actuator_gear[a, 0])

            self._actuators.append(ActuatorInfo(
                index=a, name=name, joint_name=joint_name,
                ctrl_range_low=ctrl_low, ctrl_range_high=ctrl_high,
                gear=gear,
            ))

    def _parse_bodies(self) -> None:
        m = self._model
        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
            parent_id = m.body_parentid[b]
            parent_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, parent_id) or f"body_{parent_id}"
            mass = float(m.body_mass[b])

            n_geoms = 0
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b:
                    n_geoms += 1

            n_joints = 0
            for j in range(m.njnt):
                if m.jnt_bodyid[j] == b:
                    n_joints += 1

            self._bodies.append(BodyInfo(
                index=b, name=name, parent_index=parent_id,
                parent_name=parent_name, mass=mass,
                n_geoms=n_geoms, n_joints=n_joints,
            ))

    # ---- 打印方法 ----

    def print_summary(self) -> None:
        """打印模型概要（一行总览）。"""
        m = self._model
        model_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, 0) or "unnamed"
        total_mass = sum(b.mass for b in self._bodies)
        print(f"模型概要: nq={m.nq}, nv={m.nv}, "
              f"关节={m.njnt}, 执行器={m.nu}, "
              f"Body={m.nbody}, Geom={m.ngeom}, "
              f"总质量={total_mass:.2f}kg, "
              f"dt={m.opt.timestep}s")

    def print_joints(self) -> None:
        """打印所有关节信息。"""
        print(f"\n关节列表 ({len(self._joints)} 个):")
        header = (f"  {'#':<4} {'名称':<16} {'类型':<8} "
                  f"{'qpos_idx':<10} {'dim':<6} {'范围':<24} {'阻尼':<8} {'所属Body'}")
        print(header)
        print(f"  {'-' * 90}")
        for j in self._joints:
            if j.limited:
                rng = f"[{np.degrees(j.range_low):>7.1f}°, {np.degrees(j.range_high):>7.1f}°]"
            else:
                rng = "无限制"
            print(f"  {j.index:<4} {j.name:<16} {j.type_name:<8} "
                  f"{j.qpos_start:<10} {j.qpos_dim:<6} {rng:<24} "
                  f"{j.damping:<8.2f} {j.body_name}")

    def print_actuators(self) -> None:
        """打印所有执行器信息。"""
        print(f"\n执行器列表 ({len(self._actuators)} 个):")
        header = f"  {'#':<4} {'名称':<16} {'关联关节':<16} {'控制范围':<24} {'增益'}"
        print(header)
        print(f"  {'-' * 70}")
        for a in self._actuators:
            rng = f"[{a.ctrl_range_low:>7.1f}, {a.ctrl_range_high:>7.1f}]"
            print(f"  {a.index:<4} {a.name:<16} {a.joint_name:<16} "
                  f"{rng:<24} {a.gear:.1f}")

    def print_bodies(self) -> None:
        """以树形结构打印 Body 层级。"""
        print(f"\nBody 层级树 ({len(self._bodies)} 个):")

        children_map: Dict[int, List[int]] = {}
        for b in self._bodies:
            pid = b.parent_index
            if pid not in children_map:
                children_map[pid] = []
            if b.index != 0:
                children_map[pid].append(b.index)

        def _print_tree(body_idx: int, prefix: str = "", is_last: bool = True) -> None:
            body = self._bodies[body_idx]
            connector = "└── " if is_last else "├── "
            joint_info = ""
            for j in self._joints:
                if j.body_name == body.name:
                    joint_info += f" [{j.name}: {j.type_name}]"

            if body_idx == 0:
                print(f"  {body.name} (world, {body.mass:.2f}kg)")
            else:
                print(f"  {prefix}{connector}{body.name} "
                      f"({body.mass:.2f}kg, {body.n_geoms}g){joint_info}")

            children = children_map.get(body_idx, [])
            for i, child_idx in enumerate(children):
                new_prefix = prefix + ("    " if is_last else "│   ")
                _print_tree(child_idx, new_prefix, i == len(children) - 1)

        _print_tree(0)

    # ---- 数据查询方法 ----

    def get_qpos_map(self) -> Dict[str, Tuple[int, int, str]]:
        """
        返回关节名称到 qpos 信息的映射。

        返回:
          dict: {joint_name: (start_idx, dim, type_name)}
        """
        return {
            j.name: (j.qpos_start, j.qpos_dim, j.type_name)
            for j in self._joints
        }

    def get_joint_by_qpos_index(self, idx: int) -> Optional[JointInfo]:
        """
        根据 qpos 索引反查关节信息。

        用于数据分析时: 知道 qpos[5] 对应哪个关节。
        """
        for j in self._joints:
            if j.qpos_start <= idx < j.qpos_start + j.qpos_dim:
                return j
        return None

    def to_dict(self) -> Dict[str, Any]:
        """导出全部信息为字典（JSON 序列化友好）。"""
        m = self._model
        return {
            "summary": {
                "nq": m.nq,
                "nv": m.nv,
                "njnt": m.njnt,
                "nu": m.nu,
                "nbody": m.nbody,
                "ngeom": m.ngeom,
                "timestep": m.opt.timestep,
                "gravity": m.opt.gravity.tolist(),
            },
            "joints": [j.to_dict() for j in self._joints],
            "actuators": [a.to_dict() for a in self._actuators],
            "bodies": [b.to_dict() for b in self._bodies],
            "qpos_map": {
                name: {"start": s, "dim": d, "type": t}
                for name, (s, d, t) in self.get_qpos_map().items()
            },
        }

    @property
    def model(self) -> mujoco.MjModel:
        """访问底层 MjModel 对象。"""
        return self._model

    @property
    def joints(self) -> List[JointInfo]:
        return list(self._joints)

    @property
    def actuators(self) -> List[ActuatorInfo]:
        return list(self._actuators)

    @property
    def bodies(self) -> List[BodyInfo]:
        return list(self._bodies)


# ============================================================
# 独立运行时的演示
# ============================================================

_DEMO_XML = """
<mujoco model="inspector_demo">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="sphere" size="0.05" mass="0"/>
      <body name="link1" pos="0 0 0">
        <joint name="shoulder" type="hinge" axis="0 1 0"
               limited="true" range="-180 180" damping="1.0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1.0"/>
        <body name="link2" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0"
                 limited="true" range="-150 150" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.018" mass="0.7"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-50 50"/>
    <motor name="elbow_motor" joint="elbow" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""


def main():
    print("=" * 60)
    print("ModelInspector 演示")
    print("=" * 60)

    inspector = ModelInspector.load_from_string(_DEMO_XML)

    inspector.print_summary()
    inspector.print_joints()
    inspector.print_actuators()
    inspector.print_bodies()

    print("\nqpos 映射表:")
    for name, (start, dim, type_name) in inspector.get_qpos_map().items():
        print(f"  {name}: qpos[{start}:{start + dim}] ({type_name})")

    print("\nqpos 反查:")
    for idx in range(inspector.model.nq):
        j = inspector.get_joint_by_qpos_index(idx)
        if j:
            print(f"  qpos[{idx}] → {j.name} ({j.type_name})")

    print("\nto_dict() 输出（部分）:")
    d = inspector.to_dict()
    print(f"  summary: {d['summary']}")
    print(f"  joints: {len(d['joints'])} 个")
    print(f"  qpos_map: {d['qpos_map']}")


if __name__ == "__main__":
    main()
