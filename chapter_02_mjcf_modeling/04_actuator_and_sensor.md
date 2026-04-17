# 第 2 章 · 04 — 执行器与传感器

> **目标**: 理解 MuJoCo 中所有执行器类型的工作原理和差异，掌握传感器的配置与读取。

---

## 1. 执行器类型

| 类型 | 公式 | 用途 |
| :--- | :--- | :--- |
| **motor** | `force = gain × ctrl` | 底层力控制、强化学习 |
| **position** | `force = kp × (ctrl - qpos) - kv × qvel` | 位置控制、关节空间运动 |
| **velocity** | `force = kv × (ctrl - qvel)` | 速度控制、传送带 |
| **general** | `force = gain × ctrl + bias[0] + bias[1]×qpos + bias[2]×qvel` | 自定义控制律 |

### XML 示例

```xml
<actuator>
  <motor name="act" joint="hinge" ctrlrange="-5 5"/>
  <position name="act" joint="hinge" kp="20" ctrlrange="-180 180"/>
  <velocity name="act" joint="hinge" kv="5" ctrlrange="-10 10"/>
</actuator>
```

---

## 2. 仿真对比

同一单摆，不同执行器施加相同指令后的行为：

- **motor**: 恒力矩，单摆持续加速或偏转
- **position**: 追踪目标角度，振荡后稳定
- **velocity**: 追踪目标角速度
- **general**: 行为取决于 gain/bias 参数

生成对比图 `actuator_comparison.png`。

---

## 3. 传感器类型

### 关节传感器

```xml
<jointpos name="angle" joint="hinge"/>
<jointvel name="velocity" joint="hinge"/>
```

### 执行器传感器

```xml
<actuatorpos name="act_pos" actuator="act"/>
<actuatorfrc name="act_force" actuator="act"/>
```

### IMU 传感器

```xml
<accelerometer name="imu_acc" site="imu_site"/>
<gyro name="imu_gyro" site="imu_site"/>
```

### 坐标系传感器

```xml
<framepos name="tip_pos" objtype="site" objname="tip"/>
<framequat name="tip_quat" objtype="site" objname="tip"/>
```

### 其他

```xml
<touch name="tip_touch" site="touch_site"/>
<jointlimitfrc name="limit_frc" joint="hinge"/>
```

### 读取传感器数据

```python
for s in range(model.nsensor):
    dim = model.sensor_dim[s]
    adr = model.sensor_adr[s]
    values = data.sensordata[adr:adr + dim]
```

---

## 4. 位置伺服阶跃响应

### kp 的影响（damping 固定）

- kp 越大 → 响应越快，但可能振荡
- kp 越小 → 响应慢，平滑

### 阻尼的影响（kp 固定）

- 阻尼越大 → 振荡越少，但响应变慢
- 阻尼越小 → 快但可能不稳定

最佳参数取决于具体应用场景。

---

## 总结

```
执行器:
┌─────────────┬───────────────────────────────────┐
│ motor       │ 直接力矩控制，最底层               │
│ position    │ 内置 PD，输入目标角度              │
│ velocity    │ 速度跟踪，输入目标角速度            │
│ general     │ 自定义 gain + bias，最灵活          │
└─────────────┴───────────────────────────────────┘

传感器:
┌─────────────────┬──────────────────────────────┐
│ jointpos/vel    │ 关节状态（最常用）              │
│ actuatorfrc     │ 执行器实际输出力矩              │
│ accelerometer   │ 线加速度（IMU）                │
│ gyro            │ 角速度（IMU）                  │
│ framepos/quat   │ 刚体/site 的位姿              │
│ touch           │ 接触力传感器                   │
└─────────────────┴──────────────────────────────┘
```
