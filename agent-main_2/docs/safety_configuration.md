# 机械臂运动安全配置

当前安全链路分为两层：

1. `SoftwareSafetyMonitor` 在技能执行前检查工作空间、单步平移、线速度、单位四元数、最短旋转路径、单次旋转角和累计旋转角，并在执行期间检查关节位置、速度、加速度和力矩遥测。
2. `ActionChunkGuard` 在 Pi05 action chunk 进入机械臂控制器前检查动作维度、有限数值、每维上下限、相邻动作变化、累计变化和 chunk 长度。

这两层是底层控制器安全功能的补充，不能替代机械臂厂商提供的关节硬限位、碰撞检测、急停和安全控制器。

## 真机默认关闭

仓库中的 `examples/safety_profile.example.json` 仅用于展示字段，里面的数值不是任何真实机械臂的认证参数，两个 `hardware_approved` 都是 `false`。保持该值时，真机运行和 Pi05 真机下发会失败关闭。

接入真机前，需要从机械臂说明书、控制器配置、末端工具参数和 Pi05 训练数据中确认：

- 每个关节允许的角度、速度、加速度和力矩；
- 机器人基座坐标系下允许进入的 XYZ 工作空间；
- Pi05 action 每一维的真实含义、单位，以及它是绝对值还是增量；
- 控制周期、action chunk 最大长度和可接受的单周期变化；
- 真机控制器能够按最短姿态路径执行，并能在 Agent 调用 `stop()` 后及时停止。

确认并经过负责人审核后，复制示例配置，填写真实数值并将两个 `hardware_approved` 改为 `true`。

## 加载方式

```python
from robot_agent import (
    ActionChunkGuard,
    SoftwareSafetyMonitor,
    build_agent_runtime,
)
from robot_agent.gateway import Pi05ServiceGateway
from robot_agent.safety_config import load_safety_profiles

profiles = load_safety_profiles("robot_safety_profile.json")
monitor = SoftwareSafetyMonitor(limits=profiles.motion)
gateway = Pi05ServiceGateway(
    robot=real_action_chunk_controller,
    action_guard=ActionChunkGuard(profiles.policy_action),
)

runtime = build_agent_runtime(
    planner,
    controller=real_primitive_controller,
    perception=real_perception,
    policies=real_policies,
    verifier=physical_verifier,
    safety_monitor=monitor,
    dry_run=False,
    hardware_mode=True,
)
```

真机 action controller 还必须实现 `get_action_state()`，返回与 Pi05 单个 action 相同语义和维度的当前值。对于绝对关节角 action，没有当前参考值时系统不会下发。

## 关于 360 度旋转

四元数只描述最终姿态，旋转 360 度后的姿态与旋转 0 度相同。因此系统同时采用以下约束：

- 只允许 `rotation_path="shortest"`；
- 限制相邻姿态的最短夹角和整次任务的累计夹角；
- 对 action chunk 累加每个关节的实际变化，而不是只比较最后一个 action；
- 真机控制器仍需保证轨迹规划器确实执行最短路径，并继续应用机械臂原生关节限位。
