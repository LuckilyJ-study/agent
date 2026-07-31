# 通用机械臂 Agent

本项目实现了一个面向机械臂任务的闭环 Agent 框架。用户只需要描述目标，Qwen 负责把目标拆成高层技能；本地程序负责校验、调度、选择执行器、监控、恢复和记录任务状态。大模型不直接控制电机，也不能绕过本地安全规则。

## 推荐工作流

```text
用户输入自然语言任务
        ↓
Qwen Planner：生成高层技能计划
        ↓
技能白名单与计划结构校验
        ↓
Scheduler：选择当前可执行步骤
        ↓
动作前观察；真机由 PhysicalTargetGate 检查目标是否可见且已定位
        ↓
ExecutorRouter：由本地配置选择执行方式
        ├─ policy → PolicyExecutor → 训练策略 / Pi05 Gateway → 机械臂
        └─ robot  → RobotPrimitiveExecutor → 机械臂控制接口
        ↓
感知 Provider / YOLO-World-S → StructuredActionMonitor
        ↓
正常：更新 TaskMemory，继续下一步
异常：停止、重观察、局部重试或请求 Qwen 后缀重规划
```

系统包含两个相互配合的循环：

- 任务循环：规划 → 校验 → 调度 → 路由 → 执行 → 更新记忆。
- 监控循环：相机帧 → 检测/跟踪 → Monitor → 正常或异常决策。

当前相机与 YOLO-World-S 尚未接入，因此第二个循环在真实运行中仍使用占位感知；仓库提供了结构化 Monitor 和脚本化感知，便于离线测试异常处理。

## Qwen 的职责边界

Qwen 只输出任务目标和高层技能，例如：

```json
{
  "goal": {
    "description": "把面团放到托盘",
    "conditions": []
  },
  "steps": [
    {
      "step_id": 1,
      "action_type": "pick",
      "target": "dough",
      "expected_result": "dough is held",
      "status": "pending",
      "parameters": {}
    },
    {
      "step_id": 2,
      "action_type": "place",
      "target": "tray",
      "expected_result": "dough is on tray",
      "status": "pending",
      "parameters": {}
    }
  ]
}
```

Qwen 不负责：

- 输出 XYZ、末端位姿、关节角或逐周期控制 Action；
- 选择 `robot`、`policy`、Pi05 或具体策略版本；
- 为场景中不存在的物体、坐标或能力编造数据；
- 直接向真实机械臂发送命令。

执行坐标和底层动作必须来自可信来源：

1. 相机感知、目标定位和跟踪结果；
2. 真机标定后的坐标系、命名位姿和可交互区域；
3. 已训练技能或 Pi05 策略内部生成的动作；
4. 用户明确给出的、经过边界检查的小范围相对运动。

因此，`pick(dough)` 应交给抓取策略处理接近、闭爪和抬升，而不是让 Qwen 猜测一串坐标。Router 的选择结果以本地 `CapabilityRegistry` 和 `PolicyRegistry` 为准。

## 模拟模式的开放世界

`run_agent_simulation.py` 在没有提供 `--scene-file` 时允许开放世界模拟：能够
匹配内置演示场景时先复用其中的可信实体与技能合同，并为计划中额外出现的新
target 注册虚拟实体；完全不匹配时则从空的 `open_world_simulation` 启动。例如：

```powershell
python .\run_agent_simulation.py --task "抓小球"
```

此时的处理顺序是：

```text
空的开放世界语义场景
  → Qwen 从用户原话中提取高层技能和 target
  → 技能白名单与计划结构校验
  → VirtualEntityRegistry 为未知 target 分配安全的本地实体 ID
  → SceneSkillPlanner 推导通用 pick/place/inspect 语义
  → Dry-run Robot/Policy 执行软件流程
```

虚拟实体只是模拟世界中的**语义对象记录**，例如“这是一个可抓目标，用户称它为
小球”。它只包含本地 ID、别名、语义类型和初始符号位置：

- 不包含 XYZ、姿态、关节角、检测框或轨迹；
- 不证明现实环境中真的存在该物体；
- 不会绕过技能白名单，也不会自动新增 Policy、Robot Primitive 或
  `skill_contract`；
- 不会把计划验证时模拟出来的最终状态提前写入当前世界。

因此，`pick(小球)`、`pick(积木) → place(盒子)` 可以使用现有通用语义进行
dry-run；但“打开一种未知锁扣”不会因为注册了“锁扣”这个虚拟实体就获得新的
`manipulate` 能力，缺少本地技能合同或 Policy 时仍会拒绝。显式
`--scene-file` 默认按文件声明的闭合场景校验，不会自动为漏写的目标补实体。

## 当前已经实现

| 模块 | 当前能力 |
| --- | --- |
| Planner | Qwen API 结构化规划、可选的独立计划复核、非法计划修复和失败点后缀重规划 |
| 语义落地 | `SceneSkillPlanner` 用可复用技能合同检查前置条件、推导可信状态变化，并拒绝/修复错误顺序 |
| 开放世界模拟 | 自动场景可复用已知实体并登记 Qwen 计划中的新虚拟 target；不生成坐标或能力 |
| 技能校验 | 运行时技能白名单、字段/schema 校验、依赖检查、执行器防伪和参数边界检查 |
| Scheduler | 按依赖和状态选择下一步，不生成电机命令 |
| ExecutorRouter | 将基础动作路由到机械臂控制器，将复杂技能路由到已注册策略 |
| 执行层 | `RobotPrimitiveExecutor`、`PolicyExecutor`、干运行机械臂和干运行策略后端 |
| 策略接口 | `PolicyRegistry`、策略元数据检查、Pi05 本地/服务 Gateway 接口与测试骨架 |
| 真机动作前门禁 | `PhysicalTargetGate` 在 Router 前检查感知可用性、目标新鲜度、二维框和基座系三维定位 |
| Monitor | 支持 `TARGET_LOST`、`GRASP_FAILED`、`OBJECT_DROPPED`、碰撞风险和硬件故障等结构化判断 |
| 恢复 | 停止当前执行、重观察、有限重试、安全停止、连续失败后只重规划未完成后缀 |
| 任务状态 | `TaskMemory`、事件时间线、步骤尝试记录、世界状态、可选 JSON 持久化 |
| 测试 | 无机械臂、无摄像头、无模型服务时的确定性模拟和故障注入测试 |

默认 Qwen 模拟入口只向模型公开当前运行时允许的高层技能。白名单外的动作会在进入执行器之前被拒绝。

## 当前边界

- `run_agent_simulation.py` 使用 `DryRunRobotController`，不会向真实机械臂发送命令。
- 真机控制器、机械臂标定数据和厂商 SDK 尚未接入。
- YOLO-World-S、高速相机、目标跟踪和真实视觉定位尚未接入；当前默认是 `NullPerceptionProvider`。
- `StructuredActionMonitor` 已能消费 YOLO/跟踪器风格的结构化帧，但现在主要由测试中的 `ScriptedPerceptionProvider` 提供数据。
- Pi05 Gateway 和策略注册机制已经留好接口，但模拟入口没有加载真实 Pi05 权重或推理服务。
- 模拟入口可以把计划中的未知 `pick/place/inspect` 目标登记为虚拟语义实体；真机禁止把这种登记当作目标存在或定位证据。
- 无论是否开放世界，Qwen 都只能使用技能白名单中的能力；未知 `manipulate`、未注册 Policy 或缺失技能合同仍会失败。
- dry-run 输出 `Status: completed` 只表示软件命令链、符号状态或已落地计划执行完毕，不代表真实物体已经被抓起或放置成功。没有物理感知时，系统会明确显示 `Physical verification: NOT AVAILABLE`。

## 真机感知契约与 PhysicalTargetGate

`hardware_mode=True` 时，感知 Provider 必须显式满足以下契约，否则运行时在
组装阶段直接拒绝启动：

```python
hardware_ready = True
localization_modes = frozenset({"bbox_2d", "robot_base_xyz"})

def configure_targets(labels): ...
def observe() -> dict: ...
```

同一模式也拒绝 `dry_run=True`、`DryRunRobotController`、Dry-run Policy 和
占位 Action Verifier；真机 Controller 还必须显式声明
`hardware_ready=True`，并注册真实 Policy 与物理 Monitor。

`configure_targets(labels)` 用于把当前步骤的实体 ID、名称或别名配置给
YOLO-World-S 类检测器。`localization_modes` 必须同时包含 `bbox_2d` 和
`robot_base_xyz`；二维检测框本身不能作为机械臂坐标。

对 `pick`、`place`、`manipulate` 和目标式 `move_to`，`PhysicalTargetGate`
要求最新观察中存在与 target 匹配、置信度达标且时间戳新鲜的检测，并同时具有：

```json
{
  "bbox_xyxy": [10, 20, 100, 120],
  "position_xyz_m": [0.42, 0.08, 0.16],
  "coordinate_frame": "robot_base"
}
```

也就是说，真机目标动作必须同时拿到 **fresh YOLO/provider bbox** 和经过标定的
**`robot_base` XYZ**，二者缺一不可；三维值还必须是有限数并处于配置的机械臂
工作空间范围内。

检查发生在 `ExecutorRouter.route()` 和任何机械臂/Policy 调用之前。通过后，
门禁只把精简的 `perception_grounding` 写给受信任执行层；检查失败时执行器调用
次数为零。

| 动作前结果 | 含义 |
| --- | --- |
| `PERCEPTION_UNAVAILABLE` | 相机或感知流不可用 |
| `TARGET_NOT_VISIBLE` | 没有匹配且达到置信度阈值的当前检测 |
| `TARGET_NOT_LOCALIZED` | 看见目标，但缺少合法 bbox、基座系 XYZ 或正确坐标系 |
| `PERCEPTION_STALE` | 匹配检测没有可信时间戳，或已经过期/来自未来 |

`inspect(target)` 是搜索/重新观察技能，因此只要求感知流可用，不要求目标在执行
`inspect` 之前已经可见或有三维坐标。`move_home`、`move_relative`、
`open_gripper`、`close_gripper` 等不依赖场景目标的普通 Primitive 也不要求目标
定位，但仍受控制器限位、软件安全检查和停止规则约束；目标式 `move_to` 是例外，
必须经过定位门禁。

真机模式还要求动作后的 `verification_scope=physical`。只有命令级成功、
感知结论不充分或 `monitor_inconclusive` 都不能把步骤标成物理完成，而会转成
`VERIFICATION_UNCERTAIN`。

## 配置 Qwen API Key

代码会读取 `DASHSCOPE_API_KEY` 或 `QWEN_API_KEY`。不要把 Key 写进 Python 文件，也不要提交到 Git。

仅对当前 PowerShell 会话生效：

```powershell
$env:QWEN_API_KEY = "你的新 Key"
```

永久保存到当前 Windows 用户环境变量：

```powershell
[Environment]::SetEnvironmentVariable(
    "QWEN_API_KEY",
    "你的新 Key",
    "User"
)
```

永久设置后需要重新打开 PowerShell 或 VS Code 终端。曾经公开过的 Key 应先在服务端作废并重新生成。

可选配置：

```powershell
$env:ROBOT_AGENT_API_MODEL = "qwen-plus"
$env:ROBOT_AGENT_API_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
```

## 运行 Qwen + 模拟机械臂

项目要求 Python 3.8 或更高版本。建议先安装为可编辑包：

```powershell
python -m pip install -e .
```

交互输入任务：

```powershell
python .\run_agent_simulation.py
```

从命令行直接传入任务：

```powershell
python .\run_agent_simulation.py --task "从抽屉中拿出刷子"
```

如果任务不匹配内置演示实体，例如 `--task "抓小球"`，脚本会自动使用上述
开放世界模拟；无需为了 dry-run 预先写一个只包含“小球”的场景文件。

使用自定义语义场景：

```powershell
python .\run_agent_simulation.py `
  --scene-file .\my_scene.json `
  --task "把目标物放到指定区域"
```

场景可以在 `scene.skill_contracts` 中声明可复用的高层能力合同，例如
`manipulate(drawer, operation=open)` 的前置条件和结果。它描述单个技能的
语义，不是某个任务的固定步骤模板；Qwen 会根据本次用户目标重新组合合同。

使用自定义技能目录：

```powershell
python .\run_agent_simulation.py `
  --scene-file .\my_scene.json `
  --skills-file .\my_skills.json `
  --task "执行新任务"
```

`my_skills.json` 可以是数组，或包含 `capabilities` 数组的对象。每项至少
声明 `action_type`、`executor` 和 `description`；Policy 技能还可以声明
`policy_id`。运行时会用该目录动态生成 Qwen 白名单，不需要在提示词里
硬编码新任务。可直接参考
[`examples/lab_scene.json`](examples/lab_scene.json) 和
[`examples/lab_skills.json`](examples/lab_skills.json)。

保存完整任务记忆：

```powershell
python .\run_agent_simulation.py `
  --task "从抽屉中拿出刷子" `
  --state-dir .\.agent-state `
  --json
```

## Scripted 离线回归

`scripted` 模式不调用 Qwen，只用于验证固定的 pick-and-place 闭环、局部重试和后缀重规划；它不是通用自然语言 Planner。

第一次抓取失败后本地重试：

```powershell
python .\run_agent_simulation.py `
  --planner scripted `
  --task "Put the demo object on the demo tray." `
  --pick-failures 1
```

连续两次抓取失败后触发后缀重规划：

```powershell
python .\run_agent_simulation.py `
  --planner scripted `
  --task "Put the demo object on the demo tray." `
  --pick-failures 2
```

端到端离线测试：

```powershell
python .\test_agent_simulation.py
```

## 完整测试

安装项目后运行：

```powershell
python -m unittest discover -s tests -v
python .\test_agent_simulation.py
python -m compileall -q src tests run_agent_simulation.py test_agent_simulation.py
```

如果不安装项目，先临时配置源码路径：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
python .\test_agent_simulation.py
python -m compileall -q src tests run_agent_simulation.py test_agent_simulation.py
```

更详细的模块说明和工作记录见 [docs/agent_work_report.md](docs/agent_work_report.md)。
