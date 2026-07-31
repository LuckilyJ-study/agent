# Robot Agent 工作报告

更新日期：2026-07-31

## 1. 一页看懂：这个 Agent 是做什么的

这是一个面向机械臂的任务级 Agent。它接收用户的自然语言命令，调用千问理解任务并生成**高层技能计划**，再由本地可信代码完成技能校验、步骤调度、执行器选择、动作执行、运行监控、错误恢复、任务记忆和失败点后缀重规划。

例如，用户输入：

```text
把面团拿起来，放到托盘上
```

千问应该输出：

```text
pick(dough)
place(tray)
```

千问不负责生成关节角、电机电流、控制频率或 Pi0.5 Action Chunk，也不能直接选择某个本地执行器。`pick` 内部如何靠近、对准、闭合夹爪和抬升，由已经注册的抓取 Policy 负责；`move_home`、明确的小范围相对移动和夹爪开关等基础动作，可以交给机械臂控制接口。

整个主流程是：

```text
用户命令
  ↓
千问 Planner：生成高层技能计划
  ↓
CapabilityRegistry：动态技能白名单与计划校验
  ↓
TaskGraphScheduler：选择当前可执行步骤
  ↓
动作前观察；真机 PhysicalTargetGate 校验目标可见性与定位
  ↓
ExecutorRouter：在本地选择 Robot 或 Policy
  ↓
RobotPrimitiveExecutor / PolicyExecutor
  ↓
机械臂控制接口 / 训练 Policy / 可选 Pi0.5 适配器
  ↓
PerceptionProvider：提供连续结构化观察
  ↓
StructuredActionMonitor：动作前、动作中、动作后判断
  ↓
成功则继续；普通错误本地恢复；严重或连续错误则停止或后缀重规划
```

一句话概括：**千问负责“想接下来要做哪些技能”，Scheduler 负责“现在做哪一步”，Router 负责“本机让谁执行”，Monitor 负责“现场是否异常”，普通错误本地处理，复杂错误才交回千问。**

## 2. 当前状态与真机阶段的边界

| 模块 | 当前已经实现 | 真机阶段仍需接入 |
| --- | --- | --- |
| 千问 Planner | 自然语言转高层技能计划、结构化 JSON、失败点后缀重规划、可选二次计划复核 | 根据真实动态场景提供更完整的场景上下文 |
| 开放世界模拟 | 自动场景复用已知实体，并把计划中额外的 `pick/place/inspect` target 登记为无坐标虚拟实体；完全不匹配时从空场景启动 | 该机制只用于模拟，禁止作为真机目标存在或定位证据 |
| 技能白名单 | `CapabilityRegistry` 运行时注册、模型可见技能清单、参数和依赖校验 | 按实际机械臂和已训练模型登记真实能力 |
| Scheduler | 逐步调度、依赖关系、条件、跳过、完成后推进 | 无需改变主接口 |
| Router | 在本地决定 `robot` 或 `policy`，检查 Policy 类型和停止能力 | 绑定真实 Controller 和 Policy Backend |
| Robot 执行 | 基础动作接口与 `DryRunRobotController` | 真实机械臂驱动、标定、限位和急停反馈 |
| Policy 执行 | `PolicyRegistry`、`PolicyExecutor`、Dry-run Policy、Pi0.5 Gateway 适配层 | 真实训练模型或 Pi0.5 服务 |
| 感知接口 | `PerceptionProvider` 契约、无摄像头/脚本化实现、YOLO-World HTTP 二维检测 Provider | 高速多相机、深度、可靠跟踪、三维定位和坐标变换 |
| 真机动作前门禁 | `PhysicalTargetGate` 检查新鲜检测、bbox 与 `robot_base` XYZ，并在 Router 前 fail-closed | 用真实相机与标定数据验证门禁参数 |
| Monitor | YOLO-World HTTP 检测流、`StructuredActionMonitor` 动作前/中/后检查 | 用现场数据标定阈值和物理成功判据 |
| 运行时抢停 | 执行与监控并发、超时、取消、异常触发 `stop()`、停止确认 | 验证真实后端能够及时并可靠停止 |
| 本地恢复 | 错误矩阵、停止、回安全位、重新观察、重试和升级重规划 | 根据真机工作区细化安全撤退轨迹 |
| TaskMemory | 计划、步骤、尝试次数、观察摘要、状态、事件、重规划次数和可选 JSON 持久化 | 图像/视频存储策略和现场回放系统 |

最重要的限制是：

> 当前 dry-run 输出 `completed`，只表示计划中的模拟命令链已经走完，不能解释为物体在现实中已经抓起、移动或放置成功。

## 3. 完整工作流程

### 3.1 用户输入任务

交互脚本会读取用户输入，例如：

```text
从抽屉中拿出刷子
```

Agent 同时准备当前语义场景和本机能力清单。语义场景用于告诉 Planner 当前有哪些实体、关系和状态；能力清单用于告诉 Planner 当前允许调用哪些技能。

如果没有传入 `--scene-file`，程序会先尝试内置演示工作区。匹配成功时可以
复用已知实体和可信技能合同，并把本次任务额外出现的新 target 登记为虚拟实体；
完全不匹配时，模拟入口改用一个空的 `open_world_simulation` 语义场景。这两种
情况都不代表真机发现了新物体，只表示允许 Planner 从用户原话提取语义 target。

### 3.2 千问只生成高层技能计划

千问 Planner 接收：

- 用户原始命令；
- 当前 `WorldState`；
- 当前场景可用实体；
- `CapabilityRegistry` 导出的模型可见技能；
- 失败重规划时的已完成步骤和当前现场摘要。

规范化步骤至少包含：

```json
{
  "step_id": 1,
  "action_type": "pick",
  "target": "brush",
  "expected_result": "brush is held by the gripper",
  "status": "pending",
  "parameters": {},
  "depends_on": [],
  "timeout_seconds": 60,
  "max_attempts": 2
}
```

Planner 的职责是选择和排列技能，不是做底层控制。对 `pick(brush)`，它不应继续展开为：

```text
移动到某个 XYZ
下降 8 cm
夹爪闭合
上升 10 cm
```

这些细节必须来自真实感知、标定结果、机械臂控制器或抓取 Policy，不能由语言模型猜测。

同样，“从抽屉中拿出刷子”应根据当时的场景和已注册能力组织成类似：

```text
manipulate(drawer, operation=open)
pick(brush)
```

其中 `manipulate` 对应的开抽屉 Policy 负责把手定位、接近、抓住把手和拉开；`pick` 对应的抓取 Policy 负责取刷子。如果没有注册开抽屉能力，Agent 应报告能力缺失，而不是退回到由千问猜测一串 XYZ 坐标。

当前千问提示词还明确禁止输出：

- 关节角；
- 笛卡尔坐标；
- 速度和电机命令；
- Action Chunk；
- `executor`；
- `policy_id`。

Planner 输出后还会经过 `SceneSkillPlanner`。场景中的
`scene.skill_contracts` 只描述单个可复用技能的语义合同，例如：

```json
{
  "action_type": "manipulate",
  "target": "drawer",
  "parameters_match": {"operation": "open"},
  "preconditions": [
    {"path": "objects.drawer.state", "operator": "eq", "value": "closed"}
  ],
  "effects": [
    {"path": "objects.drawer.state", "operation": "set", "value": "open"}
  ]
}
```

本地代码按当前状态模拟 Qwen 生成的技能序列，拒绝“抽屉仍关闭就直接拿
内部物体”等错误顺序，并且丢弃大模型自己声明的 `effects`，只采用可信
合同中的状态变化。若首次计划不满足合同，错误会反馈给 Qwen 修复一次。
这些合同是技能定义，不是“拿刷子”或“刷披萨”的固定任务模板；不同用户
目标仍由 Qwen 在运行时重新选择和组合。

如果用户明确要求一个已注册的基础操作，例如“机械臂向左移动 0.5 cm”，Planner 可以生成 `move_relative`。这不代表它可以为复杂任务自行编造抓取坐标。

#### 3.2.1 开放世界模拟如何处理未知目标

以“抓小球”为例，开放世界模拟的链路是：

```text
Qwen 输出 pick(小球)
  → CapabilityRegistry 确认 pick 在白名单
  → VirtualEntityRegistry 生成安全本地 ID
  → 小球作为 movable_object 写入模拟 WorldState
  → SceneSkillPlanner 推导通用 pick 的前置条件和符号效果
  → Dry-run Policy 执行
```

虚拟实体记录可以包含：

```json
{
  "virtual_entity_<hash>": {
    "type": "movable_object",
    "display_name": "小球",
    "aliases": ["小球"],
    "location": "virtual_workspace",
    "virtual": true,
    "source": "open_world_simulation",
    "physical_confirmation": false
  }
}
```

它只能回答“模拟计划中有一个叫小球的语义对象”，不能回答小球现实中是否
存在、在图像中的位置、机械臂基座坐标或抓取姿态。注册补丁只包含实体声明，
不会包含 Qwen 坐标，也不会把 Grounder 为验证未来计划而模拟出的 `held` 等
终态提前写入当前世界。

开放世界只扩展**实体词汇**，不扩展能力：

- `pick/place/inspect` 可以使用已有通用语义登记未知 target；
- `move_to` 若由部署公开，也只能登记一个语义目标，不能生成物理坐标；
- 未知 action 仍被技能白名单拒绝；
- 通用 `manipulate(未知对象)` 没有本地 `skill_contract` 时仍被拒绝；
- 没有注册的 Policy、焊接、开锁等技能不会因为出现同名实体而自动获得。

显式 `--scene-file` 默认是闭合场景：文件漏写的对象不会被自动补齐。真机模式
也绝不使用虚拟实体作为感知或定位证据。

### 3.3 动态技能白名单检查

`CapabilityRegistry` 是 Planner 和执行层共同使用的单一可信能力来源。所谓“动态白名单”，是指不同部署可以在运行时注册不同技能，而不是在提示词中永久写死一组动作。

每个能力声明：

- `action_type`；
- 本地执行器类别 `robot` 或 `policy`；
- 能力说明；
- 默认 Policy ID（如有）；
- 是否支持停止；
- 必填参数；
- 默认超时；
- 默认最大尝试次数。

对 Planner 只公开技能名称、说明和必要参数，不公开本地执行器和 Policy ID。Planner 输出后，本地代码会重新校验：

- 技能是否注册；
- `target` 和 `expected_result` 是否为空；
- 参数是否完整；
- 超时和尝试次数是否合法；
- 步骤 ID 是否重复；
- 依赖是否存在；
- 是否存在自依赖或依赖环；
- 基础位移、坐标和坐标系是否超过软件约束。

例如 Planner 输出：

```text
fly_to_moon()
```

如果本机没有注册该能力，计划会在进入执行层之前被拒绝。大模型不能通过在 JSON 中写入一个新名字，为机械臂创造实际不存在的能力。

### 3.4 Scheduler 选择当前步骤

`TaskGraphScheduler` 只选择当前满足条件的步骤，不重新思考任务目标。它负责：

1. 检查 `depends_on` 是否已经完成；
2. 检查 `conditions` 是否满足；
3. 根据 `on_condition_false` 决定跳过或失败；
4. 返回当前可运行步骤；
5. 只有步骤验证成功后才应用可信 `effects`；
6. 将步骤标记为 `completed` 后再推进。

因此，如果抓取失败，Scheduler 不会继续执行后面的移动和放置。

### 3.5 动作前观察与状态快照

每个 Action 开始前，Agent 会：

1. 将当前步骤及额外监控目标配置给感知接口；
2. 获取动作前 `observation`；
3. 获取动作前 `robot_state`；
4. 记录 `action_id`、步骤实例、尝试次数和开始时间；
5. 把执行中动作写入 `TaskMemory.inflight_action`；
6. 执行软件安全检查；
7. 真机模式调用 `PhysicalTargetGate` 检查感知流、目标检测的新鲜度和定位；
8. 如果门禁、Monitor 或安全检查失败，则在 Router 和执行器之前拦截。

这为失败恢复、进程中断恢复和问题追踪保留了明确起点。

动作前门禁和 Router 的先后关系是：

```text
configure_targets
  → observe
  → software safety
  → PhysicalTargetGate / Monitor precheck
  → 检查通过后才调用 ExecutorRouter.route()
  → Robot 或 Policy
```

因此，目标不可见、定位缺失或检测过期时，不会先选好 Policy 再尝试停止，而是
根本不向 Router 和执行器下发该动作。

### 3.6 Router 在本地选择执行方式

`ExecutorRouter` 不信任 Planner 提供的执行器选择。它根据本地 `CapabilityRegistry` 决定路由：

```text
move_home / move_relative / open_gripper
  → RobotPrimitiveExecutor
  → 机械臂控制接口

pick / place / manipulate / inspect
  → PolicyExecutor
  → PolicyRegistry 中匹配的训练模型
```

Policy 路由还会检查：

- Policy 是否真实注册；
- Policy 的 `action_type` 是否与当前技能一致；
- 多个候选 Policy 是否产生歧义；
- 当前 Policy 是否支持安全停止；
- Policy 所需图像、机器人状态等输入是否可用。

如果 Planner 私自把 `pick` 指定为 `robot`，或指定了错误 Policy，本地 Router 会拒绝，不会照单执行。

### 3.7 Robot、Policy 与 Pi0.5 执行链

当前有两条独立执行链。

基础动作：

```text
RobotPrimitiveExecutor
  → PrimitiveRobotController
  → move_relative / move_to / move_to_pose / move_linear
  → move_home / open_gripper / close_gripper / stop
```

训练模型：

```text
PolicyExecutor
  → PolicyRegistry
  → 已注册 PolicyBackend
  → 训练模型输出连续控制
```

Pi0.5 不是主架构的强制依赖。需要使用时，可以把 `Pi05GatewayBackend` 注册成某个 Policy 的 Backend：

```text
pick(dough)
  → PolicyExecutor
  → Pi05GatewayBackend
  → Pi0.5 Gateway
  → 机械臂底层控制
```

如果项目选择直接控制机械臂，或已有其他抓取/放置任务模型，只需实现相同的 Controller/Policy 接口，不需要修改 Planner、Scheduler、Router、Monitor 和 TaskMemory 主循环。

### 3.8 两个并行循环

整个系统逻辑上有两个同时运行的循环。

任务循环：

```text
规划 → 白名单校验 → 调度当前技能 → 路由 → 执行 → 验证 → 下一步
```

监控循环：

```text
相机帧 → 检测/跟踪 → 结构化 observation → Monitor → 正常或异常
```

当前闭环把同步 Executor 放入工作线程运行，Agent 主线程按照配置周期持续拉取观察和机器人状态。默认轮询间隔是 0.05 秒。执行期间如果检测到异常，Agent 不必等整个 Policy 自行返回，而是立即调用当前 Executor 的 `stop()`。

每个步骤还有 `timeout_seconds`。超过时间后会触发：

```text
ACTION_TIMEOUT
  → executor.stop()
  → 等待停止确认
```

如果超过停止宽限时间仍不能确认执行线程结束，结果升级为：

```text
ACTION_STOP_UNCONFIRMED
```

这会作为安全关键错误停止流程。真实 Controller 和 Policy Backend 必须实现可抢占、可确认的停止语义。

### 3.9 感知接口与 StructuredActionMonitor

当前已经定义通用 `PerceptionProvider`：

```python
hardware_ready: bool
supports_target_configuration: bool
supports_localization: bool
localization_modes: frozenset[str]

observe() -> dict
```

目标感知实现还可以提供：

```python
configure_targets(labels)
```

真机 Provider 必须设置 `hardware_ready=True`、实现
`configure_targets(labels)`，并在 `localization_modes` 中至少声明
`bbox_2d` 和 `robot_base_xyz`。`NullPerceptionProvider` 和测试用
`ScriptedPerceptionProvider` 都不会伪装成 `hardware_ready`。

当前 `YoloWorldHttpPerceptionProvider` 会根据每个步骤动态设置文字标签，例如：

```text
pizza dough
robot gripper
pizza tray
```

并持续输出：

```json
{
  "available": true,
  "source": "yolo_world_s",
  "timestamp": "2026-07-31T08:00:00+00:00",
  "frames": [
    {
      "timestamp": "2026-07-31T08:00:00+00:00",
      "detections": [
        {
          "label": "dough",
          "confidence": 0.89,
          "bbox_xyxy": [10, 20, 100, 120],
          "track_id": 7,
          "following_gripper": true,
          "falling": false
        }
      ],
      "signals": {
        "collision_risk": false
      }
    }
  ]
}
```

#### 真机 PhysicalTargetGate

在 `hardware_mode=True` 下，`build_agent_runtime()` 会验证 Provider 契约并
自动安装 `PhysicalTargetGate`。对 `pick`、`place`、`manipulate` 和目标式
`move_to`，门禁要求同一个匹配检测同时满足：

- target 实体 ID、label、name 或 alias 匹配；
- 置信度达到本地阈值；
- observation、frame 或 detection 提供可解析且足够新鲜的时间戳；
- `bbox_xyxy` 是合法的二维检测框；
- `position_xyz_m` 是有限的三维数值，且位于配置的工作空间边界内；
- `coordinate_frame` 严格等于 `robot_base`。

简写就是：每个真机目标动作都需要 **fresh YOLO/provider bbox +
`robot_base` XYZ**，二维框或三维位置单独存在都不够。

门禁不会把坐标发回 Qwen。检查通过后，它只在当前步骤的可信参数中加入精简的
`perception_grounding`，供 Router 后面的 Controller/Policy 使用。失败发生在
`ExecutorRouter.route()` 之前：

别名只接受本地 `SceneSkillPlanner` 写入的可信实体别名；Qwen 不能通过
`parameters.target_aliases`、`monitor_targets` 或伪造
`perception_grounding` 让另一个检测对象冒充当前 target。

| 错误 | 含义与执行结果 |
| --- | --- |
| `PERCEPTION_UNAVAILABLE` | 感知流不可用；不调用 Router/Executor |
| `TARGET_NOT_VISIBLE` | 没有匹配且达到置信度要求的检测；不执行目标动作 |
| `TARGET_NOT_LOCALIZED` | 看见目标，但 bbox、三维位置或基座坐标系无效 |
| `PERCEPTION_STALE` | 时间戳缺失、过期或超出允许的未来偏差 |

`inspect(target)` 的目的可以是搜索当前不可见的目标，所以只要求实时感知源可用，
不要求执行前已经得到 target bbox 或 XYZ。后续 `pick/place/manipulate` 仍必须
重新通过完整定位门禁。`move_home`、`move_relative`、`open_gripper` 和
`close_gripper` 等不依赖场景目标的 Primitive 不要求目标定位；目标式 `move_to`
是例外。

`StructuredActionMonitor` 已经可以消费这种结构化结果，并在三个阶段检查：

- 动作前：目标是否连续多帧不可见，是否有碰撞或硬件故障信号；
- 动作中：目标是否丢失，物体是否从夹爪分离并下落，是否出现安全风险；
- 动作后：抓取、放置或预期结果是否有足够证据。

当前动作前门禁与 Monitor 能够产生的主要结果包括：

| 现场信息 | 闭环结果 |
| --- | --- |
| 真机感知流不可用 | `PERCEPTION_UNAVAILABLE` |
| 动作前没有匹配目标 | `TARGET_NOT_VISIBLE` |
| 有目标框但没有基座系三维定位 | `TARGET_NOT_LOCALIZED` |
| 检测时间戳无效或过期 | `PERCEPTION_STALE` |
| 连续多帧找不到目标 | `TARGET_LOST` |
| 夹爪已闭合并抬升，但目标未跟随 | `GRASP_FAILED` |
| 物体先跟随夹爪，随后分离或下落 | `OBJECT_DROPPED` |
| 放置完成信号为假 | `PLACE_FAILED` |
| 有碰撞风险 | `COLLISION_RISK` |
| 硬件故障 | `HARDWARE_FAULT` |
| 证据不能支持预期结果 | `EXPECTED_RESULT_NOT_MET` 或命令级未验证 |

这里必须明确：

> YOLO-World-S 通过独立 HTTP 服务接入，Agent 进程不直接加载模型权重。当前
> Provider 只声明 `bbox_2d` 且 `hardware_ready=False`，因此可用于 DryRun 监控，
> 不能作为真机三维定位 Provider，也不会把二维检测框冒充机械臂坐标。

无相机模式使用 `NullPerceptionProvider`，明确返回 `available=false`；测试还可以
使用 `ScriptedPerceptionProvider` 或临时 HTTP 服务注入确定性的帧序列。显式配置
YOLO endpoint 后采用失败关闭策略，服务中断会得到 `PERCEPTION_FAILED`。

### 3.10 动作后验证

Executor 返回成功，只能说明执行请求被 Backend 接受并完成。Agent 还会采集动作后观察和机器人状态，再调用 Monitor/Verifier。

验证结果统一为：

```json
{
  "success": true,
  "error_type": "NONE",
  "confidence": 0.95,
  "verification_scope": "physical",
  "details": {}
}
```

只有 `success=true` 且置信度达到阈值时，Scheduler 才会完成当前步骤。置信度不足会转换为 `LOW_CONFIDENCE`。

当前存在三种必须区分的验证范围：

| 范围 | 含义 |
| --- | --- |
| `command` / `plan` | 命令或软件计划完成，未证明真实物理结果 |
| `symbolic_goal` | 模拟世界状态满足符号目标，仍不是摄像头物理证明 |
| `physical` | Monitor 根据真实感知证据验证物理结果 |

真机运行时设置 `require_physical_verification=True`。即使 Backend 返回
`status=success`，只要最终证据仍是 `command`、标记
`monitor_inconclusive`，或没有明确物理成功范围，Agent 就会把结果改为
`VERIFICATION_UNCERTAIN`，不会把步骤当作真实完成。

### 3.11 本地恢复矩阵

系统不会每次失败都调用千问。`RecoveryPolicy` 先按固定规则分类：

| 错误类型 | 默认处理 |
| --- | --- |
| 第一次 `GRASP_FAILED` | 停止、回安全位、重新观察、重试当前抓取 |
| 普通可恢复执行失败 | 停止、重新观察、重试当前步骤 |
| `TARGET_NOT_VISIBLE` / `TARGET_LOST` | 先搜索或重新观察；仍失败则重规划 |
| `TARGET_NOT_LOCALIZED` / `PERCEPTION_STALE` | 刷新检测与三维定位；不得盲目执行 |
| `PERCEPTION_UNAVAILABLE` / `PERCEPTION_FAILED` | 阻止目标动作并尝试恢复感知；仍失败则重规划或停止 |
| `LOW_CONFIDENCE` / `VERIFICATION_UNCERTAIN` | 刷新观察；仍不确定则重规划 |
| `OBJECT_DROPPED` | 场景已经改变，直接后缀重规划 |
| 连续失败或达到 `max_attempts` | 后缀重规划 |
| `COLLISION_RISK` / `EMERGENCY_STOP` | 立即安全停止，不自动重试 |
| `HARDWARE_FAULT` / `ROBOT_DISCONNECTED` | 立即安全停止 |
| 执行器、Policy 或停止能力不安全 | fail-closed，停止 |
| `ACTION_TIMEOUT` / `ACTION_STOP_UNCONFIRMED` | 安全停止 |

`ControllerLocalRecoveryHandler` 当前提供基本恢复动作：

```text
stop
  → 抓取/放置失败时尝试 move_home
  → 重新配置目标
  → reobserve
  → 重新执行当前步骤
```

如果本地恢复自身失败，不会继续盲目执行，而会升级到重规划。真机阶段需要根据工作空间、障碍物和末端工具，把 `move_home` 替换或扩展为经过验证的安全撤退动作。

### 3.12 TaskMemory 与失败点后缀重规划

`TaskMemory` 保存：

- 用户原始任务；
- 全局 Task ID；
- 当前计划和步骤状态；
- 当前步骤与步骤实例 ID；
- 每个实例的尝试次数；
- 已完成和失败记录；
- 动作前后的 observation 摘要；
- 动作前后的 robot state；
- Monitor 结果；
- 当前 `WorldState`；
- 当前执行中的 Action；
- 完整事件时间线；
- 重规划次数；
- 最终任务验证和失败原因。

普通错误超过本地恢复上限后，Agent 向千问发送精简、结构化的失败上下文，而不是四路原始视频：

```json
{
  "original_task": "把面团放到托盘",
  "completed_steps": ["move_to(dough)"],
  "current_step": "pick(dough)",
  "failed_skill": "pick",
  "failed_target": "dough",
  "error_type": "GRASP_FAILED",
  "confidence": 0.95,
  "target_visible": true,
  "retry_count": 1,
  "current_observation": {},
  "robot_state": {},
  "capabilities": []
}
```

千问被要求只返回**从当前物理状态开始的未完成后缀**。例如：

```text
原计划：
move_to(dough) → pick(dough) → place(tray)

move_to 已完成，pick 连续失败：
inspect(dough) → pick(dough) → place(tray)
```

已完成前缀保留在 `TaskMemory` 中，不会重新执行。重规划结果仍必须重新经过相同的技能白名单、参数、依赖和 Router 检查，而且不能改变持久化的原始任务目标。

## 4. 当前已经可以完成的功能

在不连接真机的情况下，当前代码可以验证以下软件能力：

1. 读取用户自然语言任务；
2. 通过千问 API 生成高层技能计划；
3. 可选使用第二次千问调用复核计划；
4. 按当前部署能力生成动态技能白名单；
5. 使用语义技能合同校验前置条件、可信效果和最终符号目标；
6. 拦截未知技能、错误参数、错误依赖和错误路由；
7. 通过 Scheduler 逐步执行任务图；
8. 在本地把基础动作路由给 Robot，把复杂技能路由给 Policy；
9. 检查 Policy 是否存在、输入是否足够、动作类型是否匹配、能否停止；
10. 记录动作前后状态和完整事件时间线；
11. 动作执行期间并发轮询 Monitor；
12. 超时、取消、碰撞风险或监控异常时主动调用 `stop()`；
13. 按错误矩阵执行本地停止、撤退、重新观察和重试；
14. 连续失败后携带当前状态进行后缀重规划；
15. 使用 JSON 持久化 TaskMemory，并支持中断后的任务恢复基础；
16. 在自动模拟场景中复用已知实体，并为通用 `pick/place/inspect` 的新 target 登记无坐标虚拟实体；
17. 验证真机 Provider 是否明确声明硬件就绪、目标配置和定位模式；
18. 在 Router 前用 `PhysicalTargetGate` 拦截不可见、未定位或过期的目标；
19. 使用 Dry-run Robot、Dry-run Policy 和脚本化感知进行无硬件测试；
20. 通过独立 HTTP 服务接入 YOLO-World-S 二维检测，并在服务中断时失败关闭。

## 5. 真机阶段需要接入的功能

真机 Controller 必须显式声明 `hardware_ready=True`；运行时同时拒绝
`dry_run=True`、`DryRunRobotController` 和 `DryRunPolicyBackend`，避免出现
“真实机械臂原语 + 假 Policy 成功”的混合执行。

### 5.1 YOLO-World-S 真机视觉链路

当前 HTTP Provider 已经负责动态标签、二维检测、滚动帧和基础跟踪信号。真机版
`PerceptionProvider` 还需要补充或验证：

- 明确声明 `hardware_ready=True`；
- 实现 `configure_targets(labels)`；
- 声明 `localization_modes={"bbox_2d", "robot_base_xyz"}` 或等价集合；
- 读取真实高速多相机和深度数据；
- 保证 YOLO-World-S 服务的延迟、断线恢复和进程监管；
- 对当前 Action 动态标签进行现场词汇与阈值标定；
- 维护带可信时间戳的连续帧缓存；
- 使用经过验证的跨帧目标与夹爪跟踪；
- 输出带可信时间戳的目标置信度、框、Track ID、跟随和下落信号；
- 输出经过标定并转换到 `robot_base` 的三维定位结果。

YOLO-World-S 负责报告“看到了什么、在哪里”。`StructuredActionMonitor` 负责把多帧检测和机器人状态转成 `TARGET_LOST`、`GRASP_FAILED`、`OBJECT_DROPPED` 等语义判断。

`hardware_ready=True` 是部署方对真实硬件链路的显式承诺，不应由
`NullPerceptionProvider`、离线图片或普通脚本桩设置。即使 Provider 声明支持
定位，每次目标动作仍必须由 `PhysicalTargetGate` 检查当前检测，而不能只依赖
启动时能力声明。

### 5.2 相机到机械臂坐标系的标定

二维检测框不能直接作为机械臂坐标。真机必须补充：

- 相机内参；
- 相机与机械臂基座外参；
- 深度或多视角三维定位；
- 目标姿态估计；
- 抓取点/放置点生成；
- 工作空间、速度和碰撞约束。

语言模型不应承担坐标变换和实时轨迹生成。

### 5.3 真实 Robot Controller

需要实现 `PrimitiveRobotController` 对应方法，并确保：

- 命令有明确成功/失败响应；
- `get_state()` 返回真实连接、位姿、关节和夹爪状态；
- `stop()` 可抢占正在运行的动作；
- 急停、断连和硬件故障能及时上报；
- 速度、加速度、工作空间和力矩限制在底层再次检查。

### 5.4 真实 Policy 或 Pi0.5

对 `pick`、`place`、`manipulate` 等能力，需要注册已经训练并测试过的 `PolicyBackend`。可以使用 Pi0.5，也可以使用其他模型或厂商控制栈。

Pi0.5 只是可选执行 Backend，不负责总任务规划、技能白名单、Scheduler、恢复决策和 TaskMemory。

### 5.5 物理任务级验证

真机最终需要一个基于真实观察的 `TaskVerifier`，确认最终目标确实成立，例如：

- 面团实际位于托盘；
- 刷子实际被取出并放到工具架；
- 酱料覆盖达到任务要求；
- 目标没有在最后一步后再次掉落。

## 6. 必须正确理解的能力边界

### 6.1 不是“任何文字任务都能执行”

千问能够重新组合已有技能，但系统仍受两个条件限制：

1. 任务目标必须能由语义场景、模拟虚拟实体或真机感知安全解析；
2. 本机必须注册完成任务所需的 Robot Primitive 或训练 Policy。

例如，用户要求“把未知零件焊接起来”，但场景中没有焊枪、没有焊接目标，也没有注册 `weld` Policy，正确结果是拒绝或报告能力缺失，而不是让千问编造动作。

这里要区分两种“未知物体”：

- 模拟模式可以把“抓小球”中的“小球”登记成虚拟语义对象，以测试 Planner、
  Scheduler、Router 和通用抓取调用链；这个登记不包含位置，也没有让现实中的
  小球自动出现。
- 真机模式必须由 `hardware_ready` 的感知 Provider 看到目标，并由
  `PhysicalTargetGate` 获得新鲜 bbox 和 `robot_base` XYZ 后，目标动作才可进入
  Router。

因此：

```text
大模型的智能
≠ 虚拟语义实体是现实物体
≠ 未训练技能自动具备
≠ 未标定坐标可以安全猜测
```

扩展任意任务的正确方法是增加：

- 新实体和实时感知标签；
- 新技能及其本地能力合同；
- 新 Robot Primitive 或 Policy Backend；
- 对应的安全约束和成功判据。

### 6.2 `completed` 不等于物理完成

当前模拟模式下：

```text
status = completed
```

只表示：

- 计划通过校验；
- Scheduler 已遍历计划；
- Dry-run Executor 返回命令完成；
- 软件流程没有触发失败。

在开放世界模拟中，它还可能表示“虚拟实体的符号状态已从
`virtual_workspace` 更新为 `held` 或某个虚拟放置目标”。这是内存中的状态
转换，不是 YOLO 检测结果，也不是机械臂坐标或物理成功证据。

它不表示：

- 真实机械臂移动过；
- 夹爪真正抓到物体；
- 物体没有掉落；
- 最终任务已经通过摄像头验证。

运行脚本会明确输出：

```text
Execution mode: SIMULATION
Physical verification: NOT AVAILABLE
```

只有接入真实感知并得到 `verification_scope=physical` 的成功证据后，才能声称相应物理动作得到验证。

## 7. 如何运行当前模拟 Agent

### 7.1 配置千问 API Key

建议把 Key 保存为 Windows 用户环境变量，不要写入源码：

```powershell
[Environment]::SetEnvironmentVariable(
    "QWEN_API_KEY",
    "在这里填写新申请的 Key",
    "User"
)
```

重新打开 PowerShell 或 VS Code 终端后运行。

### 7.2 交互输入任务

```powershell
python .\run_agent_simulation.py
```

脚本会提示：

```text
请输入机械臂任务:
```

### 7.3 命令行直接传入任务

```powershell
python .\run_agent_simulation.py --task "把面团拿起来放到托盘上"
```

### 7.3.1 开放世界模拟未知实体

```powershell
python .\run_agent_simulation.py --task "抓小球"
```

如果“小球”不匹配内置演示工作区，脚本会自动选择
`open_world_simulation`，让 Qwen 生成高层计划，再由本地注册无坐标的虚拟
语义实体。该流程只用于 dry-run，不会调用摄像头或真实机械臂，也不会新增技能。

### 7.4 使用自定义语义场景

```powershell
python .\run_agent_simulation.py `
  --scene-file .\my_scene.json `
  --task "从抽屉中拿出刷子"
```

自定义场景必须提供任务需要的实体和状态；场景文件不能代替真实摄像头定位。

### 7.5 不调用千问的离线回归

```powershell
python .\run_agent_simulation.py `
  --planner scripted `
  --task "Put the demo object on the demo tray." `
  --pick-failures 1
```

该模式用于测试“第一次抓取失败后本地重试”的软件闭环，不用于展示千问的自然语言规划能力。

### 7.6 运行测试

```powershell
python -m unittest discover -s tests -v
python .\test_agent_simulation.py
```

## 8. 主要代码位置

| 文件 | 作用 |
| --- | --- |
| `run_agent_simulation.py` | 千问 + 模拟执行的用户入口 |
| `src/robot_agent/planner.py` | Rule-based、Ollama、Qwen Planner，高层计划协议与失败重规划 |
| `src/robot_agent/capabilities.py` | 动态能力注册、模型可见白名单、计划规范化和参数校验 |
| `src/robot_agent/skill_grounding.py` | 高层技能合同、实体别名落地、前置条件模拟和可信效果 |
| `src/robot_agent/virtual_entities.py` | 开放世界模拟的安全实体 ID、语义角色和注册补丁 |
| `src/robot_agent/task_scheduler.py` | 步骤依赖、条件和完成推进 |
| `src/robot_agent/routing.py` | 本地 `ExecutorRouter`，Robot/Policy/Policy ID 安全路由 |
| `src/robot_agent/action_executors.py` | Robot/Policy Executor、Dry-run Backend、Pi0.5 Gateway Backend |
| `src/robot_agent/perception.py` | 真机感知边界、空感知和脚本化感知 |
| `src/robot_agent/yolo_world_perception.py` | YOLO-World HTTP 健康检查、动态标签、二维检测帧规范化和失败关闭 |
| `src/robot_agent/physical_target_gate.py` | 真机 Provider 契约与 Router 前目标可见性/定位门禁 |
| `src/robot_agent/monitor.py` | `StructuredActionMonitor` 和结构化观察摘要 |
| `src/robot_agent/safety_monitor.py` | 动作前、动作中、动作后的软件安全检查 |
| `src/robot_agent/local_recovery.py` | 停止、回安全位、重新观察等本地恢复动作 |
| `src/robot_agent/closed_loop.py` | 主闭环、并发监控、超时抢停、恢复、TaskMemory 和后缀重规划 |
| `src/robot_agent/runtime.py` | 组装 Planner、Controller、Policy、感知、Monitor 和 Router |
| `src/robot_agent/persistence.py` | TaskMemory JSON 快照和恢复 |
| `src/robot_agent/task_verifier.py` | 最终任务级验证接口 |
| `src/robot_agent/simulation.py` | 可重复的无真机执行与失败注入 |

## 9. 最终总结

当前项目已经具备一个通用机械臂 Agent 的软件骨架：

```text
千问高层规划
→ 动态技能白名单
→ 闭合场景落地或开放世界虚拟实体登记（仅模拟）
→ Scheduler
→ 真机 PhysicalTargetGate
→ 本地 Router
→ Robot 或 Policy/Pi0.5
→ 运行中 Monitor
→ 超时或异常抢停
→ 本地恢复
→ TaskMemory
→ 失败点后缀重规划
```

它已经能够验证“理解任务、选择已有技能、逐步执行、遇错恢复、复杂失败后继续规划剩余任务”的完整控制逻辑。

YOLO-World-S 二维检测已经通过独立 HTTP 服务接入。尚未完成的是物理世界一侧的高速多相机与深度、可靠跟踪、三维定位和手眼标定、真实机械臂 Controller、真实训练 Policy 以及最终物理目标验证。因此，当前系统适合做架构开发、Planner 评估、技能注册、调度恢复和视觉 DryRun 测试；在完成真机接入及安全验证前，不能把 dry-run 的 `completed` 当作真实任务完成。
