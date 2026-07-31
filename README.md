# Robot Agent and YOLO-World Monitor

本仓库包含机器人任务 Agent 和独立视觉监控服务：

- `agent-main_2/`：Qwen 规划、技能校验、调度、执行路由、Monitor、恢复与任务记忆。
- `yolo-world-monitor/`：YOLO-World-S HTTP 服务以及上游 YOLO-World 源码。

两个模块通过 `/health`、`/configure` 和 `/observe` HTTP 接口连接。当前演示入口
使用 `DryRunRobotController`，不会向真实机械臂发送命令。具体运行方式分别见两个
目录中的 `README.md`。

## 未包含文件

以下文件体积较大或包含本机环境信息，不提交到 GitHub：

- Python 虚拟环境 `.venv/`；
- YOLO-World `.pth` 权重；
- Python 缓存和构建产物；
- API Key 与 `.env` 文件。

部署视觉服务时，需要单独安装依赖，并把官方
`yolo_world_v2_1_s_640.pth` 权重放到
`yolo-world-monitor/YOLO-World/weights/`。
