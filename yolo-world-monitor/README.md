# YOLO-World-S Monitor Service

该目录把原作者 YOLO-World 仓库包装成独立视觉服务，避免它的 PyTorch、MMCV、
MMDetection 和 CUDA 依赖影响 Robot Agent 或机器人控制环境。

```text
摄像头 / 图片 / 视频
  -> YOLO-World-V2.1-S
  -> 简单跨帧跟踪
  -> HTTP /observe
  -> agent-main 的 YoloWorldHttpPerceptionProvider
  -> StructuredActionMonitor
```

## 当前文件

- `YOLO-World/`：原作者代码仓库。
- `YOLO-World/weights/yolo_world_v2_1_s_640.pth`：官方 S 640 权重的本地放置路径。
- `yolo_world_service.py`：本项目增加的独立视觉服务。
- `test_yolo_world_service.py`：不加载模型的跟踪逻辑测试。

## 环境边界

请在独立 Python 环境中安装 YOLO-World，不要在 Agent 或机器人环境中直接安装。
当前已使用本目录的 `.venv` 完成验证。具体 PyTorch、CUDA、MMCV 组合应根据本机
显卡和驱动选择。环境准备完成后，先确认：

`.venv` 和约 305 MB 的 `.pth` 权重不会提交到 GitHub。克隆本仓库后，需要单独
创建视觉环境，并从 YOLO-World 官方发布渠道下载
`yolo_world_v2_1_s_640.pth`，放到 `YOLO-World/weights/`。

```powershell
& ".\.venv\Scripts\python.exe" -c "import torch, cv2, mmengine, mmdet; print(torch.__version__); print(torch.cuda.is_available())"
```

## 启动服务

先运行一次官方示例图片检测，确认环境和权重匹配：

```powershell
& ".\.venv\Scripts\python.exe" .\run_image_smoketest.py
```

使用本机第一个摄像头：

```powershell
cd "D:\vscode会话\latex格式更改\yolo-world-monitor"

& ".\.venv\Scripts\python.exe" .\yolo_world_service.py `
  --source 0 `
  --device auto `
  --labels "pizza dough,robot gripper,pizza tray"
```

使用静态图片验证模型加载：

```powershell
& ".\.venv\Scripts\python.exe" .\yolo_world_service.py `
  --source ".\YOLO-World\demo\sample_images\bus.jpg" `
  --labels "bus,person"
```

服务默认地址是 `http://127.0.0.1:8765`。在另一个 PowerShell 中检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
Invoke-RestMethod http://127.0.0.1:8765/observe
```

## HTTP 接口

- `GET /health`：模型、设备、图像源和推理线程状态。
- `POST /configure`：设置本次技能需要检测的英文目标词汇。
- `GET /observe`：返回滚动帧、检测框、置信度和基础跟踪信号。

服务只返回结构化结果，不通过 HTTP 发送原始图像。当前 `following_gripper`
和 `falling` 是图像空间启发式信号，只适合框架验证。真机阶段应融合深度、标定、
夹爪状态和机器人位姿，并由底层控制器承担硬安全。

## 运行轻量测试

该测试不加载 YOLO 权重，也不要求 GPU：

```powershell
& ".\.venv\Scripts\python.exe" -m unittest test_yolo_world_service -v
```

## 尚未安装模型依赖时模拟完整接口

模拟服务与真实服务使用相同的三个 HTTP 接口，因此可以先验证 Agent 集成：

```powershell
& ".\.venv\Scripts\python.exe" .\mock_vision_service.py --scenario visible
```

可选场景包括 `visible`、`target_lost`、`grasp_failed` 和 `object_dropped`。
Agent 仍然使用 `--vision-endpoint http://127.0.0.1:8765`，以后切换为真实
YOLO-World 服务时不需要修改 Agent 代码。
