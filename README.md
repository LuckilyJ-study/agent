# Robot Agent and YOLO-World Monitor

This repository contains two deployable modules:

- `agent-main_2/`: Qwen planning, capability validation, execution routing,
  monitoring, recovery, task memory, action safety, and the LIBERO bridge.
- `yolo-world-monitor/`: the YOLO-World-S HTTP perception service and its
  upstream YOLO-World source tree.

The Agent connects to YOLO-World through the `/health`, `/configure`, and
`/observe` HTTP endpoints. The LIBERO integration uses a separate bridge
process so MuJoCo can remain in its Python 3.8 environment while the Agent
reads cached observations and submits bounded action chunks over HTTP.

See the module READMEs and `agent-main_2/docs/LIBERO_INTEGRATION.md` for setup
and run commands.

## Files intentionally excluded

The following local or large files are not committed:

- Python virtual environments such as `.venv/`;
- model weights (`*.pth`, `*.pt`, and `*.ckpt`);
- Python caches and generated package metadata;
- API keys and `.env` files;
- the separate `LIBERO-master` source checkout.

Before starting the vision service, place the official
`yolo_world_v2_1_s_640.pth` checkpoint under
`yolo-world-monitor/YOLO-World/weights/`. LIBERO must also be installed or
copied separately on the simulation server.
