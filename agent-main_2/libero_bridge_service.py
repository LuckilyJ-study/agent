"""Standalone Python 3.8 bridge between LIBERO/MuJoCo and robot_agent.

Run this file inside the dedicated LIBERO conda environment. The Agent may run
in another environment and communicates through HTTP. Only the worker thread
touches MuJoCo; /observe returns a thread-safe cached observation.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import queue
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence


ACTION_SCHEMA_ID = "robosuite_osc_pose_normalized_v1"
ACTION_DIM = 7
MAX_CHUNK_SIZE = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve one LIBERO task over HTTP.")
    parser.add_argument("--libero-root", required=True, help="Path to LIBERO-master.")
    parser.add_argument(
        "--benchmark",
        default="libero_object",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    return parser


class _Command:
    def __init__(self, name: str, payload: dict[str, Any]) -> None:
        self.name = name
        self.payload = payload
        self.response: queue.Queue = queue.Queue(maxsize=1)


class LiberoEnvironmentWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._commands: queue.Queue = queue.Queue()
        self._cache_lock = threading.Lock()
        self._cache: dict[str, Any] = {"available": False, "source": "libero_bridge"}
        self._stop_requested = threading.Event()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="libero-environment-owner",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(args.command_timeout):
            raise RuntimeError("Timed out while creating the LIBERO environment.")
        if self._startup_error is not None:
            raise RuntimeError(f"LIBERO startup failed: {self._startup_error}")

    def health(self) -> dict[str, Any]:
        with self._cache_lock:
            observation = dict(self._cache)
        return {
            "status": "ok",
            "service": "libero_bridge",
            "benchmark": observation.get("benchmark"),
            "task_id": observation.get("task_id"),
            "task_language": observation.get("task_language"),
            "action_schema": {
                "id": ACTION_SCHEMA_ID,
                "dimension": ACTION_DIM,
                "range": [-1.0, 1.0],
                "dimensions": [
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_roll",
                    "delta_pitch",
                    "delta_yaw",
                    "gripper",
                ],
                "control_frequency_hz": 20,
            },
            "observation_available": bool(observation.get("available", False)),
        }

    def observe(self) -> dict[str, Any]:
        with self._cache_lock:
            return dict(self._cache)

    def request_stop(self) -> dict[str, Any]:
        self._stop_requested.set()
        return {"status": "stopping", "reason": "STOP_REQUESTED"}

    def submit(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        command = _Command(name, payload)
        self._commands.put(command)
        try:
            success, value = command.response.get(timeout=self.args.command_timeout)
        except queue.Empty as error:
            raise TimeoutError(f"LIBERO command '{name}' timed out.") from error
        if not success:
            raise RuntimeError(str(value))
        return value

    def close(self) -> None:
        if self._thread.is_alive():
            try:
                self.submit("close", {})
            except Exception:
                pass
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            self._create_environment()
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            return
        self._ready.set()
        while True:
            command: _Command = self._commands.get()
            try:
                if command.name == "reset":
                    result = self._reset(int(command.payload.get("init_state_index", 0)))
                elif command.name == "step_chunk":
                    result = self._step_chunk(command.payload.get("actions"))
                elif command.name == "close":
                    self._env.close()
                    command.response.put((True, {"status": "closed"}))
                    return
                else:
                    raise ValueError(f"Unknown LIBERO command: {command.name}")
                command.response.put((True, result))
            except BaseException as error:
                command.response.put((False, repr(error)))

    def _create_environment(self) -> None:
        root = Path(self.args.libero_root).expanduser().resolve()
        _configure_libero(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        benchmark = get_benchmark(self.args.benchmark)()
        if not 0 <= self.args.task_id < benchmark.get_num_tasks():
            raise ValueError(
                f"task-id must be between 0 and {benchmark.get_num_tasks() - 1}."
            )
        self._benchmark = benchmark
        self._task = benchmark.get_task(self.args.task_id)
        self._init_states = benchmark.get_task_init_states(self.args.task_id)
        self._env = OffScreenRenderEnv(
            bddl_file_name=benchmark.get_task_bddl_file_path(self.args.task_id),
            camera_heights=self.args.camera_size,
            camera_widths=self.args.camera_size,
            render_gpu_device_id=self.args.render_gpu_device_id,
        )
        env_action_dim = int(getattr(self._env.env, "action_dim", 0))
        if env_action_dim != ACTION_DIM:
            raise RuntimeError(
                f"Expected a {ACTION_DIM}-D OSC_POSE action, got {env_action_dim}."
            )
        self._episode_id = 0
        self._reset(self.args.init_state_index)

    def _reset(self, init_state_index: int) -> dict[str, Any]:
        if not 0 <= init_state_index < len(self._init_states):
            raise ValueError(
                f"init_state_index must be between 0 and {len(self._init_states) - 1}."
            )
        import numpy as np

        self._stop_requested.clear()
        self._episode_id += 1
        self._step_count = 0
        self._env.reset()
        observation = self._env.set_init_state(self._init_states[init_state_index])
        reward = 0.0
        done = False
        for _ in range(5):
            observation, reward, done, _ = self._env.step(np.zeros(ACTION_DIM))
            self._step_count += 1
        self._update_cache(observation, reward, done)
        return {
            "status": "reset",
            "episode_id": self._episode_id,
            "init_state_index": init_state_index,
            "observation": self.observe(),
        }

    def _step_chunk(self, raw_actions: Any) -> dict[str, Any]:
        actions = _validate_action_chunk(raw_actions)
        self._stop_requested.clear()
        executed = 0
        reward = 0.0
        done = False
        for action in actions:
            if self._stop_requested.is_set():
                break
            observation, reward, done, _ = self._env.step(action)
            self._step_count += 1
            executed += 1
            self._update_cache(observation, reward, done)
            if done:
                break
        stopped = self._stop_requested.is_set()
        return {
            "status": "failed" if stopped else "success",
            "reason": (
                "ACTION_STOPPED"
                if stopped
                else "LIBERO_TASK_SUCCEEDED"
                if done
                else "ACTION_CHUNK_EXECUTED"
            ),
            "command_completed": not stopped,
            "physical_result_verified": False,
            "simulation_result_verified": bool(done),
            "steps_requested": len(actions),
            "steps_executed": executed,
            "episode_success": bool(done),
            "reward": float(reward),
        }

    def _update_cache(self, observation: dict[str, Any], reward: float, done: bool) -> None:
        snapshot = _build_snapshot(
            observation,
            benchmark=self.args.benchmark,
            task_id=self.args.task_id,
            task_language=self._task.language,
            episode_id=self._episode_id,
            step_count=self._step_count,
            reward=reward,
            success=bool(done),
        )
        with self._cache_lock:
            self._cache = snapshot


def _configure_libero(root: Path) -> None:
    package_root = root / "libero" / "libero"
    if not package_root.is_dir():
        raise FileNotFoundError(f"LIBERO package was not found under {package_root}.")
    config_dir = root / ".libero_agent_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(root / "libero" / "datasets"),
        "assets": str(package_root / "assets"),
    }
    (config_dir / "config.yaml").write_text(
        json.dumps(paths, indent=2),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def _validate_action_chunk(raw_actions: Any) -> list[list[float]]:
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("actions must be a non-empty list.")
    if len(raw_actions) > MAX_CHUNK_SIZE:
        raise ValueError(f"action chunk cannot exceed {MAX_CHUNK_SIZE} steps.")
    actions: list[list[float]] = []
    for row_index, row in enumerate(raw_actions):
        if not isinstance(row, list) or len(row) != ACTION_DIM:
            raise ValueError(f"action row {row_index} must contain {ACTION_DIM} values.")
        parsed = [float(value) for value in row]
        if any(
            not math.isfinite(value) or value < -1.0 or value > 1.0
            for value in parsed
        ):
            raise ValueError(
                f"action row {row_index} must contain finite values in [-1, 1]."
            )
        actions.append(parsed)
    return actions


def _build_snapshot(
    observation: dict[str, Any],
    *,
    benchmark: str,
    task_id: int,
    task_language: str,
    episode_id: int,
    step_count: int,
    reward: float,
    success: bool,
) -> dict[str, Any]:
    import numpy as np

    eef_pos = np.asarray(observation.get("robot0_eef_pos", np.zeros(3))).reshape(-1)
    eef_quat = np.asarray(observation.get("robot0_eef_quat", np.zeros(4))).reshape(-1)
    gripper_qpos = np.asarray(
        observation.get("robot0_gripper_qpos", np.zeros(2))
    ).reshape(-1)
    joint_pos = np.asarray(observation.get("robot0_joint_pos", np.zeros(7))).reshape(-1)
    gripper_scalar = float(gripper_qpos.mean()) if gripper_qpos.size else 0.0
    policy_state = [
        *[float(value) for value in eef_pos[:3]],
        *[float(value) for value in eef_quat[:4]],
        gripper_scalar,
    ]

    primary = _encode_image(observation.get("agentview_image"))
    wrist = _encode_image(observation.get("robot0_eye_in_hand_image"))
    timestamp = datetime.now(timezone.utc).isoformat()
    signals = {
        "task_success": bool(success),
        "episode_done": bool(success),
    }
    return {
        "available": True,
        "source": "libero_bridge",
        "timestamp": timestamp,
        "benchmark": benchmark,
        "task_id": task_id,
        "task_language": task_language,
        "episode_id": episode_id,
        "step_count": step_count,
        "reward": float(reward),
        "success": bool(success),
        "done": bool(success),
        "state": policy_state,
        "images": {
            "primary": primary,
            "secondary": primary,
            "wrist": wrist or primary,
        },
        "robot_state": {
            "available": True,
            "source": "libero_bridge",
            "cartesian_pose": {
                "position_xyz_m": [float(value) for value in eef_pos[:3]],
                "orientation_xyzw": [float(value) for value in eef_quat[:4]],
                "coordinate_frame": "libero_world",
            },
            "joint_positions_rad": [float(value) for value in joint_pos[:7]],
            "gripper_qpos": [float(value) for value in gripper_qpos],
            "stopped": False,
        },
        "frames": [
            {
                "timestamp": timestamp,
                "source": "libero_simulator_truth",
                "detections": [],
                "signals": signals,
            }
        ],
        "signals": signals,
    }


def _encode_image(raw_image: Any) -> str:
    if raw_image is None:
        return ""
    import cv2
    import numpy as np

    image = np.asarray(raw_image)
    if image.ndim != 3 or image.shape[-1] != 3:
        return ""
    # LIBERO camera observations are vertically inverted RGB arrays. OpenCV
    # expects BGR, so both transforms are applied before JPEG encoding.
    bgr = image[::-1, :, ::-1]
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    decoded = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Request body must be a JSON object.")
    return decoded


def _write_json(
    handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def build_handler(worker: LiberoEnvironmentWorker):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                _write_json(self, 200, worker.health())
                return
            if self.path == "/observe":
                _write_json(self, 200, worker.observe())
                return
            _write_json(self, 404, {"status": "error", "reason": "NOT_FOUND"})

        def do_POST(self) -> None:
            try:
                payload = _read_json(self)
                if self.path == "/reset":
                    result = worker.submit("reset", payload)
                elif self.path == "/step_chunk":
                    result = worker.submit("step_chunk", payload)
                elif self.path == "/stop":
                    result = worker.request_stop()
                else:
                    _write_json(
                        self, 404, {"status": "error", "reason": "NOT_FOUND"}
                    )
                    return
                _write_json(self, 200, result)
            except Exception as error:
                _write_json(
                    self,
                    400,
                    {"status": "error", "reason": "REQUEST_FAILED", "detail": str(error)},
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[libero-http] {self.address_string()} {format % args}")

    return Handler


def main() -> int:
    args = build_parser().parse_args()
    worker = LiberoEnvironmentWorker(args)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(worker))
    server.daemon_threads = True
    health = worker.health()
    print(f"LIBERO bridge: http://{args.host}:{args.port}")
    print(f"Health: http://{args.host}:{args.port}/health")
    print(f"Task: {health.get('task_language')}")
    print(f"Action schema: {ACTION_SCHEMA_ID} ({ACTION_DIM}D, normalized)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
