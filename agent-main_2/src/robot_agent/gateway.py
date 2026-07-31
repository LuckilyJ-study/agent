from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .action_safety import (
    ActionChunkGuard,
    ActionChunkSafetyLimits,
)
from .motion import parse_relative_motion_target
from .observation import summarize_observation
from .robot_controller import PrintRobotController, RobotController
from .state import PlanStep


class RobotGateway(Protocol):
    """Boundary between orchestration and Pi05 plus physical robot control."""

    def execute(self, task_text: str, step: PlanStep, retry_count: int, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one skill request and return a normalized execution result."""


class SimulatedGateway:
    """Deterministic gateway used to test retries without robot hardware."""

    def execute(self, task_text: str, step: PlanStep, retry_count: int, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        observation_summary = summarize_observation(observation)
        first_pick_attempt = step["skill"] == "pick" and retry_count == 0
        if first_pick_attempt:
            return {
                "status": "failed",
                "reason": "GRASP_FAILED",
                "details": {"source": "simulated_camera", "grasp_detected": False},
            }
        if step["skill"] == "move_relative":
            motion = parse_relative_motion_target(step["target"])
            action = {
                "type": "move",
                "target": step["target"],
                "direction": motion.direction,
                "distance_cm": motion.distance_cm,
            }
            return {
                "status": "success",
                "reason": "OK",
                "details": {
                    "source": "simulated_robot",
                    "direction": motion.direction,
                    "distance_cm": motion.distance_cm,
                    "task_text": task_text,
                    "observation_summary": observation_summary,
                    "action": action,
                },
            }
        return {
            "status": "success",
            "reason": "OK",
            "details": {"source": "simulated_camera", "task_text": task_text, "observation_summary": observation_summary},
        }


class LeRobotGateway:
    """Pi05-oriented gateway scaffold that can be wired to the LeRobot policy stack."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._pi05_backend: dict[str, Any] | None = None
        self._pi05_error: str | None = None

    def _ensure_pi05_backend(self) -> dict[str, Any] | None:
        if self._pi05_backend is not None or self._pi05_error is not None:
            return self._pi05_backend

        candidate_roots = [
            Path(__file__).resolve().parents[4] / "lerobot",
            Path(__file__).resolve().parents[3] / "lerobot",
        ]
        for candidate in candidate_roots:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        try:
            from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
            from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
        except Exception as error:  # pragma: no cover - import path can vary by environment
            self._pi05_error = f"Pi05 import unavailable: {error}"
            return None

        try:
            config = PI05Config()
            config.device = self.device
            self._pi05_backend = {
                "policy_class": PI05Policy.__name__,
                "config_class": PI05Config.__name__,
                "device": config.device,
                "status": "imported",
            }
            return self._pi05_backend
        except Exception as error:  # pragma: no cover - config setup may depend on deps
            self._pi05_error = f"Pi05 config setup failed: {error}"
            return None

    def execute(self, task_text: str, step: PlanStep, retry_count: int, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        motion = None
        if step["skill"] == "move_relative":
            motion = parse_relative_motion_target(step["target"])

        backend = self._ensure_pi05_backend()
        details: dict[str, Any] = {
            "source": "lerobot_pi05_gateway",
            "task_text": task_text,
            "skill": step["skill"],
            "target": step["target"],
            "retry_count": retry_count,
            "observation_summary": summarize_observation(observation),
            "pi05_backend": backend or {"status": "unavailable"},
        }
        if motion is not None:
            details.update({"direction": motion.direction, "distance_cm": motion.distance_cm})

        action = {
            "type": "print_only",
            "message": f"[Pi05 stub] would execute {step['skill']} target={step['target']}",
            "observation_summary": summarize_observation(observation),
        }
        if motion is not None:
            action.update({"direction": motion.direction, "distance_cm": motion.distance_cm})

        if backend is None:
            print("[Pi05 gateway stub]", action)
            return {
                "status": "success",
                "reason": "PI05_GATEWAY_READY",
                "details": {
                    **details,
                    "action": action,
                    "note": self._pi05_error or "Pi05 backend not initialized; using gateway scaffold fallback.",
                },
            }

        print("[Pi05 gateway stub]", action)
        return {
            "status": "success",
            "reason": "PI05_GATEWAY_READY",
            "details": {
                **details,
                "action": action,
                "note": "Pi05 backend was imported and the gateway scaffold is ready for real policy execution.",
            },
        }


class Pi05ServiceGateway:
    """Gateway that runs the full skill pipeline:

    structured observation -> Pi05 policy inference (HTTP service) ->
    action chunk -> robot execution (PrintRobotController stand-in) ->
    normalized result for feedback.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        timeout_seconds: int = 60,
        robot: RobotController | None = None,
        exp_id: str = "agent_master",
        stub_chunk_size: int = 50,
        stub_action_dim: int = 7,
        allow_stub_actions: bool = False,
        action_guard: ActionChunkGuard | None = None,
    ) -> None:
        self.endpoint = endpoint or os.getenv(
            "ROBOT_AGENT_PI05_ENDPOINT", "http://127.0.0.1:7777/predict"
        )
        self.timeout_seconds = timeout_seconds
        self.robot = robot or PrintRobotController()
        self.exp_id = exp_id
        self.stub_chunk_size = stub_chunk_size
        self.stub_action_dim = stub_action_dim
        self.allow_stub_actions = allow_stub_actions
        self.action_guard = action_guard or ActionChunkGuard(
            ActionChunkSafetyLimits.normalized_simulation(stub_action_dim)
        )

    def execute(self, task_text: str, step: PlanStep, retry_count: int, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        observation = observation or {}
        actions, policy_note, policy_ok = self._query_policy(task_text, observation)
        print(f"[Pi05ServiceGateway] skill={step['skill']} target={step['target']} -> {policy_note}")

        if not policy_ok and not self.allow_stub_actions:
            return {
                "status": "failed",
                "reason": "PI05_SERVICE_UNAVAILABLE",
                "details": {
                    "source": "pi05_service_gateway",
                    "policy_endpoint": self.endpoint,
                    "policy_note": policy_note,
                    "robot_execution": None,
                    "note": "Fail-closed: no stub action was sent to the robot.",
                },
            }
        if bool(getattr(self.robot, "hardware_ready", False)) and not (
            self.action_guard.limits.hardware_approved
        ):
            return {
                "status": "failed",
                "reason": "ACTION_SAFETY_CONFIG_REQUIRED",
                "command_completed": False,
                "physical_result_verified": False,
                "details": {
                    "source": "pi05_service_gateway",
                    "policy_endpoint": self.endpoint,
                    "policy_note": policy_note,
                    "robot_execution": None,
                    "action_safety": {
                        "profile_name": self.action_guard.limits.profile_name,
                        "hardware_approved": False,
                    },
                    "note": (
                        "Fail-closed: a hardware-ready controller requires an "
                        "explicit action schema and robot-specific limits."
                    ),
                },
            }

        reference_values = None
        reference_getter = getattr(self.robot, "get_action_state", None)
        if callable(reference_getter):
            try:
                reference_values = reference_getter()
            except Exception as error:
                return {
                    "status": "failed",
                    "reason": "ACTION_REFERENCE_UNAVAILABLE",
                    "command_completed": False,
                    "physical_result_verified": False,
                    "details": {
                        "source": "pi05_service_gateway",
                        "robot_execution": None,
                        "action_safety": {"exception": repr(error)},
                    },
                }
        action_safety = self.action_guard.check(
            actions,
            reference_values=reference_values,
        )
        if not action_safety.safe:
            return {
                "status": "failed",
                "reason": action_safety.reason,
                "command_completed": False,
                "physical_result_verified": False,
                "details": {
                    "source": "pi05_service_gateway",
                    "policy_endpoint": self.endpoint,
                    "policy_note": policy_note,
                    "robot_execution": None,
                    "action_safety": action_safety.details or {},
                    "note": "Fail-closed: unsafe policy actions were not sent to the robot.",
                },
            }
        robot_summary = self.robot.execute_action_chunk(actions)

        first_action = [float(value) for value in actions[0]] if actions else []
        action = {
            "type": "pi05_action_chunk",
            "chunk_size": len(actions),
            "action_dim": len(actions[0]) if actions else 0,
            "first_action": first_action,
        }
        details = {
            "source": "pi05_service_gateway",
            "skill": step["skill"],
            "target": step["target"],
            "retry_count": retry_count,
            "policy_endpoint": self.endpoint,
            "policy_note": policy_note,
            "observation_summary": summarize_observation(observation),
            "action": action,
            "action_safety": action_safety.details or {},
            "robot_execution": robot_summary,
        }
        if policy_ok:
            return {"status": "success", "reason": "PI05_ACTIONS_EXECUTED", "details": details}
        return {
            "status": "success",
            "reason": "PI05_STUB_ACTIONS_EXECUTED",
            "details": {**details, "note": "Pi05 service unavailable; executed deterministic stub actions instead."},
        }

    def stop(self) -> None:
        stop = getattr(self.robot, "stop", None)
        if callable(stop):
            stop()

    def _query_policy(self, task_text: str, observation: dict[str, Any]) -> tuple[list[list[float]], str, bool]:
        payload = self._build_service_payload(task_text, observation)
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            actions = [[float(value) for value in row] for row in decoded["act"]]
            if not actions:
                raise ValueError("Pi05 service returned an empty action chunk")
            return actions, "pi05 policy returned an action chunk", True
        except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            note = f"pi05 service call failed ({error}); using stub actions"
            stub = [[0.0] * self.stub_action_dim for _ in range(self.stub_chunk_size)]
            return stub, note, False

    def _build_service_payload(self, task_text: str, observation: dict[str, Any]) -> dict[str, Any]:
        """Match the Pi05 Flask service contract: images[0] is a list of primary
        frames, images[1]/images[2] are single secondary/wrist frames."""
        images = observation.get("images") or {}
        state = observation.get("state") or []
        return {
            "task": task_text,
            "images": [
                [images.get("primary", "")],
                images.get("secondary", ""),
                images.get("wrist", ""),
            ],
            "state": [float(value) for value in state],
            "exp_id": self.exp_id,
        }
