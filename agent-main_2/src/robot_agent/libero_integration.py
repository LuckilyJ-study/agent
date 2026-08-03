from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .action_safety import ActionChunkGuard, ActionChunkSafetyLimits
from .domain import WorldState
from .state import PlanStep
from .task_verifier import TaskVerification


LIBERO_ACTION_SCHEMA = "robosuite_osc_pose_normalized_v1"
LIBERO_ACTION_DIM = 7


class LiberoServiceError(RuntimeError):
    """Raised when the standalone LIBERO bridge cannot serve a request."""


@dataclass
class LiberoHttpClient:
    endpoint: str = "http://127.0.0.1:8770"
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint.rstrip("/")
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("LIBERO endpoint must start with http:// or https://.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def observe(self) -> dict[str, Any]:
        return self._request("GET", "/observe")

    def reset(self, init_state_index: int = 0) -> dict[str, Any]:
        return self._request(
            "POST",
            "/reset",
            {"init_state_index": int(init_state_index)},
        )

    def execute_action_chunk(
        self, actions: Sequence[Sequence[float]]
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/step_chunk",
            {"actions": [[float(value) for value in row] for row in actions]},
        )

    def stop(self) -> dict[str, Any]:
        return self._request("POST", "/stop", {})

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise LiberoServiceError(
                f"LIBERO bridge returned HTTP {error.code}: {error_body}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LiberoServiceError(f"LIBERO bridge request failed: {error}") from error
        if not isinstance(decoded, dict):
            raise LiberoServiceError("LIBERO bridge returned a non-object response.")
        return decoded


@dataclass
class LiberoPerceptionProvider:
    """Expose the bridge's cached images and proprioception to the Agent.

    The bridge owns MuJoCo. This provider only reads a JSON cache, so the
    Agent's monitor thread never calls the simulator concurrently with step().
    """

    client: LiberoHttpClient
    hardware_ready: bool = field(default=False, init=False)
    simulation_ready: bool = field(default=True, init=False)
    supports_target_configuration: bool = field(default=False, init=False)
    supports_localization: bool = field(default=False, init=False)
    localization_modes: frozenset[str] = field(default_factory=frozenset, init=False)

    def observe(self) -> dict[str, Any]:
        observation = self.client.observe()
        observation.setdefault("available", True)
        observation.setdefault("source", "libero_bridge")
        observation.setdefault("frames", [])
        return observation

    def configure_targets(self, labels: Sequence[str]) -> None:
        # LIBERO truth and RGB frames are exposed separately. Object detections
        # should come from a vision provider, not be fabricated here.
        return None


@dataclass
class LiberoActionChunkController:
    """Send normalized OSC_POSE action chunks to the LIBERO bridge."""

    client: LiberoHttpClient
    action_schema_confirmed: bool = False
    action_guard: ActionChunkGuard | None = None
    hardware_ready: bool = field(default=False, init=False)
    simulation_ready: bool = field(default=True, init=False)
    _last_action: list[float] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.action_guard is None:
            self.action_guard = build_libero_action_guard()

    def execute_action_chunk(
        self, actions: Sequence[Sequence[float]]
    ) -> dict[str, Any]:
        if not self.action_schema_confirmed:
            return {
                "status": "failed",
                "reason": "LIBERO_ACTION_SCHEMA_CONFIRMATION_REQUIRED",
                "command_completed": False,
                "physical_result_verified": False,
                "details": {
                    "required_schema": LIBERO_ACTION_SCHEMA,
                    "note": (
                        "Confirm or adapt the policy output semantics before "
                        "sending actions to LIBERO. Equal dimensions are not enough."
                    ),
                },
            }
        assert self.action_guard is not None
        safety = self.action_guard.check(
            actions,
            reference_values=self._last_action,
        )
        if not safety.safe:
            return {
                "status": "failed",
                "reason": safety.reason,
                "command_completed": False,
                "physical_result_verified": False,
                "details": {"action_safety": safety.details or {}},
            }
        health = self.client.health()
        validate_libero_action_schema(health)
        result = self.client.execute_action_chunk(actions)
        if actions:
            self._last_action = [float(value) for value in actions[-1]]
        return result

    def stop(self) -> None:
        self.client.stop()

    def get_action_state(self) -> Sequence[float] | None:
        return list(self._last_action) if self._last_action is not None else None

    def get_state(self) -> dict[str, Any]:
        observation = self.client.observe()
        state = dict(observation.get("robot_state") or {})
        state.setdefault("available", bool(observation.get("available", False)))
        state.setdefault("source", "libero_bridge")
        state["simulation"] = True
        return state


@dataclass
class LiberoTaskVerifier:
    """Use LIBERO's BDDL success predicate as the final task oracle."""

    client: LiberoHttpClient

    def verify(
        self,
        original_task: str,
        completed_steps: list[PlanStep],
        world_state: WorldState,
    ) -> TaskVerification:
        observation = self.client.observe()
        if bool(observation.get("success", False)):
            return TaskVerification(
                True,
                "LIBERO BDDL goal predicate is satisfied.",
                "simulation",
            )
        return TaskVerification(
            False,
            (
                "Agent steps ended, but LIBERO's BDDL goal predicate is not "
                "satisfied. The policy needs another rollout or replanning."
            ),
            "simulation",
        )


def validate_libero_action_schema(health: dict[str, Any]) -> None:
    schema = dict(health.get("action_schema") or {})
    schema_id = str(schema.get("id") or "")
    action_dim = int(schema.get("dimension", 0) or 0)
    if schema_id != LIBERO_ACTION_SCHEMA or action_dim != LIBERO_ACTION_DIM:
        raise LiberoServiceError(
            "LIBERO action schema mismatch: "
            f"expected {LIBERO_ACTION_SCHEMA}/{LIBERO_ACTION_DIM}, "
            f"received {schema_id or 'missing'}/{action_dim}."
        )


def build_libero_action_guard(max_chunk_size: int = 100) -> ActionChunkGuard:
    """Build the simulation-only guard for robosuite normalized OSC_POSE.

    These values are control commands, not joint positions. A full -1 to +1
    transition is valid, so per-command change is bounded by 2 rather than the
    tighter joint-space profile used for physical robots.
    """

    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be positive.")
    return ActionChunkGuard(
        ActionChunkSafetyLimits(
            semantics="normalized",
            lower_bounds=(-1.0,) * LIBERO_ACTION_DIM,
            upper_bounds=(1.0,) * LIBERO_ACTION_DIM,
            max_step_changes=(2.0,) * LIBERO_ACTION_DIM,
            max_cumulative_changes=(2.0 * max_chunk_size,) * LIBERO_ACTION_DIM,
            max_chunk_size=max_chunk_size,
            require_reference=False,
            hardware_approved=False,
            profile_name="libero-osc-pose-simulation-only",
            dimension_names=(
                "delta_x",
                "delta_y",
                "delta_z",
                "delta_roll",
                "delta_pitch",
                "delta_yaw",
                "gripper",
            ),
        )
    )


def validate_normalized_action(action: Sequence[float]) -> list[float]:
    """Small reusable validation helper for clients and tests."""

    if len(action) != LIBERO_ACTION_DIM:
        raise ValueError(f"LIBERO action must contain {LIBERO_ACTION_DIM} values.")
    parsed = [float(value) for value in action]
    if any(not math.isfinite(value) or value < -1.0 or value > 1.0 for value in parsed):
        raise ValueError("LIBERO action values must be finite and within [-1, 1].")
    return parsed
