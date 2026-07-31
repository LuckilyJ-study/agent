from __future__ import annotations

import inspect

from .events import record_event
from .gateway import RobotGateway, SimulatedGateway
from .safety import ExecutionPolicy
from .state import RobotState


def execute_pi05(
    state: RobotState,
    gateway: RobotGateway | None = None,
    policy: ExecutionPolicy | None = None,
) -> dict:
    """Execute through the gateway after the safety policy grants permission."""
    selected_policy = policy or ExecutionPolicy(max_retries=state.get("max_retries", 2))
    allowed, reason = selected_policy.can_execute(state)
    if not allowed:
        update: dict = {
            "status": "blocked_by_safety",
            "execution_result": {"status": "failed", "reason": "SAFETY_BLOCKED", "details": {"policy": reason}},
        }
        update.update(record_event(state, "execution.blocked", reason, step_id=state["current_step"]["id"]))
        return update

    selected_gateway = gateway or SimulatedGateway()
    observation = state.get("pi05_observation")
    execute_parameters = inspect.signature(selected_gateway.execute).parameters
    if "observation" in execute_parameters:
        result = selected_gateway.execute(
            state["pi05_task_text"],
            state["current_step"],
            state["retry_count"],
            observation=observation,
        )
    else:  # Backward compatibility for existing three-argument gateways.
        result = selected_gateway.execute(
            state["pi05_task_text"], state["current_step"], state["retry_count"]
        )
    action = None
    if isinstance(result.get("details"), dict):
        action = result["details"].get("action")

    update: dict = {"execution_result": result}
    if action is not None:
        update["pi05_action"] = action
    update.update(
        record_event(
            state,
            "execution.finished",
            f"Gateway returned {result['status']}.",
            step_id=state["current_step"]["id"],
            data={"reason": result["reason"], "action": action},
        )
    )
    return update
