from __future__ import annotations

from .events import record_event
from .state import RobotState


RETRYABLE_FAILURES = {"GRASP_FAILED", "TARGET_NOT_VISIBLE", "TRANSIENT_MOTION_ERROR"}


def decide_recovery(state: RobotState) -> dict:
    """Use local recovery first; hand complex failures back to the planning Agent."""
    feedback = state["feedback"]
    can_retry = (
        feedback["reason"] in RETRYABLE_FAILURES
        and state["retry_count"] < state["max_retries"]
    )
    action = "retry" if can_retry else "replan"
    message = (
        "Recovery selected a local retry."
        if can_retry
        else "Recovery escalated the failure to the planning Agent."
    )
    update: dict = {"recovery_action": action}
    update.update(
        record_event(
            state,
            "recovery.decided",
            message,
            step_id=state["current_step"]["id"],
            data={"reason": feedback["reason"], "action": action},
        )
    )
    return update
