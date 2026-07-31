from __future__ import annotations

from .events import record_event
from .state import RobotState


def evaluate_feedback(state: RobotState) -> dict:
    """Translate visual/sensor verification into a recovery-ready result."""
    result = state["execution_result"]
    feedback = {
        "success": result["status"] == "success",
        "reason": result["reason"],
        "details": result.get("details", {}),
    }
    if "pi05_action" in state:
        feedback["details"] = {**feedback["details"], "pi05_action": state["pi05_action"]}
    event_type = "feedback.success" if feedback["success"] else "feedback.failure"
    update: dict = {"feedback": feedback}
    update.update(
        record_event(
            state,
            event_type,
            f"Feedback: {feedback['reason']}.",
            step_id=state["current_step"]["id"],
            data=feedback["details"],
        )
    )
    return update
