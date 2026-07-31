from __future__ import annotations

from .events import record_event
from .state import RobotState


def choose_current_step(state: RobotState) -> dict:
    """Select a validated skill; the scheduler never creates motor commands."""
    index = state["current_step_index"]
    plan = state["plan"]
    if index >= len(plan):
        update: dict = {"status": "completed"}
        update.update(record_event(state, "task.completed", "All plan steps completed."))
        return update

    step = plan[index]
    update = {"current_step": step, "status": "running"}
    update.update(
        record_event(
            state,
            "skill.scheduled",
            f"Scheduled step {step['id']}: {step['skill']}({step['target']}).",
            step_id=step["id"],
        )
    )
    return update


def advance_step(state: RobotState) -> dict:
    step = state["current_step"]
    update: dict = {
        "current_step_index": state["current_step_index"] + 1,
        "retry_count": 0,
    }
    update.update(record_event(state, "skill.succeeded", f"Step {step['id']} succeeded.", step_id=step["id"]))
    return update


def prepare_retry(state: RobotState) -> dict:
    step = state["current_step"]
    next_retry = state["retry_count"] + 1
    update: dict = {"retry_count": next_retry}
    update.update(
        record_event(
            state,
            "skill.retry",
            f"Retrying step {step['id']} after {state['feedback']['reason']}.",
            step_id=step["id"],
            data={"retry_count": next_retry},
        )
    )
    return update
