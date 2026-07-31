from __future__ import annotations

from typing import Any

from .state import RobotEvent, RobotState


def record_event(
    state: RobotState,
    event_type: str,
    message: str,
    step_id: int | None = None,
    data: dict[str, Any] | None = None,
) -> dict:
    """Return append-only human-readable and structured execution history."""
    event: RobotEvent = {
        "type": event_type,
        "message": message,
        "step_id": step_id,
        "data": data or {},
    }
    return {
        "history": [*state.get("history", []), message],
        "events": [*state.get("events", []), event],
    }
