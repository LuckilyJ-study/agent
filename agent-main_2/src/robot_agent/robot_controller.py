from __future__ import annotations

from typing import Any, Protocol, Sequence


class RobotController(Protocol):
    """Executes a Pi05 action chunk on the robot, one control step at a time."""

    hardware_ready: bool

    def execute_action_chunk(self, actions: Sequence[Sequence[float]]) -> dict[str, Any]:
        """Send every action in the chunk to the robot and return a summary."""

    def stop(self) -> None:
        """Stop an in-flight action chunk as quickly as the driver permits."""

    def get_action_state(self) -> Sequence[float] | None:
        """Return values in the same schema as one policy action, when available."""


class PrintRobotController:
    """Stands in for the real robot driver: prints each action instead of
    sending it to hardware. Swap in a LeRobot-based controller later without
    touching the orchestration layer."""

    def __init__(self, max_lines: int = 12) -> None:
        self.max_lines = max_lines
        self.stopped = False
        self.hardware_ready = False
        self._last_action: list[float] | None = None

    def execute_action_chunk(self, actions: Sequence[Sequence[float]]) -> dict[str, Any]:
        self.stopped = False
        total = len(actions)
        print(f"[Robot] received action chunk: {total} control steps")
        shown = min(total, self.max_lines)
        for index in range(shown):
            print(f"[Robot] step {index + 1:02d}/{total} send action={_format_action(actions[index])}")
        if total > shown:
            print(f"[Robot] ... {total - shown} more steps executed (truncated from log)")
        print(f"[Robot] chunk finished: {total} actions executed")
        self._last_action = (
            [float(value) for value in actions[-1]] if total else self._last_action
        )
        return {
            "controller": "print_robot_controller",
            "steps_total": total,
            "steps_logged": shown,
            "last_action": [float(value) for value in actions[-1]] if total else [],
        }

    def stop(self) -> None:
        self.stopped = True

    def get_action_state(self) -> Sequence[float] | None:
        return list(self._last_action) if self._last_action is not None else None


def _format_action(action: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):+.4f}" for value in action) + "]"
