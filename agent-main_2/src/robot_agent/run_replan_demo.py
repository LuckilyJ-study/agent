from __future__ import annotations

from typing import Any

from .graph import build_robot_graph
from .planner import create_default_planner
from .state import PlanStep


class ReplanThenSucceedGateway:
    """Simulate a scene change once, then allow the revised plan to finish."""

    def __init__(self) -> None:
        self.has_failed = False

    def execute(self, task_text: str, step: PlanStep, retry_count: int) -> dict[str, Any]:
        if not self.has_failed:
            self.has_failed = True
            return {
                "status": "failed",
                "reason": "COLLISION_RISK",
                "details": {"source": "simulated_safety_monitor"},
            }
        return {
            "status": "success",
            "reason": "OK",
            "details": {"source": "simulated_camera", "task_text": task_text},
        }


def main() -> None:
    graph = build_robot_graph(planner=create_default_planner(), gateway=ReplanThenSucceedGateway())
    result = graph.invoke(
        {
            "user_task": "Make a pizza by picking dough and placing it on a tray.",
            "max_retries": 1,
            "max_replans": 1,
            "history": [],
            "events": [],
        }
    )
    print("Final status:", result["status"])
    print("Execution events:")
    for event in result["events"]:
        print(f"- {event['type']}: {event['message']}")


if __name__ == "__main__":
    main()
