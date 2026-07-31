from __future__ import annotations

from .graph import build_robot_graph
from .planner import RuleBasedPlanner


def main() -> None:
    graph = build_robot_graph(planner=RuleBasedPlanner())
    result = graph.invoke(
        {
            "user_task": "机械臂向左移动0.5厘米",
            "max_retries": 0,
            "max_replans": 0,
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
