from __future__ import annotations

from .graph import build_robot_graph
from .planner import create_default_planner


def main() -> None:
    graph = build_robot_graph(planner=create_default_planner())
    result = graph.invoke(
        {
            "user_task": "Make a pizza by picking dough and placing it on a tray.",
            "max_retries": 2,
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
