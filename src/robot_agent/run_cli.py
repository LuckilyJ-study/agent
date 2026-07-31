from __future__ import annotations

import argparse
from collections.abc import Sequence

from .planner import OllamaPlanner, Planner, QwenApiPlanner, RuleBasedPlanner, create_default_planner
from .simple_workflow import run_simple_workflow
from .state import RobotState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the robot Agent with a natural-language task.")
    parser.add_argument(
        "task",
        nargs="*",
        help="Natural-language robot task, for example: move the arm left by 0.5 cm",
    )
    parser.add_argument(
        "--planner",
        choices=("env", "rule_based", "ollama", "qwen_api"),
        default="env",
        help="Planner backend. 'env' uses ROBOT_AGENT_PLANNER_PROVIDER.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-replans", type=int, default=1)
    parser.add_argument(
        "--engine",
        choices=("auto", "simple", "langgraph"),
        default="auto",
        help="Workflow engine. 'simple' does not require the langgraph package.",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the validated skill plan after execution.",
    )
    return parser


def select_planner(name: str) -> Planner:
    if name == "env":
        return create_default_planner()
    if name == "rule_based":
        return RuleBasedPlanner()
    if name == "ollama":
        return OllamaPlanner()
    if name == "qwen_api":
        return QwenApiPlanner()
    raise ValueError(f"Unsupported planner: {name}")


def run_workflow(
    user_task: str,
    planner: Planner,
    max_retries: int,
    max_replans: int,
    engine: str,
) -> RobotState:
    initial_state: RobotState = {
        "user_task": user_task,
        "max_retries": max_retries,
        "max_replans": max_replans,
        "history": [],
        "events": [],
    }
    if engine == "simple":
        return run_simple_workflow(initial_state, planner)

    try:
        from .graph import build_robot_graph
    except ModuleNotFoundError as error:
        if error.name != "langgraph" or engine == "langgraph":
            raise
        print("LangGraph is not installed; falling back to the simple workflow engine.")
        return run_simple_workflow(initial_state, planner)

    graph = build_robot_graph(planner=planner)
    return graph.invoke(initial_state)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    user_task = " ".join(args.task).strip()
    if not user_task:
        user_task = input("Robot task: ").strip()
    if not user_task:
        raise SystemExit("Task cannot be empty.")

    result = run_workflow(
        user_task=user_task,
        planner=select_planner(args.planner),
        max_retries=args.max_retries,
        max_replans=args.max_replans,
        engine=args.engine,
    )

    print("User task:", user_task)
    print("Final status:", result["status"])
    if args.show_plan and "plan" in result:
        print("Skill plan:")
        for step in result["plan"]:
            print(
                f"- step {step['id']}: {step['skill']}({step['target']})"
                f" -> {step['expected_result']}"
            )
    print("Execution events:")
    for event in result["events"]:
        print(f"- {event['type']}: {event['message']}")


if __name__ == "__main__":
    main()
