from __future__ import annotations

import argparse
from collections.abc import Sequence

from .gateway import Pi05ServiceGateway
from .planner import Planner, RuleBasedPlanner, create_default_planner
from .simple_workflow import run_simple_workflow
from .state import RobotState

DEFAULT_TASK = "Make a pizza by picking dough and placing it on a tray."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop demo: user task -> Agent plan -> structured observation -> "
            "Pi05 policy -> action chunk -> robot execution (printed) -> feedback."
        )
    )
    parser.add_argument("task", nargs="*", help=f"Natural-language robot task (default: {DEFAULT_TASK!r})")
    parser.add_argument(
        "--planner",
        choices=("rule_based", "env"),
        default="rule_based",
        help="'env' uses ROBOT_AGENT_PLANNER_PROVIDER (ollama/qwen_api).",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-replans", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    user_task = " ".join(args.task).strip() or DEFAULT_TASK
    planner: Planner = RuleBasedPlanner() if args.planner == "rule_based" else create_default_planner()
    gateway = Pi05ServiceGateway()

    print("=" * 72)
    print("Pi05 closed-loop demo")
    print("user task -> plan -> structured observation -> Pi05 policy -> robot -> feedback")
    print("=" * 72)
    print(f"[Demo] user task: {user_task}")
    print(f"[Demo] Pi05 endpoint: {gateway.endpoint}")

    initial_state: RobotState = {
        "user_task": user_task,
        "max_retries": args.max_retries,
        "max_replans": args.max_replans,
        "history": [],
        "events": [],
    }
    result = run_simple_workflow(initial_state, planner, gateway=gateway)

    print("-" * 72)
    print("Final status:", result["status"])
    if "plan" in result:
        print("Skill plan:")
        for step in result["plan"]:
            print(f"- step {step['id']}: {step['skill']}({step['target']}) -> {step['expected_result']}")
    print("Execution events:")
    for event in result["events"]:
        print(f"- {event['type']}: {event['message']}")


if __name__ == "__main__":
    main()
