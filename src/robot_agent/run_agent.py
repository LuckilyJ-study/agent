from __future__ import annotations

import argparse
from collections.abc import Sequence

from .planner import OllamaPlanner, QwenApiPlanner, RuleBasedPlanner, create_default_planner
from .runtime import build_agent_runtime
from .persistence import JsonTaskStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the generic closed-loop robot Agent.")
    parser.add_argument("task", nargs="*", help="Natural-language robot task.")
    parser.add_argument(
        "--planner",
        choices=("env", "rule_based", "ollama", "qwen_api"),
        default="rule_based",
    )
    parser.add_argument("--max-replans", type=int, default=3)
    parser.add_argument("--task-id", help="Stable task ID used for persistence/resume.")
    parser.add_argument("--state-dir", help="Directory for durable task snapshots.")
    parser.add_argument("--resume", action="store_true", help="Resume --task-id from --state-dir.")
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Disable simulated policies. Real controller/policies must be injected in code.",
    )
    return parser


def _planner(name: str):
    if name == "env":
        return create_default_planner()
    if name == "rule_based":
        return RuleBasedPlanner()
    if name == "ollama":
        return OllamaPlanner()
    if name == "qwen_api":
        return QwenApiPlanner()
    raise ValueError(f"Unsupported planner: {name}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    task = " ".join(args.task).strip() or input("Robot task: ").strip()
    if not task:
        raise SystemExit("Task cannot be empty.")
    if args.resume and (not args.task_id or not args.state_dir):
        raise SystemExit("--resume requires --task-id and --state-dir.")
    task_store = JsonTaskStore(args.state_dir) if args.state_dir else None
    runtime = build_agent_runtime(
        _planner(args.planner),
        dry_run=not args.no_dry_run,
        max_replans=args.max_replans,
        task_store=task_store,
    )
    result = runtime.agent.run_safe(task, task_id=args.task_id, resume=args.resume)
    print("Task:", task)
    print("Status:", result.status)
    print("Task ID:", result.memory.task_id)
    print("Verification mode: command-only (perception placeholder)")
    print("Completed steps:")
    for step in result.memory.completed_steps:
        print(
            f"- {step['step_id']}: {step['action_type']}({step['target']}) "
            f"via {step['executor']}"
        )
    if result.reason:
        print("Reason:", result.reason)


if __name__ == "__main__":
    main()
