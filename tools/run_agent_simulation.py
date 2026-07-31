from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSPACE / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robot_agent.persistence import JsonTaskStore
from robot_agent.runtime import build_agent_runtime
from robot_agent.simulation import build_pick_and_place_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Robot Agent end-to-end without hardware."
    )
    parser.add_argument(
        "--task",
        help="Task text. If omitted, ask the user interactively.",
    )
    parser.add_argument(
        "--pick-failures",
        type=int,
        default=1,
        help="Number of simulated pick failures before success.",
    )
    parser.add_argument("--state-dir", help="Optional directory for JSON task snapshots.")
    parser.add_argument("--task-id", default="simulation-demo")
    parser.add_argument("--json", action="store_true", help="Print the complete final snapshot.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("=== Simulated Robot Agent ===")
    print("Simulation capability: 把 <物体> 放到 <目标> / Put <object> on <destination>.")
    task = (args.task or "").strip()
    if not task:
        task = input("\n请输入机械臂任务: ").strip()
    if not task:
        print("任务不能为空。")
        return 2

    simulation = build_pick_and_place_simulation(
        pick_failures=args.pick_failures,
    )
    task_store = JsonTaskStore(args.state_dir) if args.state_dir else None
    runtime = build_agent_runtime(
        simulation.planner,
        controller=simulation.controller,
        policies=simulation.policies,
        task_store=task_store,
        dry_run=False,
    )
    result = runtime.agent.run_safe(task, task_id=args.task_id)

    print("\n=== Execution Result ===")
    print("Task:", task)
    print("Task ID:", result.memory.task_id)
    print("Status:", result.status)
    print("Resolved object:", simulation.planner.object_name)
    print("Resolved destination:", simulation.planner.destination)
    print("Pick policy calls:", simulation.pick_backend.calls)
    print("Replans:", result.memory.replan_count)
    print("\nCompleted steps:")
    for step in result.memory.completed_steps:
        print(
            f"  {step['step_id']:>3}  {step['action_type']:<14} "
            f"target={step['target']} executor={step['executor']}"
        )
    print("\nEvent timeline:")
    for event in result.memory.events:
        print(f"  {event['timestamp']}  {event['type']}  {event['data']}")
    print("\nFinal symbolic world state:")
    print(json.dumps(result.memory.world_state, ensure_ascii=False, indent=2))
    print("\nVerification scope:", result.memory.task_verification)
    if result.reason:
        print("Failure reason:", result.reason)
    if args.json:
        print("\nComplete TaskMemory snapshot:")
        print(json.dumps(result.memory.snapshot(), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
