"""Run the Qwen-planned Robot Agent with simulated execution backends.

No robot or camera is required. Qwen emits high-level skills only. The local
whitelist and Router select a dry-run robot primitive or policy backend.

PowerShell:
    $env:QWEN_API_KEY = "your-key"
    python run_agent_simulation.py

Offline regression fallback:
    python run_agent_simulation.py --planner scripted
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parent
SOURCE_ROOT = WORKSPACE / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robot_agent.action_executors import (
    DryRunRobotController,
    PolicyRegistry,
)
from robot_agent.capabilities import (
    CapabilityRegistry,
    default_capabilities,
)
from robot_agent.demo_scenes import SceneSelectionError, select_demo_scene
from robot_agent.persistence import JsonTaskStore
from robot_agent.planner import PlannerServiceError, QwenApiPlanner
from robot_agent.runtime import build_agent_runtime
from robot_agent.simulation import build_pick_and_place_simulation
from robot_agent.skill_grounding import SceneSkillPlanner
from robot_agent.task_verifier import SymbolicGoalTaskVerifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use Qwen to plan a task, then execute it on a dry-run robot."
    )
    parser.add_argument(
        "--planner",
        choices=("qwen_api", "scripted"),
        default="qwen_api",
        help="qwen_api is the real intelligent planner; scripted is offline regression only.",
    )
    parser.add_argument("--task", help="If omitted, ask for the task interactively.")
    parser.add_argument("--model", default=os.getenv("ROBOT_AGENT_API_MODEL", "qwen-plus"))
    parser.add_argument(
        "--endpoint",
        default=os.getenv(
            "ROBOT_AGENT_API_ENDPOINT",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ),
    )
    parser.add_argument(
        "--scene-file",
        help="Optional semantic workspace JSON. Defaults to the demo kitchen workspace.",
    )
    parser.add_argument(
        "--skills-file",
        help=(
            "Optional JSON skill catalog. Use this to expose newly trained policy "
            "skills without changing the Planner prompt."
        ),
    )
    parser.add_argument(
        "--skip-plan-review",
        action="store_true",
        help=(
            "Skip proactive Qwen review; a validator-triggered repair call may still occur."
        ),
    )
    parser.add_argument("--pick-failures", type=int, default=1)
    parser.add_argument("--state-dir", help="Optional TaskMemory snapshot directory.")
    parser.add_argument(
        "--task-id",
        help="Optional stable task ID. If omitted, a unique ID is generated.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a persisted task; requires --state-dir and --task-id.",
    )
    parser.add_argument("--json", action="store_true", help="Print complete TaskMemory JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("=== Qwen Robot Agent + Simulated Robot ===")
    print(f"Planner: {args.planner}")
    print("Robot: DryRunRobotController (no physical commands)")
    task = (args.task or "").strip()
    if not task:
        task = input("\n请输入机械臂任务: ").strip()
    if not task:
        print("任务不能为空。")
        return 2
    if args.resume and (not args.state_dir or not args.task_id):
        print("--resume 需要同时提供 --state-dir 和 --task-id。")
        return 2

    store = JsonTaskStore(args.state_dir) if args.state_dir else None
    if args.planner == "scripted":
        return _run_scripted(args, task, store)
    return _run_qwen(args, task, store)


def _run_qwen(args: argparse.Namespace, task: str, store: JsonTaskStore | None) -> int:
    # Resolve local, deterministic inputs before asking for a credential.  An
    # invalid explicit scene/skill file should fail without prompting the user.
    try:
        capabilities = _load_capabilities(args.skills_file)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"\n技能目录不可用: {error}")
        return 2
    try:
        scene = _load_scene(args.scene_file, task)
    except (OSError, ValueError, json.JSONDecodeError, SceneSelectionError) as error:
        print(f"\n场景不可用: {error}")
        return 2

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        print("\n未检测到 QWEN_API_KEY / DASHSCOPE_API_KEY。")
        api_key = getpass.getpass("请输入千问 API Key（输入不会显示，也不会保存）: ").strip()
        if not api_key:
            print("API Key 不能为空。")
            return 2

    allowed_skills = tuple(
        item["action_type"] for item in capabilities.planner_skills()
    )
    try:
        planner = QwenApiPlanner(
            model=args.model,
            endpoint=args.endpoint,
            api_key=api_key,
            allowed_skills=allowed_skills,
            review_plans=not args.skip_plan_review,
        )
    except PlannerServiceError as error:
        print(f"Planner 初始化失败: {error}")
        return 2

    scene_metadata = dict(scene.get("scene") or {})
    print(f"\nSemantic workspace: {scene_metadata.get('scene_id', 'custom')}")
    entity_ids = [str(value) for value in scene.get("objects", {})]
    print(
        "Available entities: "
        + (
            ", ".join(entity_ids)
            if entity_ids
            else "(Qwen targets will be registered as virtual entities)"
        )
    )
    print("Planner output: high-level skills only (no XYZ/joint/action values)")
    print(
        "Skill whitelist: "
        + ", ".join(item["action_type"] for item in capabilities.planner_skills())
    )
    print(
        "Local routing: "
        + ", ".join(
            f"{item['action_type']}->{item['executor']}"
            for item in capabilities.routing_table()
        )
    )
    print(
        "Trusted semantic contracts: "
        f"{len(scene_metadata.get('skill_contracts') or [])}"
    )
    print(
        "Qwen plan review: "
        + ("disabled" if args.skip_plan_review else "enabled (draft + independent review)")
    )
    semantic_scene = _planner_scene(scene)
    skill_planner = SceneSkillPlanner(
        planner,
        semantic_scene,
        allow_virtual_entities=_allows_virtual_entities(scene),
    )
    controller = DryRunRobotController()
    runtime = build_agent_runtime(
        skill_planner,
        controller=controller,
        policies=PolicyRegistry(),
        task_store=store,
        dry_run=True,
        capabilities=capabilities,
        initial_world_state=semantic_scene,
        task_verifier=SymbolicGoalTaskVerifier(),
    )
    result = runtime.agent.run_safe(
        task,
        task_id=args.task_id,
        resume=args.resume,
    )
    return _print_result(
        result,
        task,
        planner_name=f"Qwen API ({args.model})",
        controller_state=controller.get_state(),
        show_json=args.json,
        extra={"qwen_api_calls": planner.request_count},
    )


def _run_scripted(
    args: argparse.Namespace,
    task: str,
    store: JsonTaskStore | None,
) -> int:
    simulation = build_pick_and_place_simulation(
        pick_failures=args.pick_failures,
    )
    runtime = build_agent_runtime(
        simulation.planner,
        controller=simulation.controller,
        policies=simulation.policies,
        task_store=store,
        dry_run=False,
    )
    result = runtime.agent.run_safe(
        task,
        task_id=args.task_id,
        resume=args.resume,
    )
    return _print_result(
        result,
        task,
        planner_name="scripted offline test planner",
        controller_state=simulation.controller.get_state(),
        show_json=args.json,
        extra={
            "pick_attempts": simulation.pick_backend.calls,
            "replans": result.memory.replan_count,
        },
    )


def _load_scene(scene_file: str | None, task: str) -> dict[str, Any]:
    if not scene_file:
        try:
            scene = select_demo_scene(task)
            metadata = scene.setdefault("scene", {})
            if isinstance(metadata, dict):
                metadata["simulation_allow_virtual_entities"] = True
            return scene
        except SceneSelectionError:
            return _open_world_simulation_scene()
    path = Path(scene_file)
    with path.open("r", encoding="utf-8") as stream:
        scene = json.load(stream)
    if not isinstance(scene, dict):
        raise ValueError("Scene JSON must be an object.")
    metadata = dict(scene.get("scene") or {})
    if not isinstance(scene.get("objects"), dict):
        raise ValueError("Scene JSON requires an objects map.")
    return scene


def _open_world_simulation_scene() -> dict[str, Any]:
    """Bootstrap a semantic-only scene for unknown dry-run tasks.

    Qwen may name concrete entities from the user request.  The local
    SceneSkillPlanner registers those names as virtual objects, but it still
    enforces the capability whitelist and never invents poses or skills.
    """

    return {
        "scene": {
            "scene_id": "open_world_simulation",
            "mode": "open_world_simulation",
            "open_world_simulation": True,
            "simulation_allow_virtual_entities": True,
            "skill_contracts": [],
            "goal_predicates": [],
        },
        "runtime": {
            "holding": None,
            "gripper": "unknown",
            "robot_location": "home",
        },
        "objects": {},
    }


def _is_open_world_simulation(scene: dict[str, Any]) -> bool:
    metadata = dict(scene.get("scene") or {})
    return bool(
        metadata.get("open_world_simulation")
        or metadata.get("mode") == "open_world_simulation"
    )


def _allows_virtual_entities(scene: dict[str, Any]) -> bool:
    metadata = dict(scene.get("scene") or {})
    return bool(
        _is_open_world_simulation(scene)
        or metadata.get("simulation_allow_virtual_entities")
    )


def _load_capabilities(skills_file: str | None) -> CapabilityRegistry:
    if skills_file:
        with Path(skills_file).open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        raw_capabilities = (
            document.get("capabilities")
            if isinstance(document, dict)
            else document
        )
        if not isinstance(raw_capabilities, list) or not all(
            isinstance(item, dict) for item in raw_capabilities
        ):
            raise ValueError(
                "Skill JSON must be a list or an object containing capabilities[]."
            )
        return CapabilityRegistry.from_dicts(raw_capabilities)

    high_level_action_types = {
        "pick",
        "place",
        "manipulate",
        "inspect",
        "move_home",
        "move_relative",
        "open_gripper",
        "close_gripper",
    }
    return CapabilityRegistry(
        item
        for item in default_capabilities()
        if item.action_type in high_level_action_types
    )


def _planner_scene(scene: dict[str, Any]) -> dict[str, Any]:
    """Expose semantic state to Qwen without calibrated motion parameters."""

    metadata = dict(scene.get("scene") or {})
    open_world_simulation = _is_open_world_simulation(scene)
    allow_virtual_entities = _allows_virtual_entities(scene)
    safe_metadata = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "affordances",
            "motions",
            "coordinate_frame",
            "units",
            "planner_notes",
        }
    }
    if open_world_simulation:
        safe_metadata["planner_notes"] = [
            "This is OPEN-WORLD SIMULATION; an empty objects map is valid.",
            (
                "Extract only concrete entity names explicitly present in the "
                "original user task. Local code will register them as virtual entities."
            ),
            (
                "Use high-level skills only. Do not invent coordinates, poses, "
                "sizes, affordances, skill contracts, executors, or policy IDs."
            ),
            (
                "Set goal.conditions to [] because virtual entity IDs are assigned "
                "locally after planning."
            ),
        ]
    elif allow_virtual_entities:
        safe_metadata["planner_notes"] = [
            "Plan semantic skills for the current task; do not emit motion coordinates.",
            "A closed container must be opened before picking an object inside it.",
            (
                "Use declared objects when they match. In SIMULATION only, a new "
                "concrete target explicitly named by the user may be emitted and "
                "will be registered locally as a virtual entity."
            ),
            (
                "If the plan introduces any new entity, set goal.conditions to []; "
                "do not invent a goal path, pose, property, affordance, or skill contract."
            ),
        ]
    else:
        safe_metadata["planner_notes"] = [
            "Plan semantic skills for the current task; do not emit motion coordinates.",
            "A closed container must be opened before picking an object inside it.",
            "Use only entities present in objects and skills present in capabilities.",
        ]
    safe_metadata["goal_predicates"] = [
        item
        for item in list(metadata.get("goal_predicates") or [])
        if isinstance(item, dict)
        and item.get("path") != "runtime.current_affordance_id"
    ]
    if not open_world_simulation:
        safe_metadata["goal_predicates"].append(
            {"path": "runtime.robot_location", "allowed_values": ["home"]}
        )
    semantic = {
        "scene": safe_metadata,
        "runtime": {
            key: value
            for key, value in dict(scene.get("runtime") or {}).items()
            if key
            not in {
                "cartesian_position_xyz_m",
                "current_affordance_id",
            }
        },
        "objects": dict(scene.get("objects") or {}),
    }
    semantic["runtime"].setdefault(
        "robot_location",
        (
            "home"
            if dict(scene.get("runtime") or {}).get("current_affordance_id")
            == "home"
            else "unknown"
        ),
    )
    return _strip_motion_values(semantic)


def _strip_motion_values(value: Any) -> Any:
    if isinstance(value, dict):
        blocked_fragments = {
            "xyz",
            "pose",
            "joint",
            "quaternion",
            "orientation",
            "velocity",
            "speed",
            "delta",
        }
        return {
            key: _strip_motion_values(item)
            for key, item in value.items()
            if not any(fragment in str(key).casefold() for fragment in blocked_fragments)
        }
    if isinstance(value, list):
        return [_strip_motion_values(item) for item in value]
    return value


def _print_result(
    result,
    task: str,
    *,
    planner_name: str,
    controller_state: dict[str, Any],
    show_json: bool,
    extra: dict[str, Any] | None = None,
) -> int:
    print("\n=== Execution Result ===")
    print(f"Task: {task}")
    print(f"Planner: {planner_name}")
    print(f"Task ID: {result.memory.task_id}")
    print(f"Status: {result.status}")
    print("Execution mode: SIMULATION")
    print("Physical verification: NOT AVAILABLE")
    world_values = dict(result.memory.world_state.get("values") or {})
    task_goal = dict((world_values.get("_agent") or {}).get("task_goal") or {})
    if task_goal:
        print(
            "Declared symbolic goal: "
            + json.dumps(task_goal, ensure_ascii=False)
        )
    for key, value in (extra or {}).items():
        print(f"{key}: {value}")

    print("\nCompleted high-level skill steps:")
    for step in result.memory.completed_steps:
        parameters = json.dumps(step.get("parameters", {}), ensure_ascii=False)
        print(
            f"  step={step['step_id']} action={step['action_type']} "
            f"target={step['target']} executor={step['executor']} "
            f"parameters={parameters}"
        )

    print("\nEvent timeline:")
    for event in result.memory.events:
        print(f"  {event['type']}: {event['data']}")

    print("\nFinal simulated controller state:")
    print(json.dumps(controller_state, ensure_ascii=False, indent=2))
    print("\nFinal symbolic task state:")
    print(
        json.dumps(
            {
                "version": result.memory.world_state.get("version", 0),
                "runtime": world_values.get("runtime", {}),
                "objects": world_values.get("objects", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if result.reason:
        print(f"\nFailure reason: {result.reason}")
    if show_json:
        print("\nComplete TaskMemory:")
        print(json.dumps(result.memory.snapshot(), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
