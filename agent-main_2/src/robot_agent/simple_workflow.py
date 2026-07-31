from __future__ import annotations

from .executor import execute_pi05
from .feedback import evaluate_feedback
from .gateway import LeRobotGateway, RobotGateway, SimulatedGateway
from .pi05_adapter import build_pi05_input
from .planner import Planner, plan_task, replan_task
from .recovery import decide_recovery
from .safety import ExecutionPolicy
from .scheduler import advance_step, choose_current_step, prepare_retry
from .skills import SkillRegistry, validate_plan
from .state import RobotState


def run_simple_workflow(
    initial_state: RobotState,
    planner: Planner,
    registry: SkillRegistry | None = None,
    gateway: RobotGateway | None = None,
    policy: ExecutionPolicy | None = None,
) -> RobotState:
    """Run the Agent workflow without requiring the optional LangGraph package."""
    state = dict(initial_state)
    selected_gateway = gateway or LeRobotGateway()
    selected_policy = policy or ExecutionPolicy(max_retries=state.get("max_retries", 2))

    state.update(plan_task(state, planner))
    state.update(validate_plan(state, registry))
    if state["status"] != "running":
        return state

    while True:
        state.update(choose_current_step(state))
        if state["status"] == "completed":
            return state

        state.update(build_pi05_input(state))
        state.update(execute_pi05(state, selected_gateway, selected_policy))
        if state["status"] != "running":
            return state

        state.update(evaluate_feedback(state))
        if state["feedback"]["success"]:
            state.update(advance_step(state))
            continue

        state.update(decide_recovery(state))
        if state["recovery_action"] == "retry":
            state.update(prepare_retry(state))
            continue

        state.update(replan_task(state, planner))
        if state["status"] != "planning":
            return state
        state.update(validate_plan(state, registry))
        if state["status"] != "running":
            return state
