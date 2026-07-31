from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .executor import execute_pi05
from .feedback import evaluate_feedback
from .gateway import RobotGateway, SimulatedGateway
from .pi05_adapter import build_pi05_input
from .planner import Planner, plan_task, replan_task
from .recovery import decide_recovery
from .safety import ExecutionPolicy
from .scheduler import advance_step, choose_current_step, prepare_retry
from .skills import SkillRegistry, validate_plan
from .state import RobotState


def route_after_validation(state: RobotState) -> str:
    return "schedule" if state["status"] == "running" else "end"


def route_after_schedule(state: RobotState) -> str:
    return "end" if state["status"] == "completed" else "adapt"


def route_after_execution(state: RobotState) -> str:
    return "feedback" if state["status"] == "running" else "end"


def route_after_feedback(state: RobotState) -> str:
    return "advance" if state["feedback"]["success"] else "recover"


def route_after_recovery(state: RobotState) -> str:
    return "retry" if state["recovery_action"] == "retry" else "replan"


def route_after_agent_replan(state: RobotState) -> str:
    return "validate" if state["status"] == "planning" else "end"


def build_robot_graph(
    planner: Planner | None = None,
    registry: SkillRegistry | None = None,
    gateway: RobotGateway | None = None,
    policy: ExecutionPolicy | None = None,
):
    """Build a testable workflow with explicit dependency injection."""
    selected_gateway = gateway or SimulatedGateway()
    selected_policy = policy or ExecutionPolicy()

    graph = StateGraph(RobotState)
    graph.add_node("planner", lambda state: plan_task(state, planner))
    graph.add_node("validate_plan", lambda state: validate_plan(state, registry))
    graph.add_node("scheduler", choose_current_step)
    graph.add_node("adapt_pi05_input", build_pi05_input)
    graph.add_node("execute_pi05", lambda state: execute_pi05(state, selected_gateway, selected_policy))
    graph.add_node("evaluate_feedback", evaluate_feedback)
    graph.add_node("decide_recovery", decide_recovery)
    graph.add_node("advance_step", advance_step)
    graph.add_node("prepare_retry", prepare_retry)
    graph.add_node("agent_replan", lambda state: replan_task(state, planner))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "validate_plan")
    graph.add_conditional_edges("validate_plan", route_after_validation, {"schedule": "scheduler", "end": END})
    graph.add_conditional_edges("scheduler", route_after_schedule, {"adapt": "adapt_pi05_input", "end": END})
    graph.add_edge("adapt_pi05_input", "execute_pi05")
    graph.add_conditional_edges("execute_pi05", route_after_execution, {"feedback": "evaluate_feedback", "end": END})
    graph.add_conditional_edges("evaluate_feedback", route_after_feedback, {"advance": "advance_step", "recover": "decide_recovery"})
    graph.add_conditional_edges("decide_recovery", route_after_recovery, {"retry": "prepare_retry", "replan": "agent_replan"})
    graph.add_edge("advance_step", "scheduler")
    graph.add_edge("prepare_retry", "scheduler")
    graph.add_conditional_edges("agent_replan", route_after_agent_replan, {"validate": "validate_plan", "end": END})
    return graph.compile()
