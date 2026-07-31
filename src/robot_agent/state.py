from __future__ import annotations

from typing import Any, Literal, TypedDict


class PlanStep(TypedDict, total=False):
    step_id: int
    action_type: str
    target: str
    executor: Literal["robot", "policy"]
    expected_result: str
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    parameters: dict[str, Any]
    policy_id: str
    instance_id: str
    timeout_seconds: float
    max_attempts: int
    depends_on: list[int]
    conditions: list[dict[str, Any]]
    effects: list[dict[str, Any]]
    on_condition_false: Literal["skip", "fail"]


class VerificationResult(TypedDict, total=False):
    success: bool
    error_type: str
    confidence: float
    verification_scope: Literal["command", "physical"]
    details: dict[str, Any]


class StepMemory(TypedDict, total=False):
    step: PlanStep
    status: Literal["completed", "failed"]
    robot_state: dict[str, Any]
    observation: dict[str, Any]
    verification: VerificationResult


class Feedback(TypedDict):
    success: bool
    reason: str
    details: dict[str, Any]


class RobotEvent(TypedDict):
    type: str
    message: str
    step_id: int | None
    data: dict[str, Any]


class RobotState(TypedDict, total=False):
    user_task: str
    plan: list[PlanStep]
    current_step_index: int
    current_step: PlanStep
    retry_count: int
    max_retries: int
    replan_count: int
    max_replans: int
    pi05_task_text: str
    execution_result: dict[str, Any]
    feedback: Feedback
    recovery_action: Literal["retry", "replan", "stop"]
    status: Literal[
        "planning",
        "running",
        "completed",
        "needs_agent_replan",
        "blocked_by_safety",
        "failed",
    ]
    history: list[str]
    events: list[RobotEvent]
