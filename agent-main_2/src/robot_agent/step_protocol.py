from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .capabilities import default_capabilities
from .state import PlanStep


STEP_STATUSES = {"pending", "running", "completed", "failed", "skipped"}
EXECUTORS = {"robot", "policy"}
ROBOT_ACTIONS = {
    item.action_type for item in default_capabilities() if item.executor == "robot"
}
POLICY_ACTIONS = {
    item.action_type for item in default_capabilities() if item.executor == "policy"
}
KNOWN_ACTIONS = ROBOT_ACTIONS | POLICY_ACTIONS

PLAN_STEP_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "step_id": {"type": "integer", "minimum": 1},
        "action_type": {"type": "string", "enum": sorted(KNOWN_ACTIONS)},
        "target": {"type": "string", "minLength": 1},
        # ``executor`` is accepted for backward compatibility, but it is not
        # required from an LLM.  The trusted local CapabilityRegistry/Router is
        # authoritative for execution routing.
        "executor": {"type": "string", "enum": sorted(EXECUTORS)},
        "expected_result": {"type": "string", "minLength": 1},
        "status": {"type": "string", "enum": sorted(STEP_STATUSES)},
        "parameters": {"type": "object"},
        "policy_id": {"type": "string"},
        "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
        "max_attempts": {"type": "integer", "minimum": 1},
        "depends_on": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "conditions": {"type": "array", "items": {"type": "object"}},
        "effects": {"type": "array", "items": {"type": "object"}},
        "on_condition_false": {"type": "string", "enum": ["skip", "fail"]},
    },
    "required": [
        "step_id", "action_type", "target", "expected_result", "status", "parameters"
    ],
    "additionalProperties": False,
}

GOAL_CONDITION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "operator": {"type": "string", "enum": ["eq"]},
        "value": {},
    },
    "required": ["path", "operator", "value"],
    "additionalProperties": False,
}

TASK_GOAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 1},
        "conditions": {
            "type": "array",
            "items": GOAL_CONDITION_JSON_SCHEMA,
        },
    },
    "required": ["description", "conditions"],
    "additionalProperties": False,
}

PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": TASK_GOAL_JSON_SCHEMA,
        "steps": {"type": "array", "items": PLAN_STEP_JSON_SCHEMA},
    },
    "required": ["goal", "steps"],
    "additionalProperties": False,
}


class StepProtocolError(ValueError):
    pass


def build_plan_json_schema(allowed_skills: Iterable[str]) -> dict[str, Any]:
    """Build the LLM response schema from the runtime skill whitelist.

    The old module-level schema is retained for compatibility, while every
    live planner can now receive the exact skills registered by the current
    robot instead of a stale global enum.
    """

    skills = sorted({str(value).strip() for value in allowed_skills if str(value).strip()})
    if not skills:
        raise StepProtocolError("At least one allowed skill is required.")
    schema = deepcopy(PLAN_JSON_SCHEMA)
    step_schema = schema["properties"]["steps"]["items"]
    step_schema["properties"]["action_type"]["enum"] = skills
    # A model must not select an implementation backend or a policy model.
    # These internal fields remain accepted by normalize_step for trusted,
    # hand-authored plans, but are intentionally absent from the LLM schema.
    step_schema["properties"].pop("executor", None)
    step_schema["properties"].pop("policy_id", None)
    return schema


def executor_for(action_type: str) -> str:
    if action_type in ROBOT_ACTIONS:
        return "robot"
    if action_type in POLICY_ACTIONS:
        return "policy"
    raise StepProtocolError(f"Unknown action_type '{action_type}'.")


def normalize_step(
    raw: dict[str, Any],
    default_step_id: int,
    *,
    allowed_skills: Iterable[str] | None = None,
) -> PlanStep:
    """Convert canonical or legacy id/skill input into the sole internal protocol."""
    action_type = raw.get("action_type", raw.get("skill"))
    if not isinstance(action_type, str) or not action_type.strip():
        raise StepProtocolError("Step requires a non-empty action_type.")
    action_type = action_type.strip()
    allowed = (
        {str(value).strip() for value in allowed_skills}
        if allowed_skills is not None
        else None
    )
    if allowed is not None and action_type not in allowed:
        raise StepProtocolError(f"Skill '{action_type}' is not in the runtime whitelist.")
    expected_executor = (
        executor_for(action_type) if action_type in KNOWN_ACTIONS else None
    )
    supplied_executor = raw.get("executor")
    if (
        expected_executor is not None
        and supplied_executor is not None
        and supplied_executor != expected_executor
    ):
        raise StepProtocolError(
            f"Action '{action_type}' must use executor='{expected_executor}', got '{supplied_executor}'."
        )
    try:
        step_id = int(raw.get("step_id", raw.get("id", default_step_id)))
    except (TypeError, ValueError) as error:
        raise StepProtocolError("step_id must be an integer.") from error
    target = raw.get("target")
    expected_result = raw.get("expected_result")
    status = raw.get("status", "pending")
    parameters = raw.get("parameters", {})
    if step_id < 1:
        raise StepProtocolError("step_id must be at least 1.")
    if not isinstance(target, str) or not target.strip():
        raise StepProtocolError("target must be a non-empty string.")
    if not isinstance(expected_result, str) or not expected_result.strip():
        raise StepProtocolError("expected_result must be a non-empty string.")
    if status not in STEP_STATUSES:
        raise StepProtocolError(f"Invalid step status '{status}'.")
    if not isinstance(parameters, dict):
        raise StepProtocolError("parameters must be an object.")

    step: PlanStep = {
        "step_id": step_id,
        "action_type": action_type,
        "target": target.strip(),
        "expected_result": expected_result.strip(),
        "status": status,
        "parameters": dict(parameters),
    }
    if expected_executor is not None:
        step["executor"] = expected_executor
    elif supplied_executor in EXECUTORS:
        # Custom capabilities are resolved again by CapabilityRegistry.  Keep a
        # supplied value only so the registry can reject a forged mismatch.
        step["executor"] = supplied_executor
    for optional_key in (
        "policy_id",
        "timeout_seconds",
        "max_attempts",
        "depends_on",
        "conditions",
        "effects",
        "on_condition_false",
    ):
        if optional_key in raw:
            step[optional_key] = raw[optional_key]
    if expected_executor == "policy" and "policy_id" in raw and not str(raw["policy_id"]).strip():
        raise StepProtocolError("policy_id cannot be empty when provided.")
    return step


def normalize_plan(
    raw_steps: Iterable[dict[str, Any]],
    *,
    allowed_skills: Iterable[str] | None = None,
) -> list[PlanStep]:
    steps = [
        normalize_step(raw, index, allowed_skills=allowed_skills)
        for index, raw in enumerate(raw_steps, start=1)
    ]
    seen: set[int] = set()
    for step in steps:
        step_id = int(step["step_id"])
        if step_id in seen:
            raise StepProtocolError(f"Duplicate step_id: {step_id}.")
        seen.add(step_id)
    return steps
