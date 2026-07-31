from __future__ import annotations

from .events import record_event
from .motion import parse_relative_motion_target
from .state import PlanStep, RobotState


class SkillValidationError(ValueError):
    """Raised when a planner tries to call an unavailable robot skill."""


class SkillRegistry:
    """OpenCode-style tool registry for the robot's allowed skill vocabulary."""

    def __init__(self, allowed_skills: set[str] | None = None) -> None:
        self.allowed_skills = allowed_skills or {
            "pick",
            "place",
            "move_home",
            "inspect",
            "move_relative",
            "open_gripper",
            "close_gripper",
        }

    def validate_step(self, step: PlanStep) -> None:
        required_keys = {"id", "skill", "target", "expected_result"}
        missing_keys = required_keys.difference(step)
        if missing_keys:
            raise SkillValidationError(f"Plan step is missing keys: {sorted(missing_keys)}")
        if step["skill"] not in self.allowed_skills:
            raise SkillValidationError(f"Skill '{step['skill']}' is not registered.")
        if not step["target"].strip():
            raise SkillValidationError("Skill target cannot be empty.")
        if step["skill"] == "move_relative":
            try:
                parse_relative_motion_target(step["target"])
            except ValueError as error:
                raise SkillValidationError(str(error)) from error
        if step["skill"] in {"open_gripper", "close_gripper"} and step["target"] != "gripper":
            raise SkillValidationError(f"{step['skill']} target must be 'gripper'.")

    def validate_plan(self, plan: list[PlanStep]) -> None:
        if not plan:
            raise SkillValidationError("Planner returned an empty plan.")
        seen_ids: set[int] = set()
        for step in plan:
            self.validate_step(step)
            if step["id"] in seen_ids:
                raise SkillValidationError(f"Duplicate plan step id: {step['id']}")
            seen_ids.add(step["id"])


def validate_plan(state: RobotState, registry: SkillRegistry | None = None) -> dict:
    """Reject invalid LLM output before it can reach Pi05 or robot hardware."""
    selected_registry = registry or SkillRegistry()
    try:
        selected_registry.validate_plan(state["plan"])
    except SkillValidationError as error:
        update: dict = {"status": "blocked_by_safety"}
        update.update(record_event(state, "plan.rejected", str(error)))
        return update

    update = {"status": "running"}
    update.update(record_event(state, "plan.validated", "Plan uses only registered skills."))
    return update
