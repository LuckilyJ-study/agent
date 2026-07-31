from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .domain import WorldState
from .state import PlanStep


class SchedulingError(RuntimeError):
    pass


@dataclass
class TaskGraphScheduler:
    """Dependency-aware scheduler with simple conditional-step support."""

    plan: list[PlanStep]
    completed: set[int] = field(default_factory=set)
    skipped: set[int] = field(default_factory=set)

    def next_step(self, world: WorldState) -> PlanStep | None:
        while True:
            pending = [
                step
                for step in self.plan
                if int(step["step_id"]) not in self.completed | self.skipped
            ]
            if not pending:
                return None
            progressed = False
            for step in pending:
                dependencies = set(step.get("depends_on", []))
                if not dependencies.issubset(self.completed | self.skipped):
                    continue
                conditions = step.get("conditions", [])
                if conditions and not world.conditions_met(conditions):
                    if step.get("on_condition_false", "fail") == "skip":
                        step["status"] = "skipped"
                        self.skipped.add(int(step["step_id"]))
                        progressed = True
                        break
                    raise SchedulingError(
                        f"Conditions are not satisfied for step {step['step_id']}."
                    )
                return step
            if progressed:
                continue
            blocked = [int(step["step_id"]) for step in pending]
            raise SchedulingError(f"No runnable step; blocked steps: {blocked}.")

    def complete(self, step: PlanStep, world: WorldState) -> None:
        step["status"] = "completed"
        self.completed.add(int(step["step_id"]))
        world.apply_effects(step.get("effects", []))

    def replace_unfinished(self, new_plan: list[PlanStep]) -> None:
        self.plan = new_plan
        self.completed.clear()
        self.skipped.clear()

    @property
    def status(self) -> Literal["running", "completed"]:
        remaining = [
            step for step in self.plan
            if int(step["step_id"]) not in self.completed | self.skipped
        ]
        return "running" if remaining else "completed"
