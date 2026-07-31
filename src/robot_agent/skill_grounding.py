from __future__ import annotations

from copy import deepcopy
from typing import Any

from .domain import Condition, WorldState
from .state import PlanStep
from .virtual_entities import (
    VirtualEntityRegistrationError,
    VirtualEntityRegistry,
)


class SkillPlanGroundingError(ValueError):
    """A high-level plan violates trusted scene or skill contracts."""


class SceneSkillPlanner:
    """Validate high-level LLM plans using reusable semantic skill contracts.

    The wrapper never stores a complete task recipe.  It simulates whichever
    skill sequence Qwen produced, derives trusted conditions/effects, and asks
    Qwen for one repair when the sequence cannot execute from the current state.
    """

    DIRECT_ACTIONS = {
        "move_home",
        "move_relative",
        "move_to",
        "move_to_pose",
        "move_linear",
        "open_gripper",
        "close_gripper",
    }

    def __init__(
        self,
        planner: Any,
        scene: dict[str, Any],
        *,
        allow_one_repair: bool = True,
        allow_virtual_entities: bool = False,
        max_virtual_entities_per_plan: int = 32,
    ) -> None:
        self.planner = planner
        self.scene = deepcopy(scene)
        self.allow_one_repair = allow_one_repair
        self.allow_virtual_entities = allow_virtual_entities
        self.max_virtual_entities_per_plan = max_virtual_entities_per_plan
        self.current_goal: dict[str, Any] | None = None
        self._pending_world_patch: dict[str, Any] = {}

    def consume_world_patch(self) -> dict[str, Any]:
        """Return and clear locally trusted virtual-entity registrations.

        The patch never contains effects produced while validating the future
        plan.  A runtime integration may merge it into the real current
        ``WorldState`` before executing the first action.
        """

        patch = deepcopy(self._pending_world_patch)
        self._pending_world_patch.clear()
        return patch

    def drain_world_patch(self) -> dict[str, Any]:
        """Alias for integrations that use drain-style queue terminology."""

        return self.consume_world_patch()

    def create_plan(self, user_task: str) -> list[PlanStep]:
        return self.create_plan_with_context(
            {
                "original_task": user_task,
                "world_state": {"version": 0, "values": deepcopy(self.scene)},
                "capabilities": [],
            }
        )

    def create_plan_with_context(self, context: dict[str, Any]) -> list[PlanStep]:
        enriched = self._with_scene(context)
        plan = self.planner.create_plan_with_context(enriched)
        return self._validate_with_repair(plan, enriched, suffix_only=False)

    def revise_from_failure(self, context: dict[str, Any]) -> list[PlanStep]:
        enriched = self._with_scene(context)
        plan = self.planner.revise_from_failure(enriched)
        return self._validate_with_repair(plan, enriched, suffix_only=True)

    def _with_scene(self, context: dict[str, Any]) -> dict[str, Any]:
        enriched = deepcopy(context)
        snapshot = dict(enriched.get("world_state") or {})
        if not isinstance(snapshot.get("values"), dict):
            snapshot = {"version": 0, "values": deepcopy(self.scene)}
        enriched["world_state"] = snapshot
        return enriched

    def _validate_with_repair(
        self,
        plan: list[PlanStep],
        context: dict[str, Any],
        *,
        suffix_only: bool,
    ) -> list[PlanStep]:
        try:
            grounded, world_patch = self._validate(plan, context)
        except SkillPlanGroundingError as error:
            repair = getattr(self.planner, "repair_rejected_plan", None)
            if not self.allow_one_repair or not callable(repair):
                raise
            repaired = repair(
                context,
                plan,
                str(error),
                suffix_only=suffix_only,
            )
            grounded, world_patch = self._validate(repaired, context)
        self._commit_world_patch(world_patch)
        self.current_goal = self._planner_goal()
        return grounded

    def _validate(
        self,
        plan: list[PlanStep],
        context: dict[str, Any],
    ) -> tuple[list[PlanStep], dict[str, Any]]:
        values = deepcopy(
            dict(context.get("world_state") or {}).get("values") or self.scene
        )
        objects = dict(values.get("objects") or {})
        if self.allow_virtual_entities:
            for entity_id, raw_entity in dict(
                self.scene.get("objects") or {}
            ).items():
                entity = dict(raw_entity or {})
                if entity.get("virtual") is True:
                    objects.setdefault(str(entity_id), deepcopy(entity))
            values["objects"] = objects

        world_patch: dict[str, Any] = {}
        if self.allow_virtual_entities:
            registry = VirtualEntityRegistry(
                objects,
                max_new_entities=self.max_virtual_entities_per_plan,
            )
            try:
                for raw_step in plan:
                    action_type = str(
                        raw_step.get("action_type") or raw_step.get("skill") or ""
                    )
                    target = str(raw_step.get("target") or "").strip()
                    registry.resolve_or_register(action_type, target)
            except VirtualEntityRegistrationError as error:
                raise SkillPlanGroundingError(str(error)) from error
            world_patch = registry.consume_patch()
            objects.update(deepcopy(world_patch.get("objects") or {}))
            values["objects"] = objects

        world = WorldState(values=values)
        aliases = _entity_aliases(objects)
        metadata = dict(world.get("scene") or {})
        contracts = [
            dict(item)
            for item in metadata.get("skill_contracts", [])
            if isinstance(item, dict)
        ]
        grounded: list[PlanStep] = []

        for raw_step in plan:
            step: PlanStep = deepcopy(raw_step)
            action_type = str(step.get("action_type") or step.get("skill") or "")
            target = str(step.get("target") or "").strip()
            parameters = dict(step.get("parameters") or {})
            _reject_untrusted_motion_parameters(parameters)
            resolved_target = _resolve_target(target, aliases)
            if resolved_target is not None:
                step["target"] = resolved_target
                entity = dict(objects.get(resolved_target) or {})
                trusted_aliases = _deduplicate_strings(
                    [
                        resolved_target,
                        target,
                        entity.get("display_name"),
                        *list(entity.get("aliases") or []),
                    ]
                )
                parameters["target_aliases"] = trusted_aliases
                parameters["monitor_targets"] = trusted_aliases
                step["parameters"] = parameters
                step["_trusted_target_aliases"] = trusted_aliases

            contract = _matching_contract(
                contracts,
                action_type,
                resolved_target or target,
                parameters,
            )
            if contract is not None:
                conditions = deepcopy(contract.get("preconditions") or [])
                effects = deepcopy(contract.get("effects") or [])
            else:
                conditions, effects = self._derive_generic_contract(
                    action_type,
                    resolved_target,
                    target,
                    parameters,
                    objects,
                    world,
                )

            if not world.conditions_met(conditions):
                failed = [
                    condition
                    for condition in conditions
                    if not world.evaluate(Condition.from_dict(condition))
                ]
                raise SkillPlanGroundingError(
                    f"Skill {action_type}({target}) has unsatisfied preconditions: "
                    f"{failed}."
                )
            # Never trust Planner-supplied symbolic effects.
            step["conditions"] = conditions
            step["effects"] = effects
            grounded.append(step)
            world.apply_effects(effects)

        goal = self._planner_goal()
        if (
            world_patch.get("objects")
            and isinstance(goal, dict)
            and list(goal.get("conditions") or [])
        ):
            raise SkillPlanGroundingError(
                "A simulation plan that introduces virtual entities must use "
                "goal.conditions=[]; local virtual IDs are assigned after planning."
            )
        self._validate_goal(goal, metadata, world)
        return grounded, world_patch

    def _commit_world_patch(self, patch: dict[str, Any]) -> None:
        new_objects = dict(patch.get("objects") or {})
        if not new_objects:
            return
        scene_objects = self.scene.setdefault("objects", {})
        if not isinstance(scene_objects, dict):
            scene_objects = {}
            self.scene["objects"] = scene_objects
        pending_objects = self._pending_world_patch.setdefault("objects", {})
        for entity_id, raw_entity in new_objects.items():
            entity = deepcopy(raw_entity)
            scene_objects.setdefault(str(entity_id), entity)
            pending_objects.setdefault(str(entity_id), entity)

    def _derive_generic_contract(
        self,
        action_type: str,
        resolved_target: str | None,
        raw_target: str,
        parameters: dict[str, Any],
        objects: dict[str, Any],
        world: WorldState,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if action_type in self.DIRECT_ACTIONS:
            if action_type == "move_home":
                return [], [
                    {
                        "path": "runtime.robot_location",
                        "operation": "set",
                        "value": "home",
                    }
                ]
            if action_type == "open_gripper":
                return [], [
                    {
                        "path": "runtime.gripper",
                        "operation": "set",
                        "value": "open",
                    }
                ]
            if action_type == "close_gripper":
                return [], [
                    {
                        "path": "runtime.gripper",
                        "operation": "set",
                        "value": "closed",
                    }
                ]
            return [], []

        if resolved_target is None:
            raise SkillPlanGroundingError(
                f"Target '{raw_target}' is not present in the current scene."
            )
        if action_type == "inspect":
            return [], []
        if action_type == "pick":
            entity = dict(objects.get(resolved_target) or {})
            conditions: list[dict[str, Any]] = [
                {
                    "path": "runtime.holding",
                    "operator": "eq",
                    "value": None,
                }
            ]
            container = entity.get("contained_in")
            if container:
                conditions.append(
                    {
                        "path": f"objects.{container}.state",
                        "operator": "eq",
                        "value": "open",
                    }
                )
            return conditions, [
                {
                    "path": "runtime.holding",
                    "operation": "set",
                    "value": resolved_target,
                },
                {
                    "path": f"objects.{resolved_target}.location",
                    "operation": "set",
                    "value": "held",
                },
            ]
        if action_type == "place":
            holding = world.get("runtime.holding")
            if not holding:
                raise SkillPlanGroundingError(
                    f"place({raw_target}) requires a currently held object."
                )
            return [
                {
                    "path": "runtime.holding",
                    "operator": "eq",
                    "value": holding,
                }
            ], [
                {
                    "path": "runtime.holding",
                    "operation": "set",
                    "value": None,
                },
                {
                    "path": f"objects.{holding}.location",
                    "operation": "set",
                    "value": resolved_target,
                },
            ]

        operation = str(parameters.get("operation") or "").strip()
        operation_hint = f" operation={operation!r}" if operation else ""
        raise SkillPlanGroundingError(
            f"Scene has no reusable contract for {action_type}({raw_target})"
            f"{operation_hint}. Add a skill contract instead of inventing motor data."
        )

    def _validate_goal(
        self,
        goal: dict[str, Any] | None,
        metadata: dict[str, Any],
        world: WorldState,
    ) -> None:
        if goal is None:
            return
        conditions = goal.get("conditions")
        if not isinstance(conditions, list):
            raise SkillPlanGroundingError("Planner goal requires conditions[].")
        allowed = {
            str(item.get("path")): list(item.get("allowed_values") or [])
            for item in metadata.get("goal_predicates", [])
            if isinstance(item, dict) and item.get("path")
        }
        for condition in conditions:
            parsed = Condition.from_dict(condition)
            if allowed:
                if parsed.path not in allowed:
                    raise SkillPlanGroundingError(
                        f"Goal path '{parsed.path}' is not allowed by this scene."
                    )
                if parsed.value not in allowed[parsed.path]:
                    raise SkillPlanGroundingError(
                        f"Goal value {parsed.value!r} is not allowed for "
                        f"'{parsed.path}'."
                    )
            if not world.evaluate(parsed):
                raise SkillPlanGroundingError(
                    f"Plan does not achieve goal condition {condition}."
                )

    def _planner_goal(self) -> dict[str, Any] | None:
        goal = getattr(self.planner, "current_goal", None)
        if not isinstance(goal, dict):
            goal = getattr(self.planner, "last_goal", None)
        return deepcopy(goal) if isinstance(goal, dict) else None


def _entity_aliases(objects: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for entity_id, raw in objects.items():
        values = [entity_id, *list(dict(raw or {}).get("aliases") or [])]
        for value in values:
            aliases[_canonical(str(value))] = str(entity_id)
    return aliases


def _resolve_target(target: str, aliases: dict[str, str]) -> str | None:
    return aliases.get(_canonical(target))


def _matching_contract(
    contracts: list[dict[str, Any]],
    action_type: str,
    target: str,
    parameters: dict[str, Any],
) -> dict[str, Any] | None:
    for contract in contracts:
        if str(contract.get("action_type") or "") != action_type:
            continue
        if _canonical(str(contract.get("target") or "")) != _canonical(target):
            continue
        expected = dict(contract.get("parameters_match") or {})
        if all(parameters.get(key) == value for key, value in expected.items()):
            return contract
    return None


def _canonical(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def _reject_untrusted_motion_parameters(parameters: dict[str, Any]) -> None:
    """Reject motor-space values supplied through a high-level LLM plan."""

    blocked_fragments = {
        "xyz",
        "pose",
        "joint",
        "quaternion",
        "orientation",
        "velocity",
        "speed",
        "motor",
        "action_chunk",
        "action chunk",
    }
    reserved_fields = {
        "perception_grounding",
        "target_aliases",
        "monitor_targets",
        "_trusted_target_aliases",
    }

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).casefold().replace("-", "_")
                if normalized in reserved_fields:
                    raise SkillPlanGroundingError(
                        "High-level Planner parameters cannot set trusted runtime "
                        f"field '{path + str(key)}'."
                    )
                if any(fragment in normalized for fragment in blocked_fragments):
                    raise SkillPlanGroundingError(
                        "High-level Planner parameters cannot contain motor-space "
                        f"field '{path + str(key)}'."
                    )
                walk(child, path + str(key) + ".")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}{index}.")

    walk(parameters, "parameters.")


def _deduplicate_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        canonical = _canonical(value)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        result.append(value)
    return result
