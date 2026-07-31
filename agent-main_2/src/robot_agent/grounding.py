from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from .domain import Condition, WorldState
from .state import PlanStep
from .task_verifier import TaskVerification


class PlanGroundingError(ValueError):
    pass


DIRECT_SCENE_ACTIONS = {
    "move_to_pose",
    "move_linear",
    "open_gripper",
    "close_gripper",
}


class SceneGroundedPlanner:
    """Ground an LLM plan against reusable scene affordances and state contracts."""

    def __init__(
        self,
        planner: Any,
        scene: dict[str, Any],
        tolerance: float = 1e-6,
        max_plan_repairs: int = 1,
    ) -> None:
        self.planner = planner
        self.scene = deepcopy(scene)
        self.tolerance = tolerance
        self.max_plan_repairs = max(0, int(max_plan_repairs))
        self.current_goal: dict[str, Any] | None = None

    def create_plan(self, user_task: str) -> list[PlanStep]:
        plan = self.planner.create_plan(user_task)
        return self._validate_with_repair(
            plan,
            user_task=user_task,
            initial_values=self.scene,
            candidate_goal=self._planner_goal(),
            planning_context={
                "original_task": user_task,
                "world_state": {"version": 0, "values": self.scene},
            },
        )

    def create_plan_with_context(self, context: dict[str, Any]) -> list[PlanStep]:
        method = getattr(self.planner, "create_plan_with_context", None)
        plan = method(context) if callable(method) else self.planner.create_plan(
            str(context["original_task"])
        )
        return self._validate_with_repair(
            plan,
            user_task=str(context["original_task"]),
            initial_values=_world_values(context, self.scene),
            candidate_goal=self._planner_goal(),
            planning_context=context,
        )

    def revise_from_failure(self, context: dict[str, Any]) -> list[PlanStep]:
        plan = self.planner.revise_from_failure(context)
        return self._validate_with_repair(
            plan,
            user_task=str(context["original_task"]),
            initial_values=_world_values(context, self.scene),
            candidate_goal=self._planner_goal(),
            planning_context=context,
            suffix_only=True,
        )

    def _planner_goal(self) -> dict[str, Any] | None:
        goal = getattr(self.planner, "last_goal", None)
        return deepcopy(goal) if isinstance(goal, dict) else None

    def _validate_with_repair(
        self,
        plan: list[PlanStep],
        *,
        user_task: str,
        initial_values: dict[str, Any],
        candidate_goal: dict[str, Any] | None,
        planning_context: dict[str, Any],
        suffix_only: bool = False,
    ) -> list[PlanStep]:
        try:
            return self._validate(
                plan,
                user_task=user_task,
                initial_values=initial_values,
                candidate_goal=candidate_goal,
            )
        except PlanGroundingError as first_error:
            repair = getattr(self.planner, "repair_rejected_plan", None)
            if self.max_plan_repairs < 1 or not callable(repair):
                raise
            repaired = repair(
                planning_context,
                plan,
                str(first_error),
                suffix_only=suffix_only,
            )
            try:
                return self._validate(
                    repaired,
                    user_task=user_task,
                    initial_values=initial_values,
                    candidate_goal=self._planner_goal(),
                )
            except PlanGroundingError as repaired_error:
                raise PlanGroundingError(
                    "Planner repair was still invalid. First validation error: "
                    f"{first_error} Repair validation error: {repaired_error}"
                ) from repaired_error

    def _validate(
        self,
        plan: list[PlanStep],
        *,
        user_task: str,
        initial_values: dict[str, Any],
        candidate_goal: dict[str, Any] | None,
    ) -> list[PlanStep]:
        expected_goal = _stored_goal(initial_values)
        goal = _validate_goal(
            candidate_goal or expected_goal,
            self.scene,
        )
        if expected_goal is not None and candidate_goal is not None:
            expected = _validate_goal(expected_goal, self.scene)
            if _goal_signature(goal) != _goal_signature(expected):
                raise PlanGroundingError(
                    "Replanning changed the original task goal. The unfinished suffix "
                    "must preserve the persisted goal conditions."
                )
            goal = expected
        self.current_goal = deepcopy(goal) if goal is not None else None

        initial_world = WorldState(values=deepcopy(initial_values))
        if (
            plan
            and goal is not None
            and not _unsatisfied_goal_conditions(goal, initial_world)
        ):
            raise PlanGroundingError(
                "The declared task goal is already satisfied; the Planner must "
                "return an empty plan instead of unnecessary robot motion."
            )
        if not plan:
            if goal is None:
                raise PlanGroundingError(
                    "An empty plan requires a machine-verifiable task goal."
                )
            unsatisfied = _unsatisfied_goal_conditions(goal, initial_world)
            if unsatisfied:
                raise PlanGroundingError(
                    "Planner returned no actions although the task goal is not "
                    f"already satisfied: {unsatisfied}."
                )
            return []

        simulator = _SemanticPlanSimulator(
            self.scene,
            initial_values,
            tolerance=self.tolerance,
        )
        grounded_plan, final_world = simulator.validate(plan)
        _require_task_entities_addressed(
            user_task,
            grounded_plan,
            self.scene,
            simulator.used_entities,
        )
        if goal is not None:
            unsatisfied = _unsatisfied_goal_conditions(goal, final_world)
            if unsatisfied:
                raise PlanGroundingError(
                    "The plan is executable but does not achieve its declared task goal: "
                    f"{unsatisfied}."
                )
        return grounded_plan


@dataclass
class SceneGoalTaskVerifier:
    scene: dict[str, Any]

    def verify(
        self,
        original_task: str,
        completed_steps: list[PlanStep],
        world_state: WorldState,
    ) -> TaskVerification:
        stored = world_state.get("_agent.task_goal")
        try:
            goal = _validate_goal(
                stored if isinstance(stored, dict) else None,
                self.scene,
            )
        except PlanGroundingError as error:
            return TaskVerification(False, str(error), "symbolic_goal")
        if goal is None:
            return TaskVerification(
                False,
                "No machine-verifiable task goal was persisted.",
                "symbolic_goal",
            )
        unsatisfied = _unsatisfied_goal_conditions(goal, world_state)
        if unsatisfied:
            return TaskVerification(
                False,
                f"Final symbolic goal conditions are not satisfied: {unsatisfied}.",
                "symbolic_goal",
            )
        return TaskVerification(
            True,
            "Declared symbolic goal is satisfied by the current world state and "
            "grounded affordance contracts; the physical result remains unverified.",
            "symbolic_goal",
        )


class _SemanticPlanSimulator:
    def __init__(
        self,
        scene: dict[str, Any],
        initial_values: dict[str, Any],
        *,
        tolerance: float,
    ) -> None:
        metadata = dict(scene.get("scene") or {})
        self.affordances = {
            str(item["affordance_id"]): deepcopy(item)
            for item in metadata.get("affordances", [])
            if isinstance(item, dict) and item.get("affordance_id")
        }
        self.motions = {
            str(item["motion_id"]): deepcopy(item)
            for item in metadata.get("motions", [])
            if isinstance(item, dict) and item.get("motion_id")
        }
        if not self.affordances:
            raise PlanGroundingError("Scene does not declare reusable affordances.")
        self.scene = scene
        self.frame = str(metadata.get("coordinate_frame") or "")
        self.tolerance = tolerance
        self.world = WorldState(values=deepcopy(initial_values))
        self.used_entities: set[str] = set()

    def validate(
        self,
        plan: list[PlanStep],
    ) -> tuple[list[PlanStep], WorldState]:
        if not plan:
            raise PlanGroundingError("Planner returned an empty plan.")
        grounded: list[PlanStep] = []
        for raw_step in plan:
            step = deepcopy(raw_step)
            action_type = str(step.get("action_type") or "")
            if action_type not in DIRECT_SCENE_ACTIONS:
                raise PlanGroundingError(
                    f"Scene-grounded direct control does not support action "
                    f"{action_type!r}."
                )
            step["conditions"] = []
            step["effects"] = []
            step["on_condition_false"] = "fail"
            if action_type == "move_to_pose":
                self._move_to_pose(step)
            elif action_type == "move_linear":
                self._move_linear(step)
            elif action_type == "open_gripper":
                self._open_gripper(step)
            else:
                self._close_gripper(step)
            grounded.append(step)

        holding = self.world.get("runtime.holding")
        if holding is not None and not self._grasp_is_movable(str(holding)):
            raise PlanGroundingError(
                f"Plan ends while still holding fixed scene entity {holding!r}; "
                "release it before completing the task."
            )
        self._make_sequential(grounded)
        return grounded, self.world

    def _move_to_pose(self, step: PlanStep) -> None:
        parameters = dict(step.get("parameters") or {})
        affordance_id = str(parameters.get("affordance_id") or "")
        affordance = self.affordances.get(affordance_id)
        if affordance is None:
            raise PlanGroundingError(
                "move_to_pose requires a valid parameters.affordance_id copied "
                "from scene.affordances."
            )
        self._require_pose_contract(parameters, affordance)
        holding = self.world.get("runtime.holding")
        if holding is not None and not self._grasp_is_movable(str(holding)):
            raise PlanGroundingError(
                f"Cannot use move_to_pose while holding fixed entity {holding!r}; "
                "use its declared motion or release it."
            )
        conditions = self._require_contract_state(affordance, include_gripper=True)
        effects = [
            _set_effect("runtime.current_affordance_id", affordance_id),
            _set_effect(
                "runtime.cartesian_position_xyz_m",
                list(parameters["position_xyz_m"]),
            ),
        ]
        sequence_group = affordance.get("sequence_group")
        if sequence_group is not None:
            group = str(sequence_group)
            sequence_index = int(affordance.get("sequence_index", 0))
            progress_path = f"runtime.sequence_progress.{group}"
            progress = int(self.world.get(progress_path, 0))
            if sequence_index != progress + 1:
                raise PlanGroundingError(
                    f"Affordance {affordance_id!r} is sequence item "
                    f"{sequence_index}, but the next valid item is {progress + 1}."
                )
            if progress > 0:
                conditions.append(_eq_condition(progress_path, progress))
            effects.append(_set_effect(progress_path, sequence_index))
        effects.extend(_trusted_effects(affordance.get("arrival_effects", [])))
        self._commit(step, conditions, effects)
        self.used_entities.add(str(affordance.get("entity") or ""))

    def _move_linear(self, step: PlanStep) -> None:
        parameters = dict(step.get("parameters") or {})
        motion_id = str(parameters.get("motion_id") or "")
        motion = self.motions.get(motion_id)
        if motion is None:
            raise PlanGroundingError(
                "move_linear requires a valid parameters.motion_id copied "
                "from scene.motions."
            )
        self._require_motion_contract(parameters, motion)
        current = self.world.get("runtime.current_affordance_id")
        expected_origin = motion.get("from_affordance_id")
        if expected_origin and current != expected_origin:
            raise PlanGroundingError(
                f"Motion {motion_id!r} requires current affordance "
                f"{expected_origin!r}, got {current!r}."
            )
        conditions = self._require_contract_state(motion)
        effects = _trusted_effects(motion.get("effects", []))
        current_position = self.world.get(
            "runtime.cartesian_position_xyz_m",
            [0.0, 0.0, 0.0],
        )
        delta = list(parameters["delta_xyz_m"])
        next_position = [
            float(current_position[index]) + float(delta[index])
            for index in range(3)
        ]
        result_affordance = str(motion.get("result_affordance_id") or "")
        effects.extend(
            [
                _set_effect("runtime.current_affordance_id", result_affordance),
                _set_effect("runtime.cartesian_position_xyz_m", next_position),
            ]
        )
        self._commit(step, conditions, effects)
        self.used_entities.add(str(motion.get("entity") or ""))

    def _open_gripper(self, step: PlanStep) -> None:
        affordance = self._require_current_gripper_pose(step)
        holding = self.world.get("runtime.holding")
        effects: list[dict[str, Any]] = []
        if holding is not None:
            holding = str(holding)
            if self._grasp_is_movable(holding):
                accepted = [
                    str(value) for value in affordance.get("accepts_release", [])
                ]
                if holding not in accepted:
                    raise PlanGroundingError(
                        f"Affordance {affordance['affordance_id']!r} does not "
                        f"declare a safe release for held entity {holding!r}."
                    )
                effects.extend(
                    _trusted_effects(affordance.get("release_effects", []))
                )
            self.used_entities.add(
                str(affordance.get("entity") or holding)
            )
        effects.extend(
            [
                _set_effect("runtime.gripper", "open"),
                _set_effect("runtime.holding", None),
                _set_effect("runtime.held_state", {}),
            ]
        )
        self._commit(step, [], effects)

    def _close_gripper(self, step: PlanStep) -> None:
        affordance = self._require_current_gripper_pose(step)
        grasp_entity = affordance.get("grasp_entity")
        if not grasp_entity:
            raise PlanGroundingError(
                f"Affordance {affordance['affordance_id']!r} is not graspable."
            )
        if self.world.get("runtime.gripper") != "open":
            raise PlanGroundingError(
                "close_gripper requires the gripper to be open first."
            )
        if self.world.get("runtime.holding") is not None:
            raise PlanGroundingError(
                "close_gripper cannot grasp a second entity while another is held."
            )
        conditions = self._require_contract_state(
            affordance,
            include_gripper=False,
        )
        conditions.extend(
            [
                _eq_condition("runtime.gripper", "open"),
                _eq_condition("runtime.holding", None),
            ]
        )
        effects = _trusted_effects(affordance.get("grasp_effects", []))
        effects.extend(
            [
                _set_effect("runtime.gripper", "closed"),
                _set_effect("runtime.holding", str(grasp_entity)),
                _set_effect("runtime.held_state", {}),
            ]
        )
        self._commit(step, conditions, effects)
        self.used_entities.add(str(affordance.get("entity") or grasp_entity))

    def _require_current_gripper_pose(
        self,
        step: PlanStep,
    ) -> dict[str, Any]:
        parameters = dict(step.get("parameters") or {})
        affordance_id = str(parameters.get("at_affordance_id") or "")
        affordance = self.affordances.get(affordance_id)
        if affordance is None:
            raise PlanGroundingError(
                "Gripper actions require a valid parameters.at_affordance_id."
            )
        current_affordance = self.world.get("runtime.current_affordance_id")
        if affordance_id != current_affordance:
            raise PlanGroundingError(
                f"Gripper action declares affordance {affordance_id!r}, but the "
                f"robot is at {current_affordance!r}."
            )
        contract = dict(affordance.get("parameters") or {})
        _require_equal_vector(
            parameters.get("at_position_xyz_m"),
            contract.get("position_xyz_m"),
            "at_position_xyz_m",
            self.tolerance,
        )
        self._require_frame(parameters, contract)
        return affordance

    def _require_pose_contract(
        self,
        parameters: dict[str, Any],
        affordance: dict[str, Any],
    ) -> None:
        contract = dict(affordance.get("parameters") or {})
        _require_equal_vector(
            parameters.get("position_xyz_m"),
            contract.get("position_xyz_m"),
            "position_xyz_m",
            self.tolerance,
        )
        _require_equal_vector(
            parameters.get("orientation_xyzw"),
            contract.get("orientation_xyzw"),
            "orientation_xyzw",
            self.tolerance,
            expected_length=4,
        )
        self._require_frame(parameters, contract)

    def _require_motion_contract(
        self,
        parameters: dict[str, Any],
        motion: dict[str, Any],
    ) -> None:
        contract = dict(motion.get("parameters") or {})
        _require_equal_vector(
            parameters.get("delta_xyz_m"),
            contract.get("delta_xyz_m"),
            "delta_xyz_m",
            self.tolerance,
        )
        self._require_frame(parameters, contract)

    def _require_frame(
        self,
        parameters: dict[str, Any],
        contract: dict[str, Any],
    ) -> None:
        expected = str(contract.get("coordinate_frame") or self.frame)
        if parameters.get("coordinate_frame") != expected:
            raise PlanGroundingError(
                f"Planner used coordinate_frame={parameters.get('coordinate_frame')!r}; "
                f"affordance contract requires {expected!r}."
            )

    def _require_contract_state(
        self,
        contract: dict[str, Any],
        *,
        include_gripper: bool = False,
    ) -> list[dict[str, Any]]:
        conditions = [
            deepcopy(value)
            for value in contract.get("requires", [])
            if isinstance(value, dict)
        ]
        for raw in conditions:
            condition = Condition.from_dict(raw)
            if not self.world.evaluate(condition):
                raise PlanGroundingError(
                    f"Contract {contract.get('affordance_id') or contract.get('motion_id')!r} "
                    f"requires {condition.path} {condition.operator} "
                    f"{condition.value!r}, got {self.world.get(condition.path)!r}."
                )
        required_holding = contract.get("requires_holding")
        if required_holding is not None:
            actual_holding = self.world.get("runtime.holding")
            if actual_holding != required_holding:
                raise PlanGroundingError(
                    f"Contract {contract.get('affordance_id') or contract.get('motion_id')!r} "
                    f"requires holding {required_holding!r}, got {actual_holding!r}."
                )
            conditions.append(
                _eq_condition("runtime.holding", required_holding)
            )
        for key, expected in dict(
            contract.get("requires_held_state") or {}
        ).items():
            path = f"runtime.held_state.{key}"
            actual = self.world.get(path)
            if actual != expected:
                raise PlanGroundingError(
                    f"Contract {contract.get('affordance_id')!r} requires "
                    f"{path}={expected!r}, got {actual!r}."
                )
            conditions.append(_eq_condition(path, expected))
        if include_gripper and contract.get("requires_gripper") is not None:
            expected_gripper = contract["requires_gripper"]
            actual_gripper = self.world.get("runtime.gripper")
            if actual_gripper != expected_gripper:
                raise PlanGroundingError(
                    f"Affordance {contract.get('affordance_id')!r} requires "
                    f"gripper={expected_gripper!r}, got {actual_gripper!r}."
                )
            conditions.append(
                _eq_condition("runtime.gripper", expected_gripper)
            )
        return conditions

    def _grasp_is_movable(self, grasp_entity: str) -> bool:
        matches = [
            bool(value.get("grasp_movable", False))
            for value in self.affordances.values()
            if value.get("grasp_entity") == grasp_entity
        ]
        return any(matches)

    def _commit(
        self,
        step: PlanStep,
        conditions: list[dict[str, Any]],
        effects: list[dict[str, Any]],
    ) -> None:
        step["conditions"] = _deduplicate(conditions)
        step["effects"] = _deduplicate(effects)
        self.world.apply_effects(step["effects"])

    @staticmethod
    def _make_sequential(plan: list[PlanStep]) -> None:
        known_ids = {int(step["step_id"]) for step in plan}
        for index, step in enumerate(plan):
            dependencies = {
                int(value)
                for value in step.get("depends_on", [])
                if int(value) in known_ids
            }
            if index:
                dependencies.add(int(plan[index - 1]["step_id"]))
            step["depends_on"] = sorted(dependencies)


def _validate_goal(
    goal: dict[str, Any] | None,
    scene: dict[str, Any],
) -> dict[str, Any] | None:
    predicates = list((scene.get("scene") or {}).get("goal_predicates") or [])
    if not predicates:
        return deepcopy(goal) if goal is not None else None
    if not isinstance(goal, dict):
        raise PlanGroundingError(
            "Planner must declare a machine-verifiable goal for this scene."
        )
    description = str(goal.get("description") or "").strip()
    conditions = goal.get("conditions")
    if not description or not isinstance(conditions, list) or not conditions:
        raise PlanGroundingError(
            "Task goal requires a description and at least one final condition."
        )
    allowed = {
        str(item.get("path")): list(item.get("allowed_values") or [])
        for item in predicates
        if isinstance(item, dict) and item.get("path")
    }
    normalized: list[dict[str, Any]] = []
    for raw in conditions:
        if not isinstance(raw, dict):
            raise PlanGroundingError("Every task goal condition must be an object.")
        path = str(raw.get("path") or "")
        if raw.get("operator") != "eq" or path not in allowed:
            raise PlanGroundingError(
                f"Goal condition {path!r} is not an allowed equality predicate."
            )
        value = raw.get("value")
        if not any(value == candidate for candidate in allowed[path]):
            raise PlanGroundingError(
                f"Goal value {value!r} is not allowed for path {path!r}."
            )
        normalized.append({"path": path, "operator": "eq", "value": value})
    return {"description": description, "conditions": normalized}


def _require_task_entities_addressed(
    user_task: str,
    plan: list[PlanStep],
    scene: dict[str, Any],
    used_entities: set[str],
) -> None:
    normalized_task = user_task.lower()
    objects = dict(scene.get("objects") or {})
    mentioned: set[str] = set()
    for entity, value in objects.items():
        aliases = [str(entity), *list(dict(value or {}).get("aliases", []))]
        if any(str(alias).lower() in normalized_task for alias in aliases):
            mentioned.add(str(entity))
    plan_text = " ".join(
        [
            *[str(step.get("target") or "") for step in plan],
            *[
                json.dumps(step.get("parameters") or {}, ensure_ascii=False)
                for step in plan
            ],
        ]
    ).lower()
    missing = [
        entity
        for entity in sorted(mentioned)
        if entity not in used_entities
        and not any(
            str(alias).lower() in plan_text
            for alias in [
                entity,
                *list(dict(objects.get(entity) or {}).get("aliases", [])),
            ]
        )
    ]
    if missing:
        raise PlanGroundingError(
            f"Plan does not address entities explicitly requested by the user: {missing}."
        )


def _stored_goal(values: dict[str, Any]) -> dict[str, Any] | None:
    current: Any = values
    for part in ("_agent", "task_goal"):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return deepcopy(current) if isinstance(current, dict) else None


def _world_values(
    context: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    snapshot = context.get("world_state")
    if isinstance(snapshot, dict) and isinstance(snapshot.get("values"), dict):
        return deepcopy(snapshot["values"])
    return deepcopy(fallback)


def _unsatisfied_goal_conditions(
    goal: dict[str, Any],
    world: WorldState,
) -> list[dict[str, Any]]:
    return [
        deepcopy(raw)
        for raw in goal.get("conditions", [])
        if not world.evaluate(Condition.from_dict(raw))
    ]


def _goal_signature(goal: dict[str, Any] | None) -> tuple[str, ...]:
    if goal is None:
        return ()
    return tuple(
        sorted(
            f"{item['path']}={json.dumps(item.get('value'), ensure_ascii=False, sort_keys=True)}"
            for item in goal.get("conditions", [])
        )
    )


def _trusted_effects(raw_effects: Any) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    for raw in raw_effects if isinstance(raw_effects, list) else []:
        if (
            not isinstance(raw, dict)
            or not str(raw.get("path") or "").strip()
            or raw.get("operation") not in {"set", "delete", "increment"}
        ):
            raise PlanGroundingError("Scene contains an invalid trusted effect.")
        effects.append(deepcopy(raw))
    return effects


def _set_effect(path: str, value: Any) -> dict[str, Any]:
    return {"path": path, "operation": "set", "value": deepcopy(value)}


def _eq_condition(path: str, value: Any) -> dict[str, Any]:
    return {"path": path, "operator": "eq", "value": deepcopy(value)}


def _require_equal_vector(
    actual: Any,
    expected: Any,
    field: str,
    tolerance: float,
    *,
    expected_length: int = 3,
) -> None:
    if not _is_vector(actual, expected_length) or not _is_vector(
        expected, expected_length
    ):
        raise PlanGroundingError(
            f"{field} must copy a calibrated {expected_length}-value vector "
            "from its scene contract."
        )
    if any(
        abs(float(actual[index]) - float(expected[index])) > tolerance
        for index in range(expected_length)
    ):
        raise PlanGroundingError(
            f"Planner used ungrounded {field}={actual}; contract requires {expected}."
        )


def _is_vector(value: Any, expected_length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected_length
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        )
    )


def _deduplicate(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            result.append(deepcopy(value))
            seen.add(key)
    return result
