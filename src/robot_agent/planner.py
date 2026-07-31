from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .events import record_event
from .state import PlanStep, RobotState
from .step_protocol import PLAN_JSON_SCHEMA, build_plan_json_schema, normalize_plan

DEFAULT_SKILLS = (
    "pick",
    "place",
    "manipulate",
    "move_to",
    "move_to_pose",
    "move_linear",
    "move_home",
    "inspect",
    "move_relative",
    "open_gripper",
    "close_gripper",
)
PLAN_SCHEMA = PLAN_JSON_SCHEMA


def _allows_virtual_entities_context(context: dict[str, Any]) -> bool:
    snapshot = dict(context.get("world_state") or {})
    values = dict(snapshot.get("values") or {})
    scene = dict(values.get("scene") or {})
    return bool(
        scene.get("open_world_simulation")
        or scene.get("mode") == "open_world_simulation"
        or scene.get("simulation_allow_virtual_entities")
    )


class Planner(Protocol):
    """An LLM planner must return skill-level steps, never motor commands."""

    def create_plan(self, user_task: str) -> list[PlanStep]:
        """Create a plan using only skills registered by the robot project."""


class ReplanningPlanner(Planner, Protocol):
    """A Planner that can revise its plan from structured execution feedback."""

    def revise_plan(
        self,
        user_task: str,
        previous_plan: list[PlanStep],
        failed_step: PlanStep,
        feedback: dict[str, Any],
    ) -> list[PlanStep]:
        """Return a replacement plan for the remaining task."""

    def revise_from_failure(self, context: dict[str, Any]) -> list[PlanStep]:
        """Return only the unfinished suffix using current execution context."""


class RuleBasedPlanner:
    """Deterministic planner retained for tests and offline development."""

    def create_plan(self, user_task: str) -> list[PlanStep]:
        return self._create_default_plan(user_task)

    def create_plan_with_context(self, context: dict[str, Any]) -> list[PlanStep]:
        return self.create_plan(str(context["original_task"]))

    def revise_from_failure(self, context: dict[str, Any]) -> list[PlanStep]:
        failed_step = dict(context["current_step"])
        failed_step["status"] = "pending"
        failed_step.pop("instance_id", None)
        return [failed_step]

    def _create_default_plan(self, user_task: str) -> list[PlanStep]:
        normalized_task = user_task.lower()
        motion_step = self._create_basic_motion_step(normalized_task)
        if motion_step:
            return [motion_step]

        if "pizza" in normalized_task or "dough" in normalized_task:
            return [
                {
                    "id": 1,
                    "skill": "pick",
                    "target": "dough",
                    "expected_result": "dough is held by the gripper",
                },
                {
                    "id": 2,
                    "skill": "place",
                    "target": "tray",
                    "expected_result": "dough is on the tray",
                },
            ]

        return [
            {
                "id": 1,
                "skill": "pick",
                "target": "target_object",
                "expected_result": "target object is held by the gripper",
            }
        ]

    def _create_basic_motion_step(self, normalized_task: str) -> PlanStep | None:
        if "open gripper" in normalized_task or "打开夹爪" in normalized_task:
            return {
                "id": 1,
                "skill": "open_gripper",
                "target": "gripper",
                "expected_result": "gripper is open",
            }
        if "close gripper" in normalized_task or "关闭夹爪" in normalized_task:
            return {
                "id": 1,
                "skill": "close_gripper",
                "target": "gripper",
                "expected_result": "gripper is closed",
            }
        if "home" in normalized_task or "初始位置" in normalized_task or "回零" in normalized_task:
            return {
                "id": 1,
                "skill": "move_home",
                "target": "home",
                "expected_result": "arm is at the safe home pose",
            }

        direction_aliases = {
            "left": ("left", "向左", "左移"),
            "right": ("right", "向右", "右移"),
            "forward": ("forward", "向前", "前移"),
            "back": ("back", "向后", "后移"),
            "up": ("up", "向上", "上移"),
            "down": ("down", "向下", "下移"),
        }
        direction = next(
            (
                canonical
                for canonical, aliases in direction_aliases.items()
                if any(alias in normalized_task for alias in aliases)
            ),
            None,
        )
        if direction is None:
            return None

        distance_cm = self._extract_distance_cm(normalized_task)
        return {
            "id": 1,
            "skill": "move_relative",
            "target": f"{direction} {distance_cm:g} cm",
            "expected_result": f"arm moved {direction} by {distance_cm:g} cm",
        }

    @staticmethod
    def _extract_distance_cm(normalized_task: str) -> float:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(厘米|cm|centimeter|centimeters)", normalized_task)
        if match:
            return float(match.group(1))
        mm_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(毫米|mm|millimeter|millimeters)", normalized_task)
        if mm_match:
            return float(mm_match.group(1)) / 10
        return 0.5


class OllamaPlannerError(RuntimeError):
    """Raised when the local model service cannot produce a valid plan."""


class PlannerServiceError(RuntimeError):
    """Raised when a planner backend cannot produce a usable plan."""


class JsonSchemaPlannerMixin:
    """Common structured-output planner behavior shared by LLM backends."""

    allowed_skills: tuple[str, ...]

    def create_plan(self, user_task: str) -> list[PlanStep]:
        request = f"User task: {user_task}"
        draft = self._request_plan(request)
        return self._review_if_enabled(request, draft)

    def create_plan_with_context(self, context: dict[str, Any]) -> list[PlanStep]:
        serialized = json.dumps(context, ensure_ascii=False, default=str)
        if _allows_virtual_entities_context(context):
            entity_rule = (
                "This context explicitly permits virtual targets in SIMULATION. "
                "The objects map may be empty or partial: extract concrete semantic "
                "target names that are explicitly present in original_task. Local "
                "code will register only missing targets as virtual entities. If "
                "any target is new, set goal.conditions to []. "
                "Do not invent properties, poses, coordinates, affordances, skill "
                "contracts, executors, policies, or extra objects. "
            )
        else:
            entity_rule = (
                "This is a closed world: use only entities present in the supplied "
                "objects map. "
            )
        request = (
            "Create a HIGH-LEVEL skill plan for this task using only capabilities "
            "listed in the context. Respect the supplied world_state. Compose a new "
            "plan for this specific request instead of recalling a task recipe. "
            "Do not expand pick/place/manipulate into Cartesian poses, gripper cycles, "
            "joint commands, or policy actions: trusted local skills do that work. "
            "Do not select executor or policy_id; the local Router owns that decision. "
            "When scene.skill_contracts are present, select and compose their "
            "action_type, target, and parameters_match values while satisfying "
            "their preconditions. They are reusable skills, not a task workflow. "
            f"{entity_rule}"
            "If required scene information or a required skill is missing, do not "
            "invent coordinates or unsupported actions. "
            f"\nPlanning context: {serialized}"
        )
        draft = self._request_plan(request)
        return self._review_if_enabled(request, draft)

    def revise_plan(
        self,
        user_task: str,
        previous_plan: list[PlanStep],
        failed_step: PlanStep,
        feedback: dict[str, Any],
    ) -> list[PlanStep]:
        context = json.dumps(
            {
                "user_task": user_task,
                "previous_plan": previous_plan,
                "failed_step": failed_step,
                "feedback": feedback,
            },
            ensure_ascii=False,
        )
        return self._request_plan(
            "Revise the plan using this execution feedback. Return a complete "
            f"replacement plan for the user task.\nContext: {context}"
        )

    def revise_from_failure(self, context: dict[str, Any]) -> list[PlanStep]:
        serialized = json.dumps(context, ensure_ascii=False, default=str)
        entity_rule = (
            "New concrete targets explicitly present in original_task may be used "
            "as virtual entities, but no properties or coordinates may be invented. "
            if _allows_virtual_entities_context(context)
            else "Use only entities present in the current world_state. "
        )
        request = (
            "The current robot plan failed after local recovery was exhausted. "
            "Return ONLY the unfinished suffix starting from the current physical state. "
            "Never repeat completed_steps. Use only the supplied available capabilities. "
            "Re-evaluate scene affordances and their current preconditions instead of "
            "replaying a memorized complete task. Keep the original task goal unchanged. "
            f"{entity_rule}"
            f"\nFailure context: {serialized}"
        )
        draft = self._request_plan(request)
        return self._review_if_enabled(
            request,
            draft,
            suffix_only=True,
        )

    def _review_if_enabled(
        self,
        original_request: str,
        draft: list[PlanStep],
        *,
        suffix_only: bool = False,
    ) -> list[PlanStep]:
        if not bool(getattr(self, "review_plans", False)):
            return draft
        draft_document = {
            "goal": getattr(self, "last_goal", None),
            "steps": [
                {
                    key: value
                    for key, value in step.items()
                    if key
                    not in {
                        "id",
                        "skill",
                        "instance_id",
                        "executor",
                        "policy_id",
                    }
                }
                for step in draft
            ],
        }
        suffix_rule = (
            "The reviewed result must still contain only the unfinished suffix and "
            "must not repeat completed_steps. "
            if suffix_only
            else ""
        )
        return self._request_plan(
            "Independently review the draft robot plan against the original user task "
            "and the complete scene context. Correct omitted subgoals, invalid object "
            "access, impossible gripper state transitions, unsupported coordinates, "
            "and an incorrect final goal. Build the corrected plan from the declared "
            "affordances; do not copy a memorized task recipe. "
            f"{suffix_rule}"
            "Return the final corrected JSON document only."
            f"\nOriginal request and context: {original_request}"
            "\nDraft document: "
            + json.dumps(draft_document, ensure_ascii=False, default=str)
        )

    def repair_rejected_plan(
        self,
        context: dict[str, Any],
        rejected_plan: list[PlanStep],
        validation_error: str,
        *,
        suffix_only: bool = False,
    ) -> list[PlanStep]:
        suffix_rule = (
            "Return only the unfinished suffix and do not repeat completed_steps. "
            if suffix_only
            else ""
        )
        rejected_document = {
            "goal": getattr(self, "last_goal", None),
            "steps": [
                {
                    key: value
                    for key, value in step.items()
                    if key
                    not in {
                        "id",
                        "skill",
                        "instance_id",
                        "executor",
                        "policy_id",
                    }
                }
                for step in rejected_plan
            ],
        }
        return self._request_plan(
            "The trusted local semantic validator rejected the reviewed plan. "
            "Use the exact validation error, current world state, scene affordances, "
            "motions, and original user goal to repair it. Do not weaken or change "
            f"the task goal. {suffix_rule}"
            "\nValidation error: "
            + validation_error
            + "\nPlanning context: "
            + json.dumps(context, ensure_ascii=False, default=str)
            + "\nRejected document: "
            + json.dumps(rejected_document, ensure_ascii=False, default=str)
        )

    def _system_prompt(self) -> str:
        allowed_skills = ", ".join(self.allowed_skills)
        return f"""You are a robot task-planning Agent.
Convert each user request into a fresh, short, HIGH-LEVEL skill plan using only:
{allowed_skills}.

Return JSON only. The JSON must have exactly this canonical shape:
{{
  "goal": {{
    "description": "the final state requested by the user",
    "conditions": [
      {{
        "path": "an allowed scene goal path",
        "operator": "eq",
        "value": "an allowed final value"
      }}
    ]
  }},
  "steps": [
    {{
      "step_id": 1,
      "action_type": "pick",
      "target": "object name",
      "expected_result": "observable success condition",
      "status": "pending",
      "parameters": {{}}
    }}
  ]
}}

Rules:
- Do not output executor or policy_id. The trusted local Router selects them.
- Do not output joint angles, Cartesian coordinates, motor actions, action chunks,
  velocities, code, explanations, or markdown.
- Use only the allowed skills.
- Each step must have a unique integer step_id starting at 1.
- Keep complex operations at skill level. For example, use pick(brush), not a
  sequence of move-to-pose/open-gripper/close-gripper/lift commands.
- Use a direct robot primitive only when the user explicitly requests that
  primitive and all required parameters are supplied by trusted context.
- Never infer or copy physical coordinates into the plan. Target grounding and
  calibrated coordinates belong to perception/skill/controller layers.
- Derive the goal from the current user request. When scene.goal_predicates is
  supplied, goal.conditions must use only its declared paths and values.
- When scene.skill_contracts is supplied, policy steps must match a declared
  action_type, target, and parameters_match. Satisfy contract preconditions and
  compose contracts as needed; never copy an entire memorized task workflow.
- For small relative movement commands, use move_relative and set target exactly like
  "left 0.5 cm", "right 0.5 cm", "forward 0.5 cm", "back 0.5 cm",
  "up 0.5 cm", or "down 0.5 cm".
- For gripper commands, use open_gripper or close_gripper with target "gripper".
- Add move_home only when it is needed for safety or task completion.
- Keep targets concrete and observable.
- Do not omit an explicit action or destination requested by the user.
- If every declared goal condition is already true in the supplied current
  world_state, return steps=[] instead of inventing unnecessary robot motion.
- When failure context is supplied, return only the unfinished suffix. Never repeat
  an item in completed_steps.
- Never claim an action succeeded in the plan; success is decided by feedback.
- Use depends_on for dependency edges when a step requires another step.
- Do not invent conditions or effects. Trusted local skill contracts derive
  state transitions.
- Give each step a positive timeout_seconds and max_attempts.

Example:
User task: Make a pizza by picking dough and placing it on a tray.
Valid response:
{{
  "goal": {{
    "description": "the dough is on the tray",
    "conditions": []
  }},
  "steps": [
    {{
      "step_id": 1,
      "action_type": "pick",
      "target": "dough",
      "expected_result": "dough is held by the gripper",
      "status": "pending",
      "parameters": {{}}
    }},
    {{
      "step_id": 2,
      "action_type": "place",
      "target": "tray",
      "expected_result": "dough is on the tray",
      "status": "pending",
      "parameters": {{}}
    }}
  ]
}}"""

    @staticmethod
    def _parse_plan_document(
        content: str,
        allowed_skills: tuple[str, ...] | None = None,
    ) -> tuple[list[PlanStep], dict[str, Any] | None]:
        try:
            decoded = json.loads(content)
            steps = decoded["steps"]
        except (TypeError, KeyError, json.JSONDecodeError) as error:
            raise PlannerServiceError(
                "The planner backend returned invalid plan JSON. The plan was not sent to the robot."
            ) from error

        if not isinstance(steps, list):
            raise PlannerServiceError("Plan field 'steps' must be a JSON list.")

        if any(not isinstance(step, dict) for step in steps):
            raise PlannerServiceError("Every plan step must be a JSON object.")
        try:
            normalized_steps = normalize_plan(
                steps,
                allowed_skills=allowed_skills,
            )
        except ValueError as error:
            raise PlannerServiceError(str(error)) from error
        for step in normalized_steps:
            _validate_model_step_parameters(step)
        # Keep aliases until the legacy graph/simple workflow is retired.
        for step in normalized_steps:
            step["id"] = int(step["step_id"])
            step["skill"] = str(step["action_type"])
        goal = decoded.get("goal")
        if goal is not None:
            if not isinstance(goal, dict):
                raise PlannerServiceError("Plan field 'goal' must be a JSON object.")
            description = str(goal.get("description") or "").strip()
            conditions = goal.get("conditions")
            if not description or not isinstance(conditions, list):
                raise PlannerServiceError(
                    "Plan goal requires a description and a conditions list."
                )
            normalized_conditions: list[dict[str, Any]] = []
            for condition in conditions:
                if (
                    not isinstance(condition, dict)
                    or not str(condition.get("path") or "").strip()
                    or condition.get("operator") != "eq"
                    or "value" not in condition
                ):
                    raise PlannerServiceError(
                        "Every goal condition requires path, operator='eq', and value."
                    )
                normalized_conditions.append(
                    {
                        "path": str(condition["path"]).strip(),
                        "operator": "eq",
                        "value": condition["value"],
                    }
                )
            goal = {
                "description": description,
                "conditions": normalized_conditions,
            }
        return normalized_steps, goal

    @staticmethod
    def _parse_plan(content: str) -> list[PlanStep]:
        steps, _ = JsonSchemaPlannerMixin._parse_plan_document(content)
        return steps

    def _parse_plan_response(self, content: str) -> list[PlanStep]:
        steps, goal = self._parse_plan_document(content, self.allowed_skills)
        self.last_goal = goal
        return steps


def _validate_model_step_parameters(step: PlanStep) -> None:
    """Keep model output out of trusted grounding and motor-space fields."""

    parameters = dict(step.get("parameters") or {})
    reserved = {
        "perception_grounding",
        "target_aliases",
        "monitor_targets",
        "_trusted_target_aliases",
    }
    supplied_reserved = sorted(reserved.intersection(parameters))
    if supplied_reserved:
        raise PlannerServiceError(
            "Planner parameters contain trusted runtime fields: "
            + ", ".join(supplied_reserved)
        )
    if str(step.get("action_type") or "") not in {
        "pick",
        "place",
        "manipulate",
        "inspect",
        "move_to",
    }:
        return
    blocked_fragments = (
        "xyz",
        "pose",
        "joint",
        "quaternion",
        "orientation",
        "velocity",
        "speed",
        "motor",
        "action_chunk",
    )

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).casefold().replace("-", "_")
                if any(fragment in normalized for fragment in blocked_fragments):
                    raise PlannerServiceError(
                        "High-level Planner parameters cannot contain motor-space "
                        f"field '{path + str(key)}'."
                    )
                walk(child, path + str(key) + ".")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}{index}.")

    walk(parameters, "parameters.")


class OllamaPlanner(JsonSchemaPlannerMixin):
    """Open-source LLM planner served locally through Ollama."""

    def __init__(
        self,
        model: str | None = None,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout_seconds: int = 120,
        allowed_skills: tuple[str, ...] = DEFAULT_SKILLS,
    ) -> None:
        self.model = model or os.getenv("ROBOT_AGENT_MODEL", "qwen3:4b")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.allowed_skills = allowed_skills
        self.last_goal: dict[str, Any] | None = None
        self.review_plans = False

    def _request_plan(self, user_message: str) -> list[PlanStep]:
        plan_schema = build_plan_json_schema(self.allowed_skills)
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": plan_schema,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_message},
            ],
        }
        response = self._post_json(payload)
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as error:
            raise OllamaPlannerError("Ollama response did not contain message.content.") from error
        return self._parse_plan_response(content)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise OllamaPlannerError(
                f"Could not obtain a plan from Ollama model '{self.model}'. "
                "Start Ollama and verify the model is installed with "
                f"'ollama list'. Original error: {error}"
            ) from error


class QwenApiPlanner(JsonSchemaPlannerMixin):
    """Planner that calls Qwen through an OpenAI-compatible HTTP API."""

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        allowed_skills: tuple[str, ...] = DEFAULT_SKILLS,
        review_plans: bool = True,
    ) -> None:
        self.model = model or os.getenv("ROBOT_AGENT_API_MODEL", "qwen-plus")
        self.endpoint = endpoint or os.getenv(
            "ROBOT_AGENT_API_ENDPOINT",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.allowed_skills = allowed_skills
        self.last_goal: dict[str, Any] | None = None
        self.review_plans = review_plans
        self.request_count = 0
        if not self.api_key:
            raise PlannerServiceError(
                "Qwen API key is missing. Set DASHSCOPE_API_KEY or QWEN_API_KEY before running the planner."
            )

    def _request_plan(self, user_message: str) -> list[PlanStep]:
        self.request_count += 1
        plan_schema = build_plan_json_schema(self.allowed_skills)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_message},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_skill_plan",
                    "schema": plan_schema,
                },
            },
        }
        response = self._post_json(payload)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise PlannerServiceError("Qwen API response did not contain choices[0].message.content.") from error
        return self._parse_plan_response(content)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise PlannerServiceError(
                f"Qwen API request failed with HTTP {error.code}. Response body: {error_body}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise PlannerServiceError(
                f"Could not obtain a plan from Qwen API model '{self.model}'. Original error: {error}"
            ) from error


def create_default_planner() -> Planner:
    """Select a planner backend from environment configuration."""
    provider = os.getenv("ROBOT_AGENT_PLANNER_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return OllamaPlanner()
    if provider in {"qwen_api", "qwen-api", "dashscope"}:
        return QwenApiPlanner()
    raise PlannerServiceError(
        "Unsupported planner provider. Set ROBOT_AGENT_PLANNER_PROVIDER to 'ollama' or 'qwen_api'."
    )


def plan_task(state: RobotState, planner: Planner | None = None) -> dict:
    """Create a skill-level plan and retain the original user request."""
    selected_planner = planner or RuleBasedPlanner()
    plan = selected_planner.create_plan(state["user_task"])
    update: dict = {
        "plan": plan,
        "current_step_index": 0,
        "retry_count": 0,
        "replan_count": 0,
        "status": "planning",
    }
    update.update(record_event(state, "plan.created", f"Planner created {len(plan)} skill steps."))
    return update


def replan_task(state: RobotState, planner: Planner | None = None) -> dict:
    """Ask an LLM planner for a replacement plan after a non-local failure."""
    max_replans = state.get("max_replans", 1)
    current_replans = state.get("replan_count", 0)
    if current_replans >= max_replans:
        update: dict = {"status": "needs_agent_replan"}
        update.update(
            record_event(
                state,
                "agent.replan_limit_reached",
                "Automatic replanning limit reached; operator review is required.",
                step_id=state["current_step"]["id"],
            )
        )
        return update

    selected_planner = planner or RuleBasedPlanner()
    revise_plan = getattr(selected_planner, "revise_plan", None)
    if not callable(revise_plan):
        update = {"status": "needs_agent_replan"}
        update.update(
            record_event(
                state,
                "agent.replan_requested",
                "The selected Planner cannot revise plans; operator review is required.",
                step_id=state["current_step"]["id"],
            )
        )
        return update

    try:
        plan = revise_plan(
            state["user_task"],
            state["plan"],
            state["current_step"],
            state["feedback"],
        )
    except PlannerServiceError as error:
        update = {"status": "needs_agent_replan"}
        update.update(
            record_event(
                state,
                "agent.replan_failed",
                str(error),
                step_id=state["current_step"]["id"],
            )
        )
        return update

    next_count = current_replans + 1
    update = {
        "plan": plan,
        "current_step_index": 0,
        "retry_count": 0,
        "replan_count": next_count,
        "status": "planning",
    }
    update.update(
        record_event(
            state,
            "agent.replanned",
            f"Planner created a replacement plan with {len(plan)} skill steps.",
            step_id=state["current_step"]["id"],
            data={"replan_count": next_count, "failure_reason": state["feedback"]["reason"]},
        )
    )
    return update
