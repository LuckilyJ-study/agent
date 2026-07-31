from __future__ import annotations

import time
import threading
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, Sequence

from .capabilities import CapabilityRegistry
from .control import ExecutionControl, TaskCancelled
from .domain import WorldState
from .local_recovery import LocalRecoveryHandler, NullLocalRecoveryHandler
from .monitor import summarize_observation
from .persistence import NullTaskStore, TaskStore
from .routing import ExecutorRouter, RoutingError
from .safety_monitor import RuntimeSafetyMonitor, SoftwareSafetyMonitor
from .state import PlanStep, StepMemory, VerificationResult
from .task_scheduler import SchedulingError, TaskGraphScheduler
from .task_verifier import PlanCompletionTaskVerifier, TaskVerifier


class StateProvider(Protocol):
    def observe(self) -> dict[str, Any]: ...

    def robot_state(self) -> dict[str, Any]: ...


class ActionExecutor(Protocol):
    def execute(self, step: PlanStep, observation: dict[str, Any]) -> dict[str, Any]: ...

    def stop(self) -> None: ...


class ActionVerifier(Protocol):
    def verify(
        self,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
        action: dict[str, Any],
        expected_result: str,
    ) -> VerificationResult: ...


class PreActionGate(Protocol):
    """Trusted deployment gate evaluated before routing or execution."""

    def before_action(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> VerificationResult | None: ...


class ClosedLoopPlanner(Protocol):
    def create_plan(self, user_task: str) -> list[PlanStep]: ...

    def revise_from_failure(self, context: dict[str, Any]) -> list[PlanStep]: ...


RecoveryAction = Literal["retry", "reobserve", "stop", "replan"]


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str


class RecoveryPolicy:
    def __init__(self, max_local_retries: int = 1) -> None:
        if max_local_retries < 0:
            raise ValueError("max_local_retries cannot be negative.")
        self.max_local_retries = max_local_retries

    def decide(self, error_type: str, attempts: int) -> RecoveryDecision:
        error = error_type.upper()
        if error in {
            "COLLISION_RISK",
            "EMERGENCY_STOP",
            "HARDWARE_FAULT",
            "ROBOT_DISCONNECTED",
            "FORBIDDEN_TARGET",
            "INVALID_ACTION_TIMEOUT",
            "ACTION_TIMEOUT",
            "ACTION_STOP_UNCONFIRMED",
            "EXECUTOR_NOT_AVAILABLE",
            "EXECUTOR_ROUTE_MISMATCH",
            "POLICY_ACTION_MISMATCH",
            "POLICY_ROUTE_AMBIGUOUS",
            "POLICY_STOP_UNSUPPORTED",
            "PERCEPTION_UNAVAILABLE",
            "ROBOT_STATE_FAILED",
        }:
            return RecoveryDecision("stop", f"Safety-critical error: {error}")
        if error == "OBJECT_DROPPED":
            return RecoveryDecision(
                "replan",
                "The object changed the physical scene after being dropped.",
            )
        if error in {
            "TARGET_NOT_VISIBLE",
            "TARGET_NOT_LOCALIZED",
            "TARGET_LOST",
            "PERCEPTION_FAILED",
            "PERCEPTION_STALE",
        }:
            if attempts <= self.max_local_retries:
                return RecoveryDecision("reobserve", "Refresh perception before retrying.")
            return RecoveryDecision("replan", "Perception recovery did not restore the target.")
        if error in {
            "POLICY_NOT_AVAILABLE",
            "POLICY_INPUTS_UNAVAILABLE",
            "UNSUPPORTED_ROBOT_PRIMITIVE",
            "PI05_SERVICE_UNAVAILABLE",
        }:
            return RecoveryDecision("stop", f"Required capability is unavailable: {error}")
        if error in {"LOW_CONFIDENCE", "VERIFICATION_UNCERTAIN"}:
            if attempts <= self.max_local_retries:
                return RecoveryDecision(
                    "reobserve", "Refresh observations before trusting this result."
                )
            return RecoveryDecision("replan", "Verification remained uncertain.")
        if error == "GRASP_FAILED":
            if attempts <= self.max_local_retries:
                return RecoveryDecision(
                    "retry", "Retreat/relocalize locally, then retry the grasp once."
                )
            return RecoveryDecision("replan", "Repeated grasp failure needs a new suffix.")
        if attempts <= self.max_local_retries:
            return RecoveryDecision("retry", "Retry the current step using the existing plan.")
        return RecoveryDecision("replan", "Local retry limit reached.")


class SafetyStop(RuntimeError):
    def __init__(self, reason: str, memory: TaskMemory | None = None) -> None:
        super().__init__(reason)
        self.memory = memory


class AgentExecutionError(RuntimeError):
    def __init__(self, reason: str, memory: TaskMemory | None = None) -> None:
        super().__init__(reason)
        self.memory = memory


class PlaceholderActionVerifier:
    """Command-level verifier used until the camera Monitor is connected."""

    def __init__(self, success_confidence: float = 1.0) -> None:
        self.success_confidence = success_confidence

    def verify(
        self,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
        action: dict[str, Any],
        expected_result: str,
    ) -> VerificationResult:
        success = action.get("status") == "success"
        return {
            "success": success,
            "error_type": "NONE" if success else str(action.get("reason", "EXECUTION_FAILED")),
            "confidence": self.success_confidence,
            "verification_scope": "command",
            "details": {
                "expected_result": expected_result,
                "physical_result_verified": bool(action.get("physical_result_verified", False)),
                "perception_available": bool(observation.get("available", False)),
            },
        }


TaskStatus = Literal[
    "running", "paused", "completed", "safety_stopped", "cancelled", "failed"
]


@dataclass
class TaskMemory:
    original_task: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    records: list[StepMemory] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    plan: list[PlanStep] = field(default_factory=list)
    world_state: dict[str, Any] = field(
        default_factory=lambda: {"version": 0, "values": {}}
    )
    current_step_id: int | None = None
    status: TaskStatus = "running"
    replan_count: int = 0
    failure_reason: str | None = None
    task_verification: dict[str, Any] | None = None
    attempts: dict[str, int] = field(default_factory=dict)
    inflight_action: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: _now())
    updated_at: str = field(default_factory=lambda: _now())

    @property
    def completed_steps(self) -> list[PlanStep]:
        return [record["step"] for record in self.records if record["status"] == "completed"]

    def save(
        self,
        step: PlanStep,
        status: Literal["completed", "failed"],
        observation: dict[str, Any],
        robot_state: dict[str, Any],
        verification: VerificationResult,
    ) -> None:
        self.current_step_id = int(step["step_id"])
        self.records.append(
            {
                "step": dict(step),
                "status": status,
                "observation": _compact_observation(observation),
                "robot_state": robot_state,
                "verification": verification,
            }
        )
        self.updated_at = _now()

    def event(self, event_type: str, **data: Any) -> None:
        self.events.append({"type": event_type, "timestamp": _now(), "data": data})
        self.updated_at = _now()

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_snapshot(cls, raw: dict[str, Any]) -> TaskMemory:
        return cls(
            original_task=str(raw["original_task"]),
            task_id=str(raw["task_id"]),
            records=list(raw.get("records", [])),
            events=list(raw.get("events", [])),
            plan=list(raw.get("plan", [])),
            world_state=dict(raw.get("world_state", {"version": 0, "values": {}})),
            current_step_id=raw.get("current_step_id"),
            status=raw.get("status", "running"),
            replan_count=int(raw.get("replan_count", 0)),
            failure_reason=raw.get("failure_reason"),
            task_verification=raw.get("task_verification"),
            attempts={
                str(key): int(value)
                for key, value in dict(raw.get("attempts") or {}).items()
            },
            inflight_action=raw.get("inflight_action"),
            created_at=str(raw.get("created_at", _now())),
            updated_at=str(raw.get("updated_at", _now())),
        )


@dataclass(frozen=True)
class AgentRunResult:
    status: Literal["completed", "safety_stopped", "cancelled", "failed"]
    memory: TaskMemory
    reason: str | None = None


class ClosedLoopAgent:
    def __init__(
        self,
        planner: ClosedLoopPlanner,
        robot_executor: ActionExecutor,
        policy_executor: ActionExecutor,
        state_provider: StateProvider,
        verifier: ActionVerifier | None = None,
        recovery_policy: RecoveryPolicy | None = None,
        capabilities: CapabilityRegistry | None = None,
        confidence_threshold: float = 0.7,
        available_skills: Sequence[str] | None = None,
        available_policies: list[dict[str, Any]] | None = None,
        max_replans: int = 3,
        task_store: TaskStore | None = None,
        safety_monitor: RuntimeSafetyMonitor | None = None,
        task_verifier: TaskVerifier | None = None,
        control: ExecutionControl | None = None,
        initial_world_state: dict[str, Any] | None = None,
        router: ExecutorRouter | None = None,
        monitor_interval_seconds: float = 0.05,
        stop_grace_seconds: float = 1.0,
        local_recovery: LocalRecoveryHandler | None = None,
        pre_action_gate: PreActionGate | None = None,
        require_physical_verification: bool = False,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if max_replans < 0:
            raise ValueError("max_replans cannot be negative.")
        if monitor_interval_seconds <= 0:
            raise ValueError("monitor_interval_seconds must be positive.")
        if stop_grace_seconds < 0:
            raise ValueError("stop_grace_seconds cannot be negative.")
        self.planner = planner
        self.executors = {"robot": robot_executor, "policy": policy_executor}
        self.state_provider = state_provider
        self.verifier = verifier or PlaceholderActionVerifier()
        self.recovery_policy = recovery_policy or RecoveryPolicy()
        self.capabilities = capabilities or CapabilityRegistry()
        self.confidence_threshold = confidence_threshold
        self.available_skills = (
            list(available_skills)
            if available_skills is not None
            else [item["action_type"] for item in self.capabilities.planner_skills()]
        )
        self.available_policies = list(available_policies or [])
        self.max_replans = max_replans
        self.task_store = task_store or NullTaskStore()
        self.safety_monitor = safety_monitor or SoftwareSafetyMonitor()
        self.task_verifier = task_verifier or PlanCompletionTaskVerifier()
        self.control = control or ExecutionControl()
        self.initial_world_state = deepcopy(initial_world_state or {})
        self.router = router or ExecutorRouter(
            self.capabilities,
            self.executors,
        )
        self.monitor_interval_seconds = monitor_interval_seconds
        self.stop_grace_seconds = stop_grace_seconds
        self.local_recovery = local_recovery or NullLocalRecoveryHandler()
        self.pre_action_gate = pre_action_gate
        self.require_physical_verification = require_physical_verification
        self._active_memory: TaskMemory | None = None
        self._active_world: WorldState | None = None
        self._run_lock = threading.Lock()

    def pause(self) -> None:
        self.control.pause()
        if self._active_memory is not None and self._active_world is not None:
            self._active_memory.status = "paused"
            self._active_memory.event("task.paused")
            self._persist(self._active_memory, self._active_world)

    def resume_control(self) -> None:
        self.control.resume()
        if self._active_memory is not None and self._active_world is not None:
            self._active_memory.status = "running"
            self._active_memory.event("task.control_resumed")
            self._persist(self._active_memory, self._active_world)

    def cancel(self) -> None:
        self.control.cancel()
        for executor in self.executors.values():
            executor.stop()

    def run(
        self,
        task: str,
        *,
        task_id: str | None = None,
        resume: bool = False,
    ) -> TaskMemory:
        if not self._run_lock.acquire(blocking=False):
            raise AgentExecutionError(
                "This Agent instance is already running another task."
            )
        try:
            return self._run_once(task, task_id=task_id, resume=resume)
        finally:
            self._run_lock.release()

    def _run_once(
        self,
        task: str,
        *,
        task_id: str | None = None,
        resume: bool = False,
    ) -> TaskMemory:
        if not task.strip():
            raise ValueError("Task cannot be empty.")
        memory, world, plan_version = self._start_or_resume(task, task_id, resume)
        self._active_memory = memory
        self._active_world = world
        scheduler = self._scheduler_from_memory(memory)
        attempts_by_instance: dict[str, int] = dict(memory.attempts)

        try:
            while True:
                self.control.checkpoint()
                memory.status = self.control.status
                if memory.status == "paused":
                    self._persist(memory, world)
                    continue
                step = scheduler.next_step(world)
                if step is None:
                    break
                memory.plan = scheduler.plan
                step["status"] = "running"
                instance_id = str(step["instance_id"])
                attempts_by_instance[instance_id] = attempts_by_instance.get(instance_id, 0) + 1
                memory.attempts[instance_id] = attempts_by_instance[instance_id]
                memory.current_step_id = int(step["step_id"])
                action_id = uuid.uuid4().hex
                memory.event(
                    "step.started",
                    step_id=step["step_id"],
                    instance_id=instance_id,
                    attempt=attempts_by_instance[instance_id],
                    action_id=action_id,
                )

                configure_targets = getattr(self.state_provider, "configure_targets", None)
                if callable(configure_targets):
                    monitor_targets = [
                        str(step["target"]),
                        *[
                            str(value)
                            for value in step.get(
                                "_trusted_target_aliases", []
                            )
                        ],
                        *[
                            str(value)
                            for value in step.get("parameters", {}).get(
                                "monitor_targets", []
                            )
                        ],
                    ]
                    configure_targets(monitor_targets)
                before_observation = self.state_provider.observe()
                before_state = self.state_provider.robot_state()
                memory.inflight_action = {
                    "action_id": action_id,
                    "instance_id": instance_id,
                    "step_id": int(step["step_id"]),
                    "attempt": attempts_by_instance[instance_id],
                    "started_at": _now(),
                    "before_observation": summarize_observation(before_observation),
                    "before_robot_state": deepcopy(before_state),
                }
                self._persist(memory, world)
                safety = self.safety_monitor.before_action(step, before_state)
                if not safety.safe:
                    self._safety_stop(memory, world, safety.reason)

                executor: ActionExecutor | None = None
                routed_step = step
                precheck = self._monitor_before_action(
                    step,
                    before_observation,
                    before_state,
                )
                if precheck is not None:
                    action = {
                        "status": "failed",
                        "reason": str(
                            precheck.get("error_type") or "PRE_ACTION_CHECK_FAILED"
                        ),
                        "command_completed": False,
                        "physical_result_verified": False,
                        "details": {"blocked_before_execution": True},
                        "_monitor_verification": precheck,
                    }
                    memory.event(
                        "monitor.blocked",
                        step_id=step["step_id"],
                        error_type=action["reason"],
                    )
                else:
                    try:
                        route = self.router.route(step)
                        executor = route.executor
                        routed_step = route.step
                        memory.event(
                            "step.routed",
                            step_id=step["step_id"],
                            executor=route.executor_kind,
                            route=route.executor_name,
                            policy_id=route.policy_id,
                        )
                        action = self._execute_with_monitoring(
                            executor,
                            routed_step,
                            before_observation,
                            float(step["timeout_seconds"]),
                        )
                    except RoutingError as error:
                        action = {
                            "status": "failed",
                            "reason": error.error_type,
                            "command_completed": False,
                            "physical_result_verified": False,
                            "details": {"routing_error": str(error)},
                        }
                        memory.event(
                            "step.routing_failed",
                            step_id=step["step_id"],
                            error_type=error.error_type,
                        )

                after_observation = self.state_provider.observe()
                after_state = self.state_provider.robot_state()
                safety = self.safety_monitor.after_action(routed_step, action, after_state)
                if not safety.safe:
                    if executor is not None:
                        executor.stop()
                    action = {**action, "status": "failed", "reason": safety.reason}
                action["_monitor_step"] = dict(routed_step)
                verification = action.get("_monitor_verification")
                if not isinstance(verification, dict):
                    verification = self.verifier.verify(
                        after_observation,
                        after_state,
                        action,
                        str(step["expected_result"]),
                    )
                if (
                    self.require_physical_verification
                    and bool(verification.get("success"))
                    and not _has_physical_verification(verification, action)
                ):
                    verification = {
                        "success": False,
                        "error_type": "VERIFICATION_UNCERTAIN",
                        "confidence": 1.0,
                        "verification_scope": "command",
                        "details": {
                            "required_scope": "physical",
                            "reported_scope": verification.get(
                                "verification_scope", "unknown"
                            ),
                            "expected_result": step.get("expected_result"),
                            "reported_details": deepcopy(
                                verification.get("details") or {}
                            ),
                        },
                    }

                verification_confidence = float(verification.get("confidence", 0.0))
                verified = bool(verification.get("success")) and (
                    verification_confidence >= self.confidence_threshold
                )
                if (
                    bool(verification.get("success"))
                    and verification_confidence < self.confidence_threshold
                ):
                    verification = {
                        **verification,
                        "success": False,
                        "error_type": "LOW_CONFIDENCE",
                        "details": {
                            **dict(verification.get("details") or {}),
                            "required_confidence": self.confidence_threshold,
                            "reported_confidence": verification_confidence,
                        },
                    }
                if verified:
                    scheduler.complete(step, world)
                    memory.inflight_action = None
                    memory.save(step, "completed", after_observation, after_state, verification)
                    memory.event("step.completed", step_id=step["step_id"])
                    self._persist(memory, world)
                    continue

                if executor is not None:
                    executor.stop()
                step["status"] = "failed"
                memory.inflight_action = None
                memory.save(step, "failed", after_observation, after_state, verification)
                error_type = str(verification.get("error_type", "VERIFICATION_FAILED"))
                memory.event("step.failed", step_id=step["step_id"], error_type=error_type)
                self._persist(memory, world)

                attempts = attempts_by_instance[instance_id]
                max_attempts = int(step.get("max_attempts", 2))
                decision = self.recovery_policy.decide(error_type, attempts)
                if (
                    decision.action in {"retry", "reobserve"}
                    and attempts >= max_attempts
                ):
                    decision = RecoveryDecision(
                        "replan", "Step-specific attempt limit reached."
                    )
                memory.event("recovery.decided", action=decision.action, reason=decision.reason)
                if decision.action == "stop":
                    self._safety_stop(memory, world, decision.reason)
                if decision.action in {"retry", "reobserve"}:
                    recovery_result = self.local_recovery.recover(
                        decision.action,
                        error_type,
                        step,
                    )
                    memory.event(
                        "recovery.executed",
                        action=decision.action,
                        success=bool(recovery_result.get("success")),
                        performed=list(recovery_result.get("performed") or []),
                        reason=recovery_result.get("reason"),
                    )
                    if recovery_result.get("success"):
                        step["status"] = "pending"
                        self._persist(memory, world)
                        continue
                    decision = RecoveryDecision(
                        "replan",
                        (
                            "Local recovery could not restore a safe retry state: "
                            + str(
                                recovery_result.get("reason")
                                or "LOCAL_RECOVERY_FAILED"
                            )
                        ),
                    )
                    memory.event(
                        "recovery.escalated",
                        action="replan",
                        reason=decision.reason,
                    )
                if memory.replan_count >= self.max_replans:
                    memory.status = "failed"
                    memory.failure_reason = "Maximum VLM replanning attempts reached."
                    self._persist(memory, world)
                    raise AgentExecutionError(memory.failure_reason, memory)

                context = self._failure_context(
                    task,
                    memory,
                    world,
                    step,
                    action,
                    verification,
                    before_observation,
                    before_state,
                    after_observation,
                    after_state,
                    attempts,
                    decision,
                )
                plan_version += 1
                raw_replacement = self.planner.revise_from_failure(context)
                self._synchronize_registered_entities(world, memory)
                persisted_goal = world.get("_agent.task_goal")
                candidate_goal = getattr(self.planner, "current_goal", None)
                if not isinstance(candidate_goal, dict):
                    candidate_goal = getattr(self.planner, "last_goal", None)
                if (
                    isinstance(persisted_goal, dict)
                    and isinstance(candidate_goal, dict)
                    and _goal_conditions_signature(persisted_goal)
                    != _goal_conditions_signature(candidate_goal)
                ):
                    raise AgentExecutionError(
                        "Replanner changed the original task goal.", memory
                    )
                replacement = self._prepare_plan(raw_replacement, plan_version)
                scheduler.replace_unfinished(replacement)
                memory.plan = replacement
                memory.replan_count += 1
                memory.event("plan.replaced", plan_version=plan_version)
                memory.event(
                    "plan.validated",
                    plan_version=plan_version,
                    step_count=len(replacement),
                    skills=[
                        str(replacement_step["action_type"])
                        for replacement_step in replacement
                    ],
                )
                self._persist(memory, world)

            task_verification = self.task_verifier.verify(
                task, memory.completed_steps, world
            )
            memory.task_verification = asdict(task_verification)
            if not task_verification.success:
                memory.status = "failed"
                memory.failure_reason = task_verification.reason
                memory.event("task.verification_failed", reason=task_verification.reason)
                self._persist(memory, world)
                raise AgentExecutionError(task_verification.reason, memory)
            memory.status = "completed"
            memory.event(
                "task.completed",
                verification_scope=task_verification.verification_scope,
            )
            self._persist(memory, world)
            self._active_memory = None
            self._active_world = None
            return memory
        except TaskCancelled as error:
            for executor in self.executors.values():
                executor.stop()
            memory.status = "cancelled"
            memory.failure_reason = str(error)
            memory.event("task.cancelled")
            self._persist(memory, world)
            raise AgentExecutionError(str(error), memory) from error
        except SchedulingError as error:
            memory.status = "failed"
            memory.failure_reason = str(error)
            memory.event("task.scheduling_failed", reason=str(error))
            self._persist(memory, world)
            raise AgentExecutionError(str(error), memory) from error
        except (SafetyStop, AgentExecutionError):
            raise
        except Exception as error:
            for executor in self.executors.values():
                executor.stop()
            memory.status = "failed"
            memory.failure_reason = str(error)
            memory.event(
                "task.failed",
                error_type=type(error).__name__,
                reason=str(error),
            )
            self._persist(memory, world)
            raise AgentExecutionError(str(error), memory) from error

    def run_safe(
        self,
        task: str,
        *,
        task_id: str | None = None,
        resume: bool = False,
    ) -> AgentRunResult:
        try:
            memory = self.run(task, task_id=task_id, resume=resume)
            return AgentRunResult("completed", memory)
        except SafetyStop as error:
            memory = error.memory or TaskMemory(
                original_task=task,
                status="safety_stopped",
                failure_reason=str(error),
            )
            return AgentRunResult("safety_stopped", memory, str(error))
        except AgentExecutionError as error:
            memory = error.memory or TaskMemory(
                original_task=task, status="failed", failure_reason=str(error)
            )
            status = "cancelled" if memory.status == "cancelled" else "failed"
            return AgentRunResult(status, memory, str(error))
        except Exception as error:
            memory = TaskMemory(
                original_task=task, status="failed", failure_reason=str(error)
            )
            return AgentRunResult("failed", memory, str(error))
        finally:
            self._active_memory = None
            self._active_world = None

    def _monitor_before_action(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> VerificationResult | None:
        if self.pre_action_gate is not None:
            result = self.pre_action_gate.before_action(
                step,
                observation,
                robot_state,
            )
            if isinstance(result, dict) and not bool(result.get("success", False)):
                return result
        check = getattr(self.verifier, "before_action", None)
        if not callable(check):
            return None
        result = check(step, observation, robot_state)
        if isinstance(result, dict) and not bool(result.get("success", False)):
            return result
        return None

    def _monitor_during_action(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> VerificationResult | None:
        if self.pre_action_gate is not None:
            check = getattr(self.pre_action_gate, "during_action", None)
            if callable(check):
                result = check(step, observation, robot_state)
                if isinstance(result, dict) and not bool(result.get("success", False)):
                    return result
        safety_check = getattr(self.safety_monitor, "during_action", None)
        if callable(safety_check):
            safety = safety_check(step, observation, robot_state)
            if not safety.safe:
                return {
                    "success": False,
                    "error_type": safety.reason,
                    "confidence": 1.0,
                    "verification_scope": "physical",
                    "details": {"phase": "during_action", "source": "safety_monitor"},
                }
        check = getattr(self.verifier, "during_action", None)
        if callable(check):
            result = check(step, observation, robot_state)
            if isinstance(result, dict) and not bool(result.get("success", False)):
                return result
        return None

    def _execute_with_monitoring(
        self,
        executor: ActionExecutor,
        step: PlanStep,
        before_observation: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Run the executor and the observation Monitor concurrently.

        Executors remain synchronous at their public boundary, so this wrapper
        runs one in a worker thread and polls the perception buffer in the task
        thread.  A real backend must make ``stop`` cooperative; otherwise the
        result is a safety-critical ACTION_STOP_UNCONFIRMED failure.
        """

        done = threading.Event()
        result_box: dict[str, Any] = {}

        def execute_worker() -> None:
            try:
                result_box["action"] = executor.execute(step, before_observation)
            except BaseException as error:  # propagate process-level interruptions
                result_box["exception"] = error
            finally:
                done.set()

        started = time.monotonic()
        worker = threading.Thread(
            target=execute_worker,
            name=f"robot-action-{step.get('instance_id', step.get('step_id'))}",
            daemon=True,
        )
        worker.start()
        samples = 0
        deadline = started + timeout_seconds

        while not done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._stop_running_action(
                    executor,
                    done,
                    "ACTION_TIMEOUT",
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "monitor_samples": samples,
                    },
                )
            if done.wait(min(self.monitor_interval_seconds, remaining)):
                break
            if self.control.status == "cancelled":
                executor.stop()
                done.wait(self.stop_grace_seconds)
                raise TaskCancelled("Task was cancelled during an active action.")

            try:
                observation = self.state_provider.observe()
            except Exception as error:
                return self._stop_running_action(
                    executor,
                    done,
                    "PERCEPTION_FAILED",
                    {
                        "monitor_samples": samples,
                        "exception": repr(error),
                    },
                )
            try:
                robot_state = self.state_provider.robot_state()
            except Exception as error:
                return self._stop_running_action(
                    executor,
                    done,
                    "ROBOT_STATE_FAILED",
                    {
                        "monitor_samples": samples,
                        "exception": repr(error),
                    },
                )
            samples += 1
            monitor_result = self._monitor_during_action(
                step,
                observation,
                robot_state,
            )
            if monitor_result is not None:
                error_type = str(
                    monitor_result.get("error_type") or "MONITOR_REJECTED_ACTION"
                )
                return self._stop_running_action(
                    executor,
                    done,
                    error_type,
                    {
                        "monitor_samples": samples,
                        "observation": summarize_observation(observation),
                    },
                    monitor_verification=monitor_result,
                )

        elapsed = time.monotonic() - started
        if "exception" in result_box:
            if not isinstance(result_box["exception"], Exception):
                raise result_box["exception"]
            return {
                "status": "failed",
                "reason": "EXECUTOR_EXCEPTION",
                "command_completed": False,
                "physical_result_verified": False,
                "details": {
                    "exception": repr(result_box["exception"]),
                    "elapsed_seconds": elapsed,
                    "monitor_samples": samples,
                },
            }
        action = result_box.get("action")
        if not isinstance(action, dict):
            return {
                "status": "failed",
                "reason": "INVALID_EXECUTOR_RESULT",
                "command_completed": False,
                "physical_result_verified": False,
                "details": {
                    "elapsed_seconds": elapsed,
                    "monitor_samples": samples,
                },
            }
        details = dict(action.get("details") or {})
        details.update(
            {
                "elapsed_seconds": elapsed,
                "monitor_samples": samples,
            }
        )
        return {**action, "details": details}

    def _stop_running_action(
        self,
        executor: ActionExecutor,
        done: threading.Event,
        reason: str,
        details: dict[str, Any],
        *,
        monitor_verification: VerificationResult | None = None,
    ) -> dict[str, Any]:
        executor.stop()
        stopped = done.wait(self.stop_grace_seconds)
        final_reason = reason if stopped else "ACTION_STOP_UNCONFIRMED"
        result: dict[str, Any] = {
            "status": "failed",
            "reason": final_reason,
            "command_completed": False,
            "physical_result_verified": False,
            "details": {
                **details,
                "stop_confirmed": stopped,
                "trigger_reason": reason,
            },
        }
        if monitor_verification is not None:
            result["_monitor_verification"] = (
                monitor_verification
                if stopped
                else {
                    "success": False,
                    "error_type": "ACTION_STOP_UNCONFIRMED",
                    "confidence": 1.0,
                    "verification_scope": "command",
                    "details": {
                        "trigger_reason": reason,
                        "monitor_verification": monitor_verification,
                    },
                }
            )
        return result

    def _start_or_resume(
        self, task: str, task_id: str | None, resume: bool
    ) -> tuple[TaskMemory, WorldState, int]:
        if resume:
            if not task_id:
                raise ValueError("task_id is required when resume=True.")
            snapshot = self.task_store.load(task_id)
            if snapshot is None:
                raise ValueError(f"No persisted task found for task_id '{task_id}'.")
            memory = TaskMemory.from_snapshot(snapshot)
            if memory.original_task != task:
                raise ValueError("Persisted task does not match the requested task.")
            world = WorldState(
                values=dict(memory.world_state.get("values", {})),
                version=int(memory.world_state.get("version", 0)),
            )
            memory.status = "running"
            memory.event("task.resumed")
            if memory.inflight_action is not None:
                memory.event(
                    "task.inflight_reconciliation_required",
                    action_id=memory.inflight_action.get("action_id"),
                    step_id=memory.inflight_action.get("step_id"),
                    policy="reobserve_before_retry",
                )
            plan_version = memory.replan_count + 1
            return memory, world, plan_version

        world = WorldState(values=deepcopy(self.initial_world_state))
        memory = TaskMemory(original_task=task, task_id=task_id or uuid.uuid4().hex)
        context = {
            "original_task": task,
            "world_state": world.snapshot(),
            "capabilities": self.capabilities.planner_skills(),
        }
        create_with_context = getattr(self.planner, "create_plan_with_context", None)
        try:
            raw_plan = (
                create_with_context(context)
                if callable(create_with_context)
                else self.planner.create_plan(task)
            )
            memory.plan = self._prepare_plan(raw_plan, 1)
        except Exception as error:
            memory.status = "failed"
            memory.failure_reason = str(error)
            memory.event(
                "plan.rejected",
                error_type=type(error).__name__,
                reason=str(error),
            )
            self._persist(memory, world)
            raise AgentExecutionError(str(error), memory) from error
        task_goal = getattr(self.planner, "current_goal", None)
        if not isinstance(task_goal, dict):
            task_goal = getattr(self.planner, "last_goal", None)
        if isinstance(task_goal, dict):
            world.set("_agent.task_goal", task_goal)
        memory.event("task.created")
        self._synchronize_registered_entities(world, memory)
        memory.event(
            "plan.validated",
            step_count=len(memory.plan),
            skills=[str(step["action_type"]) for step in memory.plan],
        )
        self._persist(memory, world)
        return memory, world, 1

    def _synchronize_registered_entities(
        self,
        world: WorldState,
        memory: TaskMemory,
    ) -> None:
        """Merge only trusted entity declarations emitted by a grounding wrapper.

        The hook intentionally accepts an entity-only patch.  It never copies
        the grounder's simulated end state, because doing so would mark actions
        complete before an executor has actually run.
        """

        consume = getattr(self.planner, "consume_world_patch", None)
        entity_only = False
        if not callable(consume):
            consume = getattr(self.planner, "consume_registered_entities", None)
            entity_only = True
        if not callable(consume):
            return
        raw_patch = consume()
        if not isinstance(raw_patch, dict):
            raise AgentExecutionError(
                "Planner returned an invalid registered-entity patch.",
                memory,
            )
        raw_entities = raw_patch if entity_only else raw_patch.get("objects", {})
        if not isinstance(raw_entities, dict):
            raise AgentExecutionError(
                "Planner returned an invalid registered-entity patch.",
                memory,
            )
        registered: list[str] = []
        for entity_id, raw_entity in raw_entities.items():
            identifier = str(entity_id).strip()
            if (
                not identifier
                or "." in identifier
                or "/" in identifier
                or "\\" in identifier
                or not isinstance(raw_entity, dict)
                or raw_entity.get("virtual") is not True
            ):
                raise AgentExecutionError(
                    "Planner returned an unsafe virtual-entity declaration.",
                    memory,
                )
            existing = world.get(f"objects.{identifier}")
            if existing is not None and existing != raw_entity:
                raise AgentExecutionError(
                    f"Virtual entity '{identifier}' conflicts with world state.",
                    memory,
                )
            if existing is None:
                world.set(f"objects.{identifier}", raw_entity)
                registered.append(identifier)
        if registered:
            memory.event(
                "world.virtual_entities_registered",
                entity_ids=registered,
                source="qwen_open_world_simulation",
            )

    @staticmethod
    def _scheduler_from_memory(memory: TaskMemory) -> TaskGraphScheduler:
        completed = {
            int(step["step_id"]) for step in memory.plan
            if step.get("status") == "completed"
        }
        skipped = {
            int(step["step_id"]) for step in memory.plan
            if step.get("status") == "skipped"
        }
        for step in memory.plan:
            if step.get("status") in {"running", "failed"}:
                step["status"] = "pending"
        return TaskGraphScheduler(memory.plan, completed=completed, skipped=skipped)

    def _prepare_plan(self, plan: list[PlanStep], plan_version: int) -> list[PlanStep]:
        if not plan:
            return []
        normalized = self.capabilities.normalize_plan(plan)
        for index, step in enumerate(normalized, start=1):
            step["instance_id"] = f"plan-{plan_version}-step-{step['step_id']}-{index}"
        return normalized

    def _persist(self, memory: TaskMemory, world: WorldState) -> None:
        memory.world_state = world.snapshot()
        memory.updated_at = _now()
        self.task_store.save(memory.task_id, memory.snapshot())

    def _safety_stop(
        self, memory: TaskMemory, world: WorldState, reason: str
    ) -> None:
        for executor in self.executors.values():
            executor.stop()
        memory.status = "safety_stopped"
        memory.failure_reason = reason
        memory.event("task.safety_stopped", reason=reason)
        self._persist(memory, world)
        raise SafetyStop(reason, memory)

    def _failure_context(
        self,
        task: str,
        memory: TaskMemory,
        world: WorldState,
        step: PlanStep,
        action: dict[str, Any],
        verification: VerificationResult,
        before_observation: dict[str, Any],
        before_state: dict[str, Any],
        after_observation: dict[str, Any],
        after_state: dict[str, Any],
        attempts: int,
        decision: RecoveryDecision,
    ) -> dict[str, Any]:
        current_summary = summarize_observation(after_observation)
        before_summary = summarize_observation(before_observation)
        error_type = str(
            verification.get("error_type", "VERIFICATION_FAILED")
        )
        return {
            "original_task": task,
            "completed_steps": memory.completed_steps,
            "current_step": step,
            "failed_skill": step.get("action_type"),
            "failed_target": step.get("target"),
            "failed_action": _compact_action(action),
            "error_type": error_type,
            "reason": error_type,
            "confidence": float(verification.get("confidence", 0.0)),
            "current_observation": current_summary,
            "target_visible": _target_visible(current_summary, str(step.get("target") or "")),
            "robot_state": after_state,
            "pre_action_observation": before_summary,
            "pre_action_robot_state": before_state,
            "world_state": world.snapshot(),
            "available_skills": self.available_skills,
            "capabilities": self.capabilities.planner_skills(),
            "local_attempts": attempts,
            "retry_count": max(0, attempts - 1),
            "recovery_reason": decision.reason,
        }

    @staticmethod
    def _normalize_plan(plan: list[PlanStep]) -> list[PlanStep]:
        return CapabilityRegistry().normalize_plan(plan)


def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    compact = summarize_observation(observation)
    for key, value in observation.items():
        if key not in {"images", "frames", "detections", "signals"}:
            compact.setdefault(key, value)
    if "images" in observation:
        compact["image_references"] = {
            name: value
            if isinstance(value, str) and not value.startswith(("data:", "/9j/"))
            else f"<embedded:{len(value)}>"
            for name, value in dict(observation.get("images") or {}).items()
        }
    if "frames" in observation:
        compact["frame_count"] = len(observation.get("frames") or [])
    return compact


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in action.items()
        if not key.startswith("_") and key not in {"images", "frames"}
    }


def _target_visible(summary: dict[str, Any], target: str) -> bool | None:
    signals = dict(summary.get("signals") or {})
    if "target_visible" in signals:
        return bool(signals["target_visible"])
    detections = list(summary.get("detections") or [])
    if not summary.get("available"):
        return None
    canonical_target = " ".join(target.casefold().replace("_", " ").split())
    return any(
        " ".join(
            str(item.get("label") or "").casefold().replace("_", " ").split()
        )
        == canonical_target
        for item in detections
        if isinstance(item, dict)
    )


def _goal_conditions_signature(goal: dict[str, Any]) -> tuple[str, ...]:
    conditions = goal.get("conditions")
    if not isinstance(conditions, list):
        return ()
    import json

    return tuple(
        sorted(
            json.dumps(condition, ensure_ascii=False, sort_keys=True, default=str)
            for condition in conditions
            if isinstance(condition, dict)
        )
    )


def _has_physical_verification(
    verification: dict[str, Any],
    action: dict[str, Any],
) -> bool:
    if str(verification.get("verification_scope") or "").casefold() == "physical":
        return True
    details = dict(verification.get("details") or {})
    return bool(
        verification.get("physical_result_verified")
        or details.get("physical_result_verified")
        or action.get("physical_result_verified")
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
