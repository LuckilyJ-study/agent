from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .action_executors import PolicyRegistry
from .capabilities import CapabilityRegistry
from .state import PlanStep


class RoutingError(RuntimeError):
    """A validated skill has no safe, compatible execution backend."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class RouteDecision:
    executor_kind: str
    executor_name: str
    executor: Any
    step: PlanStep
    policy_id: str | None = None


class ExecutorRouter:
    """Select an executor from trusted local registries.

    Planner output is treated as a request for a semantic skill.  It cannot
    choose a policy model or redirect a policy skill to the primitive robot
    controller.
    """

    def __init__(
        self,
        capabilities: CapabilityRegistry,
        executors: Mapping[str, Any],
        policies: PolicyRegistry | None = None,
        *,
        require_policy_registry: bool = False,
    ) -> None:
        self.capabilities = capabilities
        self.executors = dict(executors)
        self.policies = policies
        self.require_policy_registry = require_policy_registry

    def route(self, step: PlanStep) -> RouteDecision:
        action_type = str(step.get("action_type") or "")
        capability = self.capabilities.get(action_type)
        supplied_executor = step.get("executor")
        if supplied_executor is not None and supplied_executor != capability.executor:
            raise RoutingError(
                "EXECUTOR_ROUTE_MISMATCH",
                (
                    f"Skill '{action_type}' is registered for "
                    f"executor='{capability.executor}', not '{supplied_executor}'."
                ),
            )
        executor = self.executors.get(capability.executor)
        if executor is None:
            raise RoutingError(
                "EXECUTOR_NOT_AVAILABLE",
                f"Executor '{capability.executor}' is not configured.",
            )

        routed_step: PlanStep = dict(step)
        routed_step["executor"] = capability.executor
        if capability.executor == "robot":
            return RouteDecision(
                executor_kind="robot",
                executor_name="robot_control",
                executor=executor,
                step=routed_step,
            )

        if self.policies is None:
            if self.require_policy_registry:
                raise RoutingError(
                    "POLICY_NOT_AVAILABLE",
                    f"No policy registry is configured for skill '{action_type}'.",
                )
            policy_id = str(step.get("policy_id") or capability.policy_id or "") or None
            if policy_id is not None:
                routed_step["policy_id"] = policy_id
            return RouteDecision(
                executor_kind="policy",
                executor_name=f"policy:{policy_id or 'configured_executor'}",
                executor=executor,
                step=routed_step,
                policy_id=policy_id,
            )
        policy_id = self._select_policy_id(step, action_type, capability.policy_id)
        metadata = self.policies.metadata(policy_id)
        if metadata is None or self.policies.get(policy_id) is None:
            raise RoutingError(
                "POLICY_NOT_AVAILABLE",
                f"Policy '{policy_id}' is not registered.",
            )
        if metadata.action_type not in {action_type, "*"}:
            raise RoutingError(
                "POLICY_ACTION_MISMATCH",
                (
                    f"Policy '{policy_id}' handles '{metadata.action_type}', "
                    f"not '{action_type}'."
                ),
            )
        if capability.supports_stop and not metadata.supports_stop:
            raise RoutingError(
                "POLICY_STOP_UNSUPPORTED",
                (
                    f"Policy '{policy_id}' cannot be safely stopped, but skill "
                    f"'{action_type}' requires preemption."
                ),
            )
        routed_step["policy_id"] = policy_id
        return RouteDecision(
            executor_kind="policy",
            executor_name=f"policy:{policy_id}",
            executor=executor,
            step=routed_step,
            policy_id=policy_id,
        )

    def _select_policy_id(
        self,
        step: PlanStep,
        action_type: str,
        default_policy_id: str | None,
    ) -> str:
        # ``policy_id`` is supported for trusted, hand-authored integration
        # plans. It is intentionally excluded from the Qwen response schema.
        requested = str(step.get("policy_id") or "").strip()
        if requested:
            return requested
        if (
            default_policy_id
            and self.policies is not None
            and self.policies.get(default_policy_id) is not None
        ):
            return default_policy_id
        candidates = (
            self.policies.find_for_action(action_type)
            if self.policies is not None
            else []
        )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise RoutingError(
                "POLICY_NOT_AVAILABLE",
                f"No registered policy can execute skill '{action_type}'.",
            )
        raise RoutingError(
            "POLICY_ROUTE_AMBIGUOUS",
            (
                f"Multiple policies can execute '{action_type}': {candidates}. "
                "Choose one in the trusted capability configuration."
            ),
        )
