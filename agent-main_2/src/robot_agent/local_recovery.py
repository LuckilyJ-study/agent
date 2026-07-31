from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .monitor import summarize_observation
from .state import PlanStep


class LocalRecoveryHandler(Protocol):
    def recover(
        self,
        action: str,
        error_type: str,
        step: PlanStep,
    ) -> dict[str, Any]: ...


@dataclass
class NullLocalRecoveryHandler:
    """Explicit placeholder when a deployment has no safe recovery motion."""

    def recover(
        self,
        action: str,
        error_type: str,
        step: PlanStep,
    ) -> dict[str, Any]:
        return {
            "success": True,
            "performed": [],
            "note": "No local recovery handler is configured.",
        }


@dataclass
class ControllerLocalRecoveryHandler:
    """Basic stop/retreat/reobserve sequence for recoverable failures."""

    controller: Any
    state_provider: Any
    retreat_on_errors: tuple[str, ...] = ("GRASP_FAILED", "PLACE_FAILED")

    def recover(
        self,
        action: str,
        error_type: str,
        step: PlanStep,
    ) -> dict[str, Any]:
        performed: list[str] = []
        try:
            stop = getattr(self.controller, "stop", None)
            if callable(stop):
                stop()
                performed.append("stop")
            if action == "retry" and error_type in self.retreat_on_errors:
                move_home = getattr(self.controller, "move_home", None)
                if not callable(move_home):
                    return {
                        "success": False,
                        "performed": performed,
                        "reason": "SAFE_RETREAT_UNAVAILABLE",
                    }
                result = move_home()
                performed.append("move_home")
                if isinstance(result, dict) and result.get("status") == "failed":
                    return {
                        "success": False,
                        "performed": performed,
                        "reason": str(result.get("reason") or "SAFE_RETREAT_FAILED"),
                    }
            configure = getattr(self.state_provider, "configure_targets", None)
            if callable(configure):
                configure([str(step.get("target") or "")])
            observation = self.state_provider.observe()
            performed.append("reobserve")
            return {
                "success": True,
                "performed": performed,
                "observation": summarize_observation(observation),
            }
        except Exception as error:
            return {
                "success": False,
                "performed": performed,
                "reason": "LOCAL_RECOVERY_EXCEPTION",
                "details": {"exception": repr(error)},
            }
