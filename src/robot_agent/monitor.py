from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import PlanStep, VerificationResult


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float] | None = None
    track_id: str | int | None = None
    following_gripper: bool | None = None
    falling: bool | None = None


class StructuredActionMonitor:
    """Fast rule monitor for YOLO/tracker-shaped structured observations.

    This class deliberately does not load YOLO-World.  A future high-rate
    perception provider can maintain a rolling ``frames`` buffer and expose
    detections with the fields consumed here.  Tests and simulation can supply
    the same contract without a camera.
    """

    # ``inspect`` is deliberately excluded: it is the recovery skill used to
    # search for a target that is not currently visible.
    TARGET_GATED_ACTIONS = {"pick", "place", "manipulate", "move_to"}

    def __init__(
        self,
        *,
        target_lost_frames: int = 3,
        minimum_detection_confidence: float = 0.25,
        success_confidence: float = 0.95,
    ) -> None:
        if target_lost_frames < 1:
            raise ValueError("target_lost_frames must be at least 1.")
        if not 0 <= minimum_detection_confidence <= 1:
            raise ValueError("minimum_detection_confidence must be between 0 and 1.")
        self.target_lost_frames = target_lost_frames
        self.minimum_detection_confidence = minimum_detection_confidence
        self.success_confidence = success_confidence

    def before_action(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> VerificationResult | None:
        if not observation.get("available", False):
            return None
        critical = self._critical_signal(step, observation)
        if critical is not None:
            return critical
        if (
            str(step.get("action_type")) in self.TARGET_GATED_ACTIONS
            and self._target_is_lost(step, observation)
        ):
            return self._failure(
                "TARGET_LOST",
                0.98,
                step,
                observation,
                phase="before_action",
            )
        return None

    def during_action(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> VerificationResult | None:
        if not observation.get("available", False):
            return None
        critical = self._critical_signal(step, observation)
        if critical is not None:
            return critical
        if self._object_dropped(step, observation):
            return self._failure(
                "OBJECT_DROPPED",
                0.98,
                step,
                observation,
                phase="during_action",
            )
        if (
            str(step.get("action_type")) in self.TARGET_GATED_ACTIONS
            and self._target_is_lost(step, observation)
        ):
            return self._failure(
                "TARGET_LOST",
                0.95,
                step,
                observation,
                phase="during_action",
            )
        return None

    def verify(
        self,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
        action: dict[str, Any],
        expected_result: str,
    ) -> VerificationResult:
        step = dict(action.get("_monitor_step") or {})
        if action.get("status") != "success":
            return {
                "success": False,
                "error_type": str(action.get("reason") or "EXECUTION_FAILED"),
                "confidence": 1.0,
                "verification_scope": "command",
                "details": {
                    "expected_result": expected_result,
                    "observation_summary": summarize_observation(observation),
                },
            }
        if not observation.get("available", False):
            return {
                "success": True,
                "error_type": "NONE",
                "confidence": 1.0,
                "verification_scope": "command",
                "details": {
                    "expected_result": expected_result,
                    "physical_result_verified": False,
                    "perception_available": False,
                },
            }

        critical = self._critical_signal(step, observation)
        if critical is not None:
            return critical
        if self._object_dropped(step, observation):
            return self._failure(
                "OBJECT_DROPPED", 0.98, step, observation, phase="after_action"
            )
        if step and self._target_is_lost(step, observation):
            return self._failure(
                "TARGET_LOST", 0.95, step, observation, phase="after_action"
            )

        signals = _latest_signals(observation)
        action_type = str(step.get("action_type") or "")
        if action_type == "inspect":
            return self._success(
                expected_result,
                observation,
                "inspection_observation_received",
            )
        if action_type == "pick":
            following = self._latest_target_detection(step, observation)
            if following is not None and following.following_gripper is True:
                return self._success(expected_result, observation, "target_follows_gripper")
            explicit_grasp = signals.get("grasp_succeeded")
            if explicit_grasp is True:
                return self._success(expected_result, observation, "grasp_succeeded")
            gripper = str(robot_state.get("gripper") or "").lower()
            lifted = bool(
                robot_state.get("lifting")
                or signals.get("gripper_lifted")
                or signals.get("lift_complete")
            )
            if explicit_grasp is False or (gripper == "closed" and lifted):
                return self._failure(
                    "GRASP_FAILED", 0.95, step, observation, phase="after_action"
                )
        if action_type == "place" and "placement_complete" in signals:
            if bool(signals["placement_complete"]):
                return self._success(expected_result, observation, "placement_complete")
            return self._failure(
                "PLACE_FAILED", 0.95, step, observation, phase="after_action"
            )
        if "expected_result_met" in signals:
            if bool(signals["expected_result_met"]):
                return self._success(expected_result, observation, "expected_result_met")
            return self._failure(
                "EXPECTED_RESULT_NOT_MET",
                0.9,
                step,
                observation,
                phase="after_action",
            )

        # Perception was present but did not expose enough task semantics. Do
        # not pretend that a physical result was proven.
        return {
            "success": True,
            "error_type": "NONE",
            "confidence": 1.0,
            "verification_scope": "command",
            "details": {
                "expected_result": expected_result,
                "physical_result_verified": False,
                "perception_available": True,
                "monitor_inconclusive": True,
                "observation_summary": summarize_observation(observation),
            },
        }

    def _critical_signal(
        self, step: PlanStep, observation: dict[str, Any]
    ) -> VerificationResult | None:
        signals = _latest_signals(observation)
        if signals.get("collision_risk"):
            return self._failure(
                "COLLISION_RISK", 1.0, step, observation, phase="monitor"
            )
        if signals.get("hardware_fault"):
            return self._failure(
                "HARDWARE_FAULT", 1.0, step, observation, phase="monitor"
            )
        return None

    def _target_is_lost(
        self, step: PlanStep, observation: dict[str, Any]
    ) -> bool:
        frames = _frames(observation)
        if len(frames) < self.target_lost_frames:
            return False
        recent = frames[-self.target_lost_frames :]
        return all(not self._target_visible_in_frame(step, frame) for frame in recent)

    def _target_visible_in_frame(
        self, step: PlanStep, frame: dict[str, Any]
    ) -> bool:
        signals = dict(frame.get("signals") or {})
        if "target_visible" in signals:
            return bool(signals["target_visible"])
        targets = _target_names(step)
        if not targets:
            return False
        return any(
            (
                _canonical(
                    str(
                        item.get("entity_id")
                        or item.get("label")
                        or item.get("name")
                        or ""
                    )
                )
                in targets
            )
            and float(item.get("confidence", 0.0)) >= self.minimum_detection_confidence
            for item in frame.get("detections", [])
            if isinstance(item, dict)
        )

    def _latest_target_detection(
        self, step: PlanStep, observation: dict[str, Any]
    ) -> Detection | None:
        targets = _target_names(step)
        for frame in reversed(_frames(observation)):
            for raw in frame.get("detections", []):
                if not isinstance(raw, dict):
                    continue
                label = str(raw.get("label") or raw.get("name") or "")
                confidence = float(raw.get("confidence", 0.0))
                identity = _canonical(str(raw.get("entity_id") or label))
                if identity not in targets or confidence < self.minimum_detection_confidence:
                    continue
                bbox = raw.get("bbox_xyxy")
                return Detection(
                    label=label,
                    confidence=confidence,
                    bbox_xyxy=(
                        tuple(float(value) for value in bbox)
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
                        else None
                    ),
                    track_id=raw.get("track_id"),
                    following_gripper=_optional_bool(raw.get("following_gripper")),
                    falling=_optional_bool(raw.get("falling")),
                )
        return None

    def _object_dropped(
        self, step: PlanStep, observation: dict[str, Any]
    ) -> bool:
        if str(step.get("action_type") or "") != "pick":
            return bool(_latest_signals(observation).get("object_dropped"))
        targets = _target_names(step)
        following_seen = False
        for frame in _frames(observation):
            for raw in frame.get("detections", []):
                if (
                    isinstance(raw, dict)
                    and _canonical(
                        str(
                            raw.get("entity_id")
                            or raw.get("label")
                            or raw.get("name")
                            or ""
                        )
                    )
                    in targets
                ):
                    if raw.get("following_gripper") is True:
                        following_seen = True
                    elif following_seen and (
                        raw.get("falling") is True
                        or dict(frame.get("signals") or {}).get("object_dropped")
                    ):
                        return True
        return bool(_latest_signals(observation).get("object_dropped"))

    def _success(
        self,
        expected_result: str,
        observation: dict[str, Any],
        evidence: str,
    ) -> VerificationResult:
        return {
            "success": True,
            "error_type": "NONE",
            "confidence": self.success_confidence,
            "verification_scope": "physical",
            "details": {
                "expected_result": expected_result,
                "evidence": evidence,
                "observation_summary": summarize_observation(observation),
            },
        }

    @staticmethod
    def _failure(
        error_type: str,
        confidence: float,
        step: PlanStep,
        observation: dict[str, Any],
        *,
        phase: str,
    ) -> VerificationResult:
        return {
            "success": False,
            "error_type": error_type,
            "confidence": confidence,
            "verification_scope": "physical",
            "details": {
                "phase": phase,
                "failed_skill": step.get("action_type"),
                "target": step.get("target"),
                "observation_summary": summarize_observation(observation),
            },
        }


def summarize_observation(observation: dict[str, Any] | None) -> dict[str, Any]:
    """Return a compact structure safe to store or send to a replanner."""

    if not observation:
        return {"available": False}
    frames = _frames(observation)
    latest = frames[-1] if frames else {}
    detections: list[dict[str, Any]] = []
    for raw in latest.get("detections", []):
        if not isinstance(raw, dict):
            continue
        item = {
            "label": raw.get("label") or raw.get("name"),
            "confidence": raw.get("confidence"),
        }
        for key in (
            "bbox_xyxy",
            "track_id",
            "following_gripper",
            "falling",
            "entity_id",
            "position_xyz_m",
            "coordinate_frame",
            "localization_confidence",
            "timestamp",
        ):
            if key in raw:
                item[key] = raw[key]
        detections.append(item)
    return {
        "available": bool(observation.get("available", False)),
        "source": observation.get("source"),
        "timestamp": observation.get("timestamp"),
        "frame_count": len(frames),
        "detections": detections,
        "signals": dict(latest.get("signals") or observation.get("signals") or {}),
    }


def _frames(observation: dict[str, Any]) -> list[dict[str, Any]]:
    raw_frames = observation.get("frames")
    if isinstance(raw_frames, list):
        return [frame for frame in raw_frames if isinstance(frame, dict)]
    if isinstance(observation.get("detections"), list) or isinstance(
        observation.get("signals"), dict
    ):
        return [
            {
                "detections": list(observation.get("detections") or []),
                "signals": dict(observation.get("signals") or {}),
            }
        ]
    return []


def _latest_signals(observation: dict[str, Any]) -> dict[str, Any]:
    frames = _frames(observation)
    if frames:
        return dict(frames[-1].get("signals") or {})
    return dict(observation.get("signals") or {})


def _canonical(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def _target_names(step: PlanStep) -> set[str]:
    values = [
        step.get("target"),
        *list(step.get("_trusted_target_aliases") or []),
    ]
    return {
        canonical
        for value in values
        if (canonical := _canonical(str(value or "")))
    }


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None
