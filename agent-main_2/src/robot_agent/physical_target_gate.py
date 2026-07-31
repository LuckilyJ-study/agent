from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from .state import PlanStep, VerificationResult


TARGET_LOCALIZED_ACTIONS = frozenset({"pick", "place", "manipulate", "move_to"})
PERCEPTION_ONLY_ACTIONS = frozenset({"inspect"})
REQUIRED_HARDWARE_LOCALIZATION_MODES = frozenset(
    {"bbox_2d", "robot_base_xyz"}
)


class PhysicalPerceptionConfigurationError(ValueError):
    """Raised when a provider is not explicitly suitable for hardware use."""


def validate_physical_perception_provider(provider: Any) -> None:
    """Fail closed unless a perception provider opts into physical execution.

    Simulation providers intentionally do not satisfy this contract.  A future
    YOLO/camera provider should expose ``hardware_ready=True``, implement
    ``configure_targets()``, and advertise both ``bbox_2d`` and
    ``robot_base_xyz`` in ``localization_modes``.
    """

    if not bool(getattr(provider, "hardware_ready", False)):
        raise PhysicalPerceptionConfigurationError(
            "hardware_mode requires a perception provider with hardware_ready=True."
        )
    if not callable(getattr(provider, "configure_targets", None)):
        raise PhysicalPerceptionConfigurationError(
            "hardware perception must implement configure_targets(labels)."
        )
    raw_modes = getattr(provider, "localization_modes", ())
    if isinstance(raw_modes, str):
        modes = {raw_modes}
    else:
        try:
            modes = {str(value) for value in raw_modes}
        except TypeError as error:
            raise PhysicalPerceptionConfigurationError(
                "hardware perception localization_modes must be iterable."
            ) from error
    missing = REQUIRED_HARDWARE_LOCALIZATION_MODES.difference(modes)
    if missing:
        raise PhysicalPerceptionConfigurationError(
            "hardware perception is missing localization modes: "
            + ", ".join(sorted(missing))
        )


@dataclass(frozen=True)
class _DetectionCandidate:
    raw: Mapping[str, Any]
    frame_timestamp: Any
    confidence: float


class PhysicalTargetGate:
    """Fail-closed pre-action gate for target-dependent hardware skills.

    ``pick``, ``place`` and ``manipulate`` need a fresh matching detection,
    a valid 2-D bounding box, and a 3-D position already transformed into the
    robot-base frame.  ``inspect`` only needs a live perception source because
    its purpose may be to find a currently invisible target.  Target-independent
    primitives are left untouched.

    On success, the gate writes a compact, media-free ``perception_grounding``
    structure into ``step.parameters`` for the trusted router/executor.
    """

    def __init__(
        self,
        *,
        minimum_detection_confidence: float = 0.5,
        max_observation_age_seconds: float = 1.0,
        max_future_skew_seconds: float = 0.25,
        max_abs_position_m: float = 2.0,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        if not 0 <= minimum_detection_confidence <= 1:
            raise ValueError(
                "minimum_detection_confidence must be between 0 and 1."
            )
        if max_observation_age_seconds <= 0:
            raise ValueError("max_observation_age_seconds must be positive.")
        if max_future_skew_seconds < 0:
            raise ValueError("max_future_skew_seconds cannot be negative.")
        if max_abs_position_m <= 0:
            raise ValueError("max_abs_position_m must be positive.")
        self.minimum_detection_confidence = minimum_detection_confidence
        self.max_observation_age_seconds = max_observation_age_seconds
        self.max_future_skew_seconds = max_future_skew_seconds
        self.max_abs_position_m = max_abs_position_m
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def before_action(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> VerificationResult | None:
        del robot_state
        action_type = str(step.get("action_type") or "")
        if action_type not in TARGET_LOCALIZED_ACTIONS | PERCEPTION_ONLY_ACTIONS:
            return None
        if not bool(observation.get("available", False)):
            return self._failure(
                "PERCEPTION_UNAVAILABLE",
                step,
                reason=str(
                    observation.get("reason")
                    or "The physical perception source is unavailable."
                ),
                observation=observation,
            )
        if action_type in PERCEPTION_ONLY_ACTIONS:
            return None

        target_names = _target_names(step)
        candidates = _matching_candidates(
            observation,
            target_names,
            self.minimum_detection_confidence,
        )
        if not candidates:
            return self._failure(
                "TARGET_NOT_VISIBLE",
                step,
                reason=(
                    "No current detection matched the target at the required "
                    "confidence."
                ),
                observation=observation,
                extra={"target_names": sorted(target_names)},
            )

        now = self._now_provider()
        if now.tzinfo is None:
            raise ValueError("now_provider must return a timezone-aware datetime.")
        stale_candidates: list[dict[str, Any]] = []
        localized_candidates: list[
            tuple[_DetectionCandidate, dict[str, Any], datetime]
        ] = []
        unlocalized_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            observed_at = _detection_timestamp(
                candidate.raw,
                candidate.frame_timestamp,
                observation.get("timestamp"),
            )
            if observed_at is None:
                stale_candidates.append(
                    {
                        "label": _detection_label(candidate.raw),
                        "reason": "timestamp_missing_or_invalid",
                    }
                )
                continue
            age_seconds = (now - observed_at).total_seconds()
            if (
                age_seconds > self.max_observation_age_seconds
                or age_seconds < -self.max_future_skew_seconds
            ):
                stale_candidates.append(
                    {
                        "label": _detection_label(candidate.raw),
                        "observed_at": observed_at.isoformat(),
                        "age_seconds": age_seconds,
                    }
                )
                continue
            grounding, missing_fields = _grounding_from_detection(
                candidate.raw,
                observation,
                observed_at,
                self.max_abs_position_m,
            )
            if grounding is None:
                unlocalized_candidates.append(
                    {
                        "label": _detection_label(candidate.raw),
                        "confidence": candidate.confidence,
                        "missing_or_invalid_fields": missing_fields,
                    }
                )
                continue
            localized_candidates.append((candidate, grounding, observed_at))

        if localized_candidates:
            selected, grounding, _ = max(
                localized_candidates,
                key=lambda value: value[0].confidence,
            )
            parameters = dict(step.get("parameters") or {})
            parameters["perception_grounding"] = grounding
            step["parameters"] = parameters
            return None

        if unlocalized_candidates:
            return self._failure(
                "TARGET_NOT_LOCALIZED",
                step,
                reason=(
                    "The target was detected but lacks a valid bounding box and "
                    "robot-base XYZ position."
                ),
                observation=observation,
                extra={"detections": unlocalized_candidates},
            )
        return self._failure(
            "PERCEPTION_STALE",
            step,
            reason="Matching target detections are missing a fresh trusted timestamp.",
            observation=observation,
            extra={"detections": stale_candidates},
        )

    def during_action(
        self,
        step: PlanStep,
        observation: dict[str, Any],
        robot_state: dict[str, Any],
    ) -> VerificationResult | None:
        """Stop target-dependent motion if the physical perception stream drops."""

        del robot_state
        action_type = str(step.get("action_type") or "")
        if action_type not in TARGET_LOCALIZED_ACTIONS | PERCEPTION_ONLY_ACTIONS:
            return None
        if bool(observation.get("available", False)):
            return None
        result = self._failure(
            "PERCEPTION_UNAVAILABLE",
            step,
            reason=str(
                observation.get("reason")
                or "The physical perception stream became unavailable."
            ),
            observation=observation,
        )
        result["details"]["phase"] = "during_action"
        result["details"]["blocked_before_execution"] = False
        return result

    @staticmethod
    def _failure(
        error_type: str,
        step: PlanStep,
        *,
        reason: str,
        observation: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        details: dict[str, Any] = {
            "phase": "before_action",
            "blocked_before_execution": True,
            "failed_skill": step.get("action_type"),
            "target": step.get("target"),
            "reason": reason,
            "perception_source": observation.get("source"),
            "observation_timestamp": observation.get("timestamp"),
        }
        if extra:
            details.update(dict(extra))
        return {
            "success": False,
            "error_type": error_type,
            "confidence": 1.0,
            "verification_scope": "physical",
            "details": details,
        }


def _matching_candidates(
    observation: Mapping[str, Any],
    target_names: set[str],
    minimum_confidence: float,
) -> list[_DetectionCandidate]:
    frame, frame_timestamp = _latest_frame(observation)
    detections = frame.get("detections")
    if not isinstance(detections, list):
        detections = observation.get("detections")
    if not isinstance(detections, list):
        return []
    matches: list[_DetectionCandidate] = []
    for raw in detections:
        if not isinstance(raw, Mapping):
            continue
        confidence = _finite_float(raw.get("confidence"))
        if confidence is None or confidence < minimum_confidence:
            continue
        identifiers = {
            _canonical(str(raw.get(key) or ""))
            for key in ("entity_id", "label", "name")
            if raw.get(key)
        }
        raw_aliases = raw.get("aliases")
        if isinstance(raw_aliases, Iterable) and not isinstance(
            raw_aliases, (str, bytes, Mapping)
        ):
            identifiers.update(
                _canonical(str(value)) for value in raw_aliases if str(value).strip()
            )
        if identifiers.isdisjoint(target_names):
            continue
        matches.append(_DetectionCandidate(raw, frame_timestamp, confidence))
    return matches


def _latest_frame(
    observation: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Any]:
    frames = observation.get("frames")
    if isinstance(frames, list):
        for frame in reversed(frames):
            if isinstance(frame, Mapping):
                return frame, frame.get("timestamp")
    return observation, observation.get("timestamp")


def _target_names(step: PlanStep) -> set[str]:
    names = {_canonical(str(step.get("target") or ""))}
    aliases = step.get("_trusted_target_aliases")
    if isinstance(aliases, Iterable) and not isinstance(
        aliases, (str, bytes, Mapping)
    ):
        names.update(
            _canonical(str(value))
            for value in aliases
            if str(value).strip()
        )
    return {name for name in names if name}


def _grounding_from_detection(
    raw: Mapping[str, Any],
    observation: Mapping[str, Any],
    observed_at: datetime,
    max_abs_position_m: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    bbox = _valid_bbox(raw.get("bbox_xyxy"))
    localization = (
        raw.get("localization")
        if isinstance(raw.get("localization"), Mapping)
        else raw
    )
    position = _valid_xyz(
        localization.get("position_xyz_m"),
        max_abs=max_abs_position_m,
    )
    coordinate_frame = str(
        localization.get("coordinate_frame")
        or raw.get("coordinate_frame")
        or ""
    )
    missing: list[str] = []
    if bbox is None:
        missing.append("bbox_xyxy")
    if position is None:
        missing.append("position_xyz_m")
    if coordinate_frame != "robot_base":
        missing.append("coordinate_frame=robot_base")
    if missing:
        return None, missing

    grounding: dict[str, Any] = {
        "source": observation.get("source"),
        "observed_at": observed_at.isoformat(),
        "entity_id": raw.get("entity_id"),
        "label": _detection_label(raw),
        "confidence": float(raw["confidence"]),
        "bbox_xyxy": bbox,
        "position_xyz_m": position,
        "coordinate_frame": "robot_base",
    }
    for key in ("track_id", "localization_confidence"):
        value = localization.get(key, raw.get(key))
        if value is not None:
            grounding[key] = value
    return grounding, []


def _detection_timestamp(
    detection: Mapping[str, Any],
    frame_timestamp: Any,
    observation_timestamp: Any,
) -> datetime | None:
    for value in (
        detection.get("timestamp"),
        detection.get("observed_at"),
        frame_timestamp,
        observation_timestamp,
    ):
        parsed = _parse_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = [_finite_float(item) for item in value]
    if any(item is None for item in numbers):
        return None
    x1, y1, x2, y2 = (float(item) for item in numbers)
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _valid_xyz(value: Any, *, max_abs: float) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    numbers = [_finite_float(item) for item in value]
    if any(item is None for item in numbers):
        return None
    result = [float(item) for item in numbers]
    if any(abs(item) > max_abs for item in result):
        return None
    return result


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _detection_label(raw: Mapping[str, Any]) -> str | None:
    value = raw.get("label") or raw.get("name") or raw.get("entity_id")
    return str(value) if value is not None else None


def _canonical(value: str) -> str:
    return " ".join(
        value.casefold().replace("_", " ").replace("-", " ").split()
    )
