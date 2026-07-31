from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VisionServiceError(RuntimeError):
    """Raised when the standalone YOLO-World service contract is unavailable."""


@dataclass
class YoloWorldHttpPerceptionProvider:
    """Read structured 2-D observations from the YOLO-World HTTP service.

    This provider intentionally does not opt into physical execution. It has no
    depth or hand-eye calibration output, so ``hardware_mode`` must reject it
    until a calibrated provider adds ``robot_base_xyz`` localization.
    """

    endpoint: str = "http://127.0.0.1:8765"
    timeout_seconds: float = 2.0
    warmup_timeout_seconds: float = 5.0
    warmup_poll_seconds: float = 0.1
    warmup_minimum_frames: int = 3
    always_targets: tuple[str, ...] = ("robot gripper",)
    target_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    hardware_ready: bool = field(default=False, init=False)
    supports_target_configuration: bool = field(default=True, init=False)
    supports_localization: bool = field(default=True, init=False)
    localization_modes: frozenset[str] = field(
        default_factory=lambda: frozenset({"bbox_2d"}),
        init=False,
    )
    _configured_targets: list[str] = field(default_factory=list, init=False, repr=False)
    _configured_entity_map: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _last_error: str | None = field(default=None, init=False, repr=False)
    _configuration_error: str | None = field(default=None, init=False, repr=False)
    _cached_observation: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint.rstrip("/")
        self.target_aliases = self._normalize_aliases(self.target_aliases)
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("YOLO-World endpoint must start with http:// or https://.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.warmup_timeout_seconds < 0 or self.warmup_poll_seconds <= 0:
            raise ValueError(
                "warmup_timeout_seconds must be non-negative and "
                "warmup_poll_seconds must be positive."
            )
        if self.warmup_minimum_frames < 1:
            raise ValueError("warmup_minimum_frames must be at least 1.")

    def configure_targets(self, labels: Iterable[str]) -> None:
        logical_targets = self._normalize_targets(labels)
        targets: list[str] = []
        entity_map: dict[str, str] = {}
        for logical_target in logical_targets:
            prompts = [logical_target, *self._aliases_for(logical_target)]
            for prompt in self._normalize_targets(prompts):
                targets.append(prompt)
                entity_map[_canonical(prompt)] = logical_target
        for always_target in self._normalize_targets(self.always_targets):
            targets.append(always_target)
            entity_map[_canonical(always_target)] = always_target
        targets = self._normalize_targets(targets)
        if not targets or (
            targets == self._configured_targets and self._configuration_error is None
        ):
            return
        try:
            response = self._request_json(
                "/configure",
                method="POST",
                payload={"targets": targets},
            )
            configured = response.get("targets")
            if not isinstance(configured, list):
                raise VisionServiceError(
                    "YOLO-World /configure response did not contain targets[]."
                )
            self._configured_targets = self._normalize_targets(configured)
            self._configured_entity_map = {
                _canonical(prompt): entity_map.get(_canonical(prompt), prompt)
                for prompt in self._configured_targets
            }
            self._configuration_error = None
            self._cached_observation = self._wait_for_configured_observation()
            self._last_error = None
        except VisionServiceError as error:
            self._last_error = str(error)
            self._configuration_error = str(error)
            self._cached_observation = None

    def observe(self) -> dict[str, Any]:
        if self._configuration_error is not None:
            return self._unavailable_observation(
                "VISION_CONFIGURATION_FAILED",
                self._configuration_error,
            )
        if self._cached_observation is not None:
            observation = self._cached_observation
            self._cached_observation = None
            return observation
        try:
            response = self._request_json("/observe")
            observation = self._normalize_observation(response)
            self._last_error = None
            return observation
        except VisionServiceError as error:
            self._last_error = str(error)
            return self._unavailable_observation(
                "VISION_SERVICE_UNAVAILABLE",
                str(error),
            )

    def health(self) -> dict[str, Any]:
        try:
            response = self._request_json("/health")
            self._last_error = None
            return response
        except VisionServiceError as error:
            self._last_error = str(error)
            return {
                "status": "unavailable",
                "endpoint": self.endpoint,
                "error": str(error),
            }

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @staticmethod
    def _normalize_targets(labels: Iterable[Any]) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for raw in labels:
            label = str(raw).strip()
            key = " ".join(label.casefold().replace("_", " ").split())
            if not label or key in seen:
                continue
            selected.append(label)
            seen.add(key)
        return selected

    @staticmethod
    def _normalize_aliases(
        aliases: dict[str, tuple[str, ...]] | dict[str, Iterable[str]]
    ) -> dict[str, tuple[str, ...]]:
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_target, raw_aliases in dict(aliases or {}).items():
            target = str(raw_target).strip()
            if not target:
                raise ValueError("Vision target alias keys cannot be empty.")
            values = [raw_aliases] if isinstance(raw_aliases, str) else raw_aliases
            if values is None:
                values = []
            aliases_for_target = tuple(
                YoloWorldHttpPerceptionProvider._normalize_targets(values)
            )
            normalized[target] = aliases_for_target
        return normalized

    def _aliases_for(self, logical_target: str) -> tuple[str, ...]:
        target_key = _canonical(logical_target)
        for configured_target, aliases in self.target_aliases.items():
            if _canonical(configured_target) == target_key:
                return aliases
            if target_key in {_canonical(alias) for alias in aliases}:
                return (configured_target, *aliases)
        return ()

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.endpoint + path,
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            message = error.read().decode("utf-8", errors="replace")
            raise VisionServiceError(
                f"YOLO-World service returned HTTP {error.code}: {message}"
            ) from error
        except (
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise VisionServiceError(
                f"Could not call YOLO-World service at {self.endpoint}: {error}"
            ) from error
        if not isinstance(decoded, dict):
            raise VisionServiceError("YOLO-World service returned non-object JSON.")
        return decoded

    def _wait_for_configured_observation(self) -> dict[str, Any] | None:
        if self.warmup_timeout_seconds == 0:
            return None
        deadline = time.monotonic() + self.warmup_timeout_seconds
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                observation = self._normalize_observation(
                    self._request_json("/observe")
                )
                actual_targets = self._normalize_targets(
                    observation.get("configured_targets") or []
                )
                if (
                    observation["available"]
                    and actual_targets == self._configured_targets
                    and len(observation["frames"]) >= self.warmup_minimum_frames
                ):
                    return observation
            except VisionServiceError as error:
                last_error = str(error)
            time.sleep(self.warmup_poll_seconds)
        reason = (
            "Timed out waiting for "
            f"{self.warmup_minimum_frames} frames after target configuration."
        )
        if last_error:
            reason += f" Last service error: {last_error}"
        raise VisionServiceError(reason)

    def _unavailable_observation(
        self,
        reason: str,
        error: str,
    ) -> dict[str, Any]:
        return {
            "available": False,
            "source": "yolo_world_http",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configured_targets": list(self._configured_targets),
            "frames": [],
            "reason": reason,
            "details": {"error": error, "endpoint": self.endpoint},
        }

    def _normalize_observation(self, response: dict[str, Any]) -> dict[str, Any]:
        raw_frames = response.get("frames")
        if not isinstance(raw_frames, list):
            raise VisionServiceError("YOLO-World /observe response requires frames[].")
        frames: list[dict[str, Any]] = []
        for raw_frame in raw_frames[-30:]:
            if not isinstance(raw_frame, dict):
                continue
            detections: list[dict[str, Any]] = []
            raw_detections = raw_frame.get("detections", [])
            if not isinstance(raw_detections, list):
                raw_detections = []
            for raw_detection in raw_detections:
                if not isinstance(raw_detection, dict):
                    continue
                label = str(raw_detection.get("label") or "").strip()
                confidence = raw_detection.get("confidence")
                box = raw_detection.get("bbox_xyxy")
                if (
                    not label
                    or isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not isinstance(box, list)
                    or len(box) != 4
                    or any(
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        for value in box
                    )
                ):
                    continue
                detection = {
                    "label": label,
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(value) for value in box],
                }
                entity_id = self._configured_entity_map.get(_canonical(label))
                if entity_id:
                    detection["entity_id"] = entity_id
                for key in ("track_id", "following_gripper", "falling"):
                    if key in raw_detection:
                        detection[key] = raw_detection[key]
                detections.append(detection)
            frames.append(
                {
                    "timestamp": raw_frame.get("timestamp"),
                    "camera_id": raw_frame.get("camera_id"),
                    "image_size": dict(raw_frame.get("image_size") or {}),
                    "detections": detections,
                    "signals": dict(raw_frame.get("signals") or {}),
                }
            )
        return {
            "available": bool(response.get("available", False)) and bool(frames),
            "source": str(response.get("source") or "yolo_world_http"),
            "timestamp": response.get("timestamp"),
            "configured_targets": list(response.get("configured_targets") or []),
            "camera_id": response.get("camera_id"),
            "frames": frames,
            "last_error": response.get("last_error"),
        }


def _canonical(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())
