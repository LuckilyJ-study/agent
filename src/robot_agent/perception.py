from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from copy import deepcopy
from typing import Any, Iterable, Protocol


class PerceptionProvider(Protocol):
    """Reserved boundary for the future high-rate camera/perception pipeline."""

    hardware_ready: bool
    supports_target_configuration: bool
    supports_localization: bool
    localization_modes: frozenset[str]

    def observe(self) -> dict[str, Any]: ...


class TargetAwarePerceptionProvider(PerceptionProvider, Protocol):
    """Optional extension implemented by a YOLO-World style provider."""

    def configure_targets(self, labels: Iterable[str]) -> None: ...


@dataclass
class NullPerceptionProvider:
    """Explicit no-perception provider for software bring-up.

    Unlike a fake camera image, this marks observations as unavailable so no
    caller can mistake the placeholder for real environmental verification.
    """

    reason: str = "Perception is not connected; attach the high-rate camera provider on hardware."
    hardware_ready: bool = field(default=False, init=False)
    supports_target_configuration: bool = field(default=False, init=False)
    supports_localization: bool = field(default=False, init=False)
    localization_modes: frozenset[str] = field(
        default_factory=frozenset,
        init=False,
    )

    def observe(self) -> dict[str, Any]:
        return {
            "available": False,
            "source": "null_perception",
            "reason": self.reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frames": [],
        }

    def configure_targets(self, labels: Iterable[str]) -> None:
        return None


@dataclass
class ScriptedPerceptionProvider:
    """Camera-free provider for deterministic Monitor tests and demos."""

    snapshots: list[dict[str, Any]]
    repeat_last: bool = True
    targets: list[str] = field(default_factory=list)
    hardware_ready: bool = field(default=False, init=False)
    supports_target_configuration: bool = field(default=True, init=False)
    supports_localization: bool = field(default=True, init=False)
    localization_modes: frozenset[str] = field(
        default_factory=lambda: frozenset({"bbox_2d", "robot_base_xyz"}),
        init=False,
    )
    _queue: deque[dict[str, Any]] = field(init=False, repr=False)
    _last: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = deque(deepcopy(self.snapshots))

    def configure_targets(self, labels: Iterable[str]) -> None:
        self.targets = [str(value) for value in labels if str(value).strip()]

    def observe(self) -> dict[str, Any]:
        if self._queue:
            self._last = self._queue.popleft()
        elif not self.repeat_last:
            self._last = None
        snapshot = deepcopy(
            self._last
            or {
                "available": False,
                "source": "scripted_perception_exhausted",
                "frames": [],
            }
        )
        snapshot.setdefault("available", True)
        snapshot.setdefault("source", "scripted_perception")
        snapshot.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        snapshot["configured_targets"] = list(self.targets)
        return snapshot


@dataclass
class AgentStateProvider:
    perception: PerceptionProvider = field(default_factory=NullPerceptionProvider)
    controller: Any = None

    def observe(self) -> dict[str, Any]:
        return self.perception.observe()

    def configure_targets(self, labels: Iterable[str]) -> None:
        configure = getattr(self.perception, "configure_targets", None)
        if callable(configure):
            configure(labels)

    def robot_state(self) -> dict[str, Any]:
        if self.controller is None:
            return {"available": False, "source": "no_robot_controller"}
        return self.controller.get_state()
