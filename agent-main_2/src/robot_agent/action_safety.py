from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal


ActionSemantics = Literal[
    "normalized",
    "joint_position_rad",
    "joint_delta_rad",
    "cartesian_delta_si",
]


class ActionSafetyConfigurationError(ValueError):
    """Raised when action limits cannot form a fail-closed profile."""


@dataclass(frozen=True)
class ActionSafetyCheck:
    safe: bool
    reason: str = "SAFE"
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class ActionChunkSafetyLimits:
    """Per-dimension limits for a policy action chunk.

    ``hardware_approved`` is intentionally false for the bundled normalized
    simulation profile. A deployment must construct a profile from the real
    action schema and robot/controller limits before a hardware-ready controller
    can receive policy output.
    """

    semantics: ActionSemantics
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    max_step_changes: tuple[float, ...]
    max_cumulative_changes: tuple[float, ...]
    max_chunk_size: int = 100
    require_reference: bool = False
    hardware_approved: bool = False
    profile_name: str = "unnamed-action-profile"
    dimension_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        dimension = len(self.lower_bounds)
        if dimension < 1:
            raise ActionSafetyConfigurationError(
                "Action safety limits require at least one dimension."
            )
        vectors = {
            "upper_bounds": self.upper_bounds,
            "max_step_changes": self.max_step_changes,
            "max_cumulative_changes": self.max_cumulative_changes,
        }
        for name, values in vectors.items():
            if len(values) != dimension:
                raise ActionSafetyConfigurationError(
                    f"{name} must contain {dimension} values."
                )
        if self.dimension_names and len(self.dimension_names) != dimension:
            raise ActionSafetyConfigurationError(
                f"dimension_names must contain {dimension} values."
            )
        if self.max_chunk_size < 1:
            raise ActionSafetyConfigurationError("max_chunk_size must be positive.")
        if not str(self.profile_name).strip():
            raise ActionSafetyConfigurationError("profile_name cannot be empty.")
        all_values = [
            *self.lower_bounds,
            *self.upper_bounds,
            *self.max_step_changes,
            *self.max_cumulative_changes,
        ]
        if any(not math.isfinite(float(value)) for value in all_values):
            raise ActionSafetyConfigurationError("Action limits must be finite.")
        if any(
            float(lower) >= float(upper)
            for lower, upper in zip(self.lower_bounds, self.upper_bounds)
        ):
            raise ActionSafetyConfigurationError(
                "Every action lower bound must be below its upper bound."
            )
        if any(
            float(value) <= 0
            for values in (self.max_step_changes, self.max_cumulative_changes)
            for value in values
        ):
            raise ActionSafetyConfigurationError(
                "Step and cumulative action limits must be positive."
            )
        if self.hardware_approved and self.profile_name == "unnamed-action-profile":
            raise ActionSafetyConfigurationError(
                "A hardware-approved profile requires an explicit profile_name."
            )

    @property
    def action_dim(self) -> int:
        return len(self.lower_bounds)

    @classmethod
    def normalized_simulation(cls, action_dim: int) -> ActionChunkSafetyLimits:
        if action_dim < 1:
            raise ActionSafetyConfigurationError("action_dim must be positive.")
        return cls(
            semantics="normalized",
            lower_bounds=(-1.0,) * action_dim,
            upper_bounds=(1.0,) * action_dim,
            max_step_changes=(0.5,) * action_dim,
            max_cumulative_changes=(1.5,) * action_dim,
            max_chunk_size=100,
            require_reference=False,
            hardware_approved=False,
            profile_name="normalized-simulation-only",
        )


class ActionChunkGuard:
    def __init__(self, limits: ActionChunkSafetyLimits) -> None:
        self.limits = limits

    def check(
        self,
        actions: Sequence[Sequence[float]],
        *,
        reference_values: Sequence[float] | None = None,
    ) -> ActionSafetyCheck:
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            return ActionSafetyCheck(False, "ACTION_CHUNK_INVALID")
        if not actions:
            return ActionSafetyCheck(False, "ACTION_CHUNK_EMPTY")
        if len(actions) > self.limits.max_chunk_size:
            return ActionSafetyCheck(
                False,
                "ACTION_CHUNK_TOO_LARGE",
                {"size": len(actions), "maximum": self.limits.max_chunk_size},
            )

        parsed_rows: list[tuple[float, ...]] = []
        for row_index, row in enumerate(actions):
            parsed = self._parse_row(row)
            if parsed is None:
                return ActionSafetyCheck(
                    False,
                    "ACTION_DIMENSION_OR_VALUE_INVALID",
                    {"row_index": row_index, "expected_dim": self.limits.action_dim},
                )
            for dimension, value in enumerate(parsed):
                lower = self.limits.lower_bounds[dimension]
                upper = self.limits.upper_bounds[dimension]
                if value < lower or value > upper:
                    return ActionSafetyCheck(
                        False,
                        "ACTION_VALUE_LIMIT_EXCEEDED",
                        self._dimension_details(row_index, dimension, value, lower, upper),
                    )
            parsed_rows.append(parsed)

        reference = None
        if reference_values is not None:
            reference = self._parse_row(reference_values)
            if reference is None:
                return ActionSafetyCheck(False, "ACTION_REFERENCE_INVALID")
        if self.limits.require_reference and reference is None:
            return ActionSafetyCheck(False, "ACTION_REFERENCE_REQUIRED")

        cumulative = [0.0] * self.limits.action_dim
        if self.limits.semantics == "joint_delta_rad":
            for row_index, row in enumerate(parsed_rows):
                for dimension, value in enumerate(row):
                    change = abs(value)
                    if change > self.limits.max_step_changes[dimension]:
                        return ActionSafetyCheck(
                            False,
                            "ACTION_STEP_DELTA_EXCEEDED",
                            self._dimension_details(
                                row_index,
                                dimension,
                                change,
                                0.0,
                                self.limits.max_step_changes[dimension],
                            ),
                        )
                    cumulative[dimension] += change
                    if cumulative[dimension] > self.limits.max_cumulative_changes[dimension]:
                        return ActionSafetyCheck(
                            False,
                            "ACTION_CUMULATIVE_DELTA_EXCEEDED",
                            self._dimension_details(
                                row_index,
                                dimension,
                                cumulative[dimension],
                                0.0,
                                self.limits.max_cumulative_changes[dimension],
                            ),
                        )
        else:
            previous = reference or parsed_rows[0]
            first_index = 0 if reference is not None else 1
            for row_index in range(first_index, len(parsed_rows)):
                row = parsed_rows[row_index]
                for dimension, value in enumerate(row):
                    change = abs(value - previous[dimension])
                    if change > self.limits.max_step_changes[dimension]:
                        return ActionSafetyCheck(
                            False,
                            "ACTION_STEP_DELTA_EXCEEDED",
                            self._dimension_details(
                                row_index,
                                dimension,
                                change,
                                0.0,
                                self.limits.max_step_changes[dimension],
                            ),
                        )
                    cumulative[dimension] += change
                    if cumulative[dimension] > self.limits.max_cumulative_changes[dimension]:
                        return ActionSafetyCheck(
                            False,
                            "ACTION_CUMULATIVE_DELTA_EXCEEDED",
                            self._dimension_details(
                                row_index,
                                dimension,
                                cumulative[dimension],
                                0.0,
                                self.limits.max_cumulative_changes[dimension],
                            ),
                        )
                previous = row

        return ActionSafetyCheck(
            True,
            details={
                "profile_name": self.limits.profile_name,
                "semantics": self.limits.semantics,
                "chunk_size": len(parsed_rows),
                "action_dim": self.limits.action_dim,
            },
        )

    def _parse_row(self, row: Sequence[float]) -> tuple[float, ...] | None:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return None
        if len(row) != self.limits.action_dim:
            return None
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in row
        ):
            return None
        parsed = tuple(float(value) for value in row)
        if any(not math.isfinite(value) for value in parsed):
            return None
        return parsed

    def _dimension_details(
        self,
        row_index: int,
        dimension: int,
        value: float,
        lower: float,
        upper: float,
    ) -> dict[str, Any]:
        details: dict[str, Any] = {
            "row_index": row_index,
            "dimension": dimension,
            "value": value,
            "minimum": lower,
            "maximum": upper,
        }
        if self.limits.dimension_names:
            details["dimension_name"] = self.limits.dimension_names[dimension]
        return details
