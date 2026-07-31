from __future__ import annotations

import re
from dataclasses import dataclass


ALLOWED_DIRECTIONS = {"left", "right", "forward", "back", "up", "down"}
MAX_RELATIVE_MOVE_CM = 1.0


@dataclass(frozen=True)
class RelativeMotion:
    direction: str
    distance_cm: float


def parse_relative_motion_target(target: str) -> RelativeMotion:
    """Parse a whitelist-safe target such as 'left 0.5 cm'."""
    match = re.fullmatch(
        r"\s*(left|right|forward|back|up|down)\s+([0-9]+(?:\.[0-9]+)?)\s*cm\s*",
        target.lower(),
    )
    if not match:
        raise ValueError("move_relative target must look like 'left 0.5 cm'.")

    direction = match.group(1)
    distance_cm = float(match.group(2))
    if direction not in ALLOWED_DIRECTIONS:
        raise ValueError(f"Unsupported direction: {direction}.")
    if distance_cm <= 0 or distance_cm > MAX_RELATIVE_MOVE_CM:
        raise ValueError(f"Relative move distance must be > 0 and <= {MAX_RELATIVE_MOVE_CM} cm.")
    return RelativeMotion(direction=direction, distance_cm=distance_cm)
