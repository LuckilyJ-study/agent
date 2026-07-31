from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal


ConditionOperator = Literal["eq", "ne", "exists", "not_exists", "in", "gt", "gte", "lt", "lte"]
EffectOperation = Literal["set", "delete", "increment"]


class WorldStateError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    path: str
    operator: ConditionOperator = "eq"
    value: Any = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Condition:
        path = str(raw.get("path") or "").strip()
        operator = str(raw.get("operator", "eq"))
        if not path:
            raise WorldStateError("Condition path cannot be empty.")
        if operator not in {
            "eq", "ne", "exists", "not_exists", "in", "gt", "gte", "lt", "lte"
        }:
            raise WorldStateError(f"Unsupported condition operator: {operator}.")
        return cls(path, operator, raw.get("value"))


@dataclass(frozen=True)
class Effect:
    path: str
    operation: EffectOperation = "set"
    value: Any = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Effect:
        path = str(raw.get("path") or "").strip()
        operation = str(raw.get("operation", "set"))
        if not path:
            raise WorldStateError("Effect path cannot be empty.")
        if operation not in {"set", "delete", "increment"}:
            raise WorldStateError(f"Unsupported effect operation: {operation}.")
        return cls(path, operation, raw.get("value"))


@dataclass
class WorldState:
    """Small symbolic world model used before camera-derived state is available."""

    values: dict[str, Any] = field(default_factory=dict)
    version: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.version, "values": deepcopy(self.values)}

    def get(self, path: str, default: Any = None) -> Any:
        current: Any = self.values
        for part in _parts(path):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def contains(self, path: str) -> bool:
        sentinel = object()
        return self.get(path, sentinel) is not sentinel

    def set(self, path: str, value: Any) -> None:
        parts = _parts(path)
        current = self.values
        for part in parts[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = child
        current[parts[-1]] = deepcopy(value)
        self.version += 1

    def delete(self, path: str) -> None:
        parts = _parts(path)
        current: Any = self.values
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                return
            current = current[part]
        if isinstance(current, dict) and parts[-1] in current:
            del current[parts[-1]]
            self.version += 1

    def evaluate(self, condition: Condition) -> bool:
        exists = self.contains(condition.path)
        if condition.operator == "exists":
            return exists
        if condition.operator == "not_exists":
            return not exists
        actual = self.get(condition.path)
        if condition.operator == "eq":
            return actual == condition.value
        if condition.operator == "ne":
            return actual != condition.value
        if condition.operator == "in":
            try:
                return actual in condition.value
            except TypeError:
                return False
        try:
            if condition.operator == "gt":
                return actual > condition.value
            if condition.operator == "gte":
                return actual >= condition.value
            if condition.operator == "lt":
                return actual < condition.value
            if condition.operator == "lte":
                return actual <= condition.value
        except TypeError:
            return False
        return False

    def conditions_met(self, raw_conditions: list[dict[str, Any]]) -> bool:
        return all(self.evaluate(Condition.from_dict(raw)) for raw in raw_conditions)

    def apply_effects(self, raw_effects: list[dict[str, Any]]) -> None:
        for raw in raw_effects:
            effect = Effect.from_dict(raw)
            if effect.operation == "set":
                self.set(effect.path, effect.value)
            elif effect.operation == "delete":
                self.delete(effect.path)
            else:
                current = self.get(effect.path, 0)
                if not isinstance(current, (int, float)) or not isinstance(
                    effect.value, (int, float)
                ):
                    raise WorldStateError(
                        f"Increment effect requires numeric values at '{effect.path}'."
                    )
                self.set(effect.path, current + effect.value)


def _parts(path: str) -> list[str]:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise WorldStateError("World-state path cannot be empty.")
    return parts
