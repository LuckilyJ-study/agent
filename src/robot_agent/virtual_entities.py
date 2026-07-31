from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping


class VirtualEntityRegistrationError(ValueError):
    """A Planner target cannot be represented as a safe virtual entity."""


class VirtualEntityRegistry:
    """Resolve scene aliases and register simulation-only semantic entities.

    The registry deliberately stores no pose, coordinate, joint, or controller
    data.  It only turns an unknown semantic target into a safe local entity ID
    and records the minimal symbolic metadata needed by high-level grounding.
    """

    ACTION_ENTITY_TYPES = {
        "pick": "movable_object",
        "place": "placement_target",
        "inspect": "observable_entity",
        "move_to": "semantic_location",
    }
    ACTION_ROLES = {
        "pick": "grasp_target",
        "place": "placement_target",
        "inspect": "observation_target",
        "move_to": "navigation_target",
    }
    _TYPE_PRIORITY = {
        "observable_entity": 0,
        "semantic_location": 1,
        "placement_target": 2,
        "movable_object": 3,
    }

    def __init__(
        self,
        objects: Mapping[str, Any] | None = None,
        *,
        max_new_entities: int = 32,
        max_target_length: int = 256,
    ) -> None:
        if max_new_entities < 1:
            raise ValueError("max_new_entities must be at least 1.")
        if max_target_length < 1:
            raise ValueError("max_target_length must be at least 1.")
        self.max_new_entities = max_new_entities
        self.max_target_length = max_target_length
        self._objects = deepcopy(dict(objects or {}))
        self._aliases = _entity_aliases(self._objects)
        self._pending: dict[str, dict[str, Any]] = {}

    def resolve(self, target: str) -> str | None:
        """Return the canonical scene ID for an existing target alias."""

        return self._aliases.get(_canonical(target))

    def resolve_or_register(self, action_type: str, target: str) -> str | None:
        """Resolve a target, or register it when the action supports simulation.

        ``None`` means the action is not eligible for virtual registration.
        Callers can then apply their normal closed-world rejection behavior.
        """

        raw_target = str(target).strip()
        existing = self.resolve(raw_target)
        if existing is not None:
            self._add_role(existing, action_type)
            return existing

        entity_type = self.ACTION_ENTITY_TYPES.get(str(action_type))
        if entity_type is None:
            return None
        self._validate_target(raw_target)
        if len(self._pending) >= self.max_new_entities:
            raise VirtualEntityRegistrationError(
                "The plan requests too many new virtual entities."
            )

        entity_id = self._allocate_id(raw_target)
        role = self.ACTION_ROLES[action_type]
        entity: dict[str, Any] = {
            "type": entity_type,
            "display_name": raw_target,
            "aliases": [raw_target],
            "roles": [role],
            "virtual": True,
            "source": "open_world_simulation",
            "physical_confirmation": False,
        }
        if action_type == "pick":
            entity["location"] = "virtual_workspace"

        self._objects[entity_id] = entity
        self._pending[entity_id] = entity
        self._aliases[_canonical(entity_id)] = entity_id
        self._aliases[_canonical(raw_target)] = entity_id
        return entity_id

    def peek_patch(self) -> dict[str, Any]:
        """Return new registrations without clearing them."""

        if not self._pending:
            return {}
        return {"objects": deepcopy(self._pending)}

    def consume_patch(self) -> dict[str, Any]:
        """Return and clear only the newly registered entity records."""

        patch = self.peek_patch()
        self._pending.clear()
        return patch

    def _validate_target(self, target: str) -> None:
        if not target:
            raise VirtualEntityRegistrationError(
                "A virtual entity target cannot be empty."
            )
        if len(target) > self.max_target_length:
            raise VirtualEntityRegistrationError(
                "A virtual entity target is longer than the configured limit."
            )
        if any(ord(character) < 32 for character in target):
            raise VirtualEntityRegistrationError(
                "A virtual entity target cannot contain control characters."
            )

    def _allocate_id(self, target: str) -> str:
        digest = sha256(_canonical(target).encode("utf-8")).hexdigest()[:12]
        base = f"virtual_entity_{digest}"
        candidate = base
        suffix = 2
        while candidate in self._objects:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _add_role(self, entity_id: str, action_type: str) -> None:
        entity = self._pending.get(entity_id)
        entity_type = self.ACTION_ENTITY_TYPES.get(str(action_type))
        role = self.ACTION_ROLES.get(str(action_type))
        if entity is None or entity_type is None or role is None:
            return
        roles = list(entity.get("roles") or [])
        if role not in roles:
            roles.append(role)
            entity["roles"] = roles
        current_type = str(entity.get("type") or "")
        if self._TYPE_PRIORITY[entity_type] > self._TYPE_PRIORITY.get(
            current_type, -1
        ):
            entity["type"] = entity_type
        if action_type == "pick":
            entity.setdefault("location", "virtual_workspace")


def _entity_aliases(objects: Mapping[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for entity_id, raw in objects.items():
        entity = dict(raw or {}) if isinstance(raw, Mapping) else {}
        for value in [entity_id, *list(entity.get("aliases") or [])]:
            canonical = _canonical(str(value))
            if canonical:
                aliases[canonical] = str(entity_id)
    return aliases


def _canonical(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())
