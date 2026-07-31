from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol


class TaskStore(Protocol):
    def save(self, task_id: str, snapshot: dict[str, Any]) -> None: ...

    def load(self, task_id: str) -> dict[str, Any] | None: ...


class NullTaskStore:
    def save(self, task_id: str, snapshot: dict[str, Any]) -> None:
        return None

    def load(self, task_id: str) -> dict[str, Any] | None:
        return None


class JsonTaskStore:
    """Atomic JSON task snapshots suitable for restart recovery and replay."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, task_id: str, snapshot: dict[str, Any]) -> None:
        target = self._path(task_id)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{task_id}-", suffix=".tmp", dir=self.directory
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(snapshot, stream, ensure_ascii=False, indent=2, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def load(self, task_id: str) -> dict[str, Any] | None:
        target = self._path(task_id)
        if not target.exists():
            return None
        with target.open("r", encoding="utf-8") as stream:
            decoded = json.load(stream)
        if not isinstance(decoded, dict):
            raise ValueError(f"Task snapshot '{target}' is not a JSON object.")
        return decoded

    def _path(self, task_id: str) -> Path:
        safe = "".join(character for character in task_id if character.isalnum() or character in "-_")
        if not safe or safe != task_id:
            raise ValueError("task_id may contain only letters, numbers, '-' and '_'.")
        return self.directory / f"{safe}.json"
