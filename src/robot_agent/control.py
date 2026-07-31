from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal


class TaskCancelled(RuntimeError):
    pass


@dataclass
class ExecutionControl:
    """Thread-safe cooperative pause/cancel state for long-running tasks."""

    _resume_event: threading.Event = field(default_factory=threading.Event)
    _cancel_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._resume_event.set()

    @property
    def status(self) -> Literal["running", "paused", "cancelled"]:
        if self._cancel_event.is_set():
            return "cancelled"
        return "running" if self._resume_event.is_set() else "paused"

    def pause(self) -> None:
        self._resume_event.clear()

    def resume(self) -> None:
        if not self._cancel_event.is_set():
            self._resume_event.set()

    def cancel(self) -> None:
        self._cancel_event.set()
        self._resume_event.set()

    def checkpoint(self, timeout_seconds: float = 0.1) -> None:
        while not self._resume_event.wait(timeout_seconds):
            if self._cancel_event.is_set():
                raise TaskCancelled("Task was cancelled.")
        if self._cancel_event.is_set():
            raise TaskCancelled("Task was cancelled.")
