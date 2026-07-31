from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Protocol

from .state import PlanStep

CAMERA_NAMES = ("primary", "secondary", "wrist")
DEFAULT_STATE_DIM = 8  # x, y, z, quat(4), gripper — the Pi05 training state layout


class ObservationSource(Protocol):
    """Provides raw camera frames and robot proprioception for one skill step."""

    def read_state(self, step: PlanStep) -> list[float]:
        """Return the current robot state vector."""

    def read_images(self, step: PlanStep) -> dict[str, bytes]:
        """Return one encoded image (JPEG/PNG bytes) per camera name."""


class StaticObservationSource:
    """Observation source for bring-up: reads images from a directory and the
    robot state from an environment variable, falling back to placeholders.

    Configuration:
    - ROBOT_AGENT_OBS_IMAGE_DIR: directory holding primary.jpg, secondary.jpg,
      wrist.jpg (or .png). Missing cameras get a generated placeholder image.
    - ROBOT_AGENT_ROBOT_STATE: JSON list for the state vector, e.g. "[0,0,0,0,0,0,1,0]".
    """

    def __init__(
        self,
        image_dir: str | None = None,
        state_dim: int = DEFAULT_STATE_DIM,
    ) -> None:
        self.image_dir = Path(image_dir or os.getenv("ROBOT_AGENT_OBS_IMAGE_DIR", ""))
        self.state_dim = state_dim

    def read_state(self, step: PlanStep) -> list[float]:
        raw = os.getenv("ROBOT_AGENT_ROBOT_STATE")
        if raw:
            try:
                state = [float(value) for value in json.loads(raw)]
                if state:
                    return state
            except (TypeError, ValueError):
                pass
        return [0.0] * self.state_dim

    def read_images(self, step: PlanStep) -> dict[str, bytes]:
        images: dict[str, bytes] = {}
        for camera in CAMERA_NAMES:
            images[camera] = self._read_camera_image(camera)
        return images

    def _read_camera_image(self, camera: str) -> bytes:
        if self.image_dir.is_dir():
            for suffix in (".jpg", ".jpeg", ".png"):
                candidate = self.image_dir / f"{camera}{suffix}"
                if candidate.is_file():
                    return candidate.read_bytes()
        return _placeholder_jpeg()


_PLACEHOLDER_JPEG_CACHE: bytes | None = None

# Smallest valid JPEG, used only when Pillow is unavailable. Real inference
# will reject it and the gateway falls back to stub actions.
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////"
    "////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QA"
    "HwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQR"
    "BRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdI"
    "SUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2"
    "t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD/2Q=="
)


def _placeholder_jpeg(width: int = 1280, height: int = 720) -> bytes:
    """Black placeholder frame large enough for the Pi05 service crop logic."""
    global _PLACEHOLDER_JPEG_CACHE
    if _PLACEHOLDER_JPEG_CACHE is not None:
        return _PLACEHOLDER_JPEG_CACHE
    try:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (width, height)).save(buffer, format="JPEG")
        _PLACEHOLDER_JPEG_CACHE = buffer.getvalue()
    except ImportError:
        # Minimal JPEG marker stream. It is intentionally only a transport
        # placeholder; production deployments must supply real camera frames.
        try:
            _PLACEHOLDER_JPEG_CACHE = base64.b64decode(_TINY_JPEG_B64, validate=True)
        except (ValueError, base64.binascii.Error):
            _PLACEHOLDER_JPEG_CACHE = b"\xff\xd8\xff\xd9"
    return _PLACEHOLDER_JPEG_CACHE


def build_structured_observation(
    step: PlanStep,
    task_text: str,
    source: ObservationSource | None = None,
) -> dict[str, Any]:
    """Build the structured observation consumed by the Pi05 policy.

    Shape mirrors the Pi05 inference service contract:
    task text, robot state vector, and one base64 image per camera.
    """
    selected_source = source or StaticObservationSource()
    state = selected_source.read_state(step)
    raw_images = selected_source.read_images(step)
    images = {
        camera: base64.b64encode(raw_images[camera]).decode("ascii")
        for camera in CAMERA_NAMES
    }
    return {
        "task_text": task_text,
        "skill": step["skill"],
        "target": step["target"],
        "expected_result": step["expected_result"],
        "state": state,
        "images": images,
        "source": "agent_master_observation_builder",
    }


def summarize_observation(observation: dict[str, Any] | None) -> dict[str, Any]:
    """Compact, log-safe view of an observation (no base64 image payloads)."""
    if not observation:
        return {}
    images = observation.get("images") or {}
    return {
        "task_text": observation.get("task_text"),
        "skill": observation.get("skill"),
        "target": observation.get("target"),
        "state_dim": len(observation.get("state") or []),
        "image_bytes": {camera: len(images.get(camera, "")) for camera in CAMERA_NAMES},
        "source": observation.get("source"),
    }
