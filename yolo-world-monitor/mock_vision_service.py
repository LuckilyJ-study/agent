from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from typing import Any, Iterable

from yolo_world_service import VisionRequestHandler, canonical, is_gripper_label


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MockVisionRuntime:
    """Deterministic implementation of the real vision service contract."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.targets = ["dough", "robot gripper"]
        self.generation = 0

    def configure(self, labels: Iterable[str]) -> dict[str, Any]:
        selected: list[str] = []
        seen: set[str] = set()
        for raw in labels:
            label = str(raw).strip()
            key = canonical(label)
            if label and key not in seen:
                selected.append(label)
                seen.add(key)
        if not selected:
            raise ValueError("At least one target label is required.")
        self.targets = selected
        self.generation += 1
        return {"targets": list(self.targets), "generation": self.generation}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": "mock-yolo-world-s",
            "device": "simulation",
            "source": self.scenario,
            "configured_targets": list(self.targets),
            "frame_count": 4,
            "last_error": None,
        }

    def snapshot(self) -> dict[str, Any]:
        frames = self._frames()
        return {
            "available": True,
            "source": "mock_yolo_world_s",
            "timestamp": frames[-1]["timestamp"],
            "configured_targets": list(self.targets),
            "camera_id": "simulated_primary",
            "frames": frames,
            "last_error": None,
        }

    def _frames(self) -> list[dict[str, Any]]:
        primary = next(
            (target for target in self.targets if not is_gripper_label(target)),
            self.targets[0],
        )
        gripper = next(
            (target for target in self.targets if is_gripper_label(target)),
            "robot gripper",
        )
        frames: list[dict[str, Any]] = []
        for index in range(4):
            target_box = [100 + index * 8, 180, 180 + index * 8, 260]
            gripper_box = [145 + index * 8, 120, 225 + index * 8, 200]
            detections: list[dict[str, Any]] = []
            signals: dict[str, Any] = {}
            if self.scenario == "target_lost":
                signals["target_visible"] = False
            else:
                target_detection: dict[str, Any] = {
                    "label": primary,
                    "confidence": 0.95,
                    "bbox_xyxy": target_box,
                    "track_id": 1,
                    "following_gripper": self.scenario != "grasp_failed",
                }
                if self.scenario == "object_dropped" and index >= 2:
                    target_detection["following_gripper"] = False
                    target_detection["falling"] = True
                    target_detection["bbox_xyxy"] = [
                        target_box[0],
                        target_box[1] + index * 12,
                        target_box[2],
                        target_box[3] + index * 12,
                    ]
                detections.extend(
                    [
                        target_detection,
                        {
                            "label": gripper,
                            "confidence": 0.96,
                            "bbox_xyxy": gripper_box,
                            "track_id": 2,
                        },
                    ]
                )
                signals["target_visible"] = True
                signals["placement_complete"] = self.scenario == "visible"
                if self.scenario == "grasp_failed":
                    signals.update(
                        {
                            "grasp_succeeded": False,
                            "gripper_lifted": True,
                        }
                    )
                if self.scenario == "object_dropped" and index >= 2:
                    signals["object_dropped"] = True
            frames.append(
                {
                    "timestamp": utc_now(),
                    "camera_id": "simulated_primary",
                    "image_size": {"width": 640, "height": 480},
                    "detections": detections,
                    "signals": signals,
                }
            )
        return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a dependency-free mock of the YOLO-World HTTP service."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--scenario",
        choices=("visible", "target_lost", "grasp_failed", "object_dropped"),
        default="visible",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    VisionRequestHandler.runtime = MockVisionRuntime(args.scenario)
    server = ThreadingHTTPServer((args.host, args.port), VisionRequestHandler)
    print(f"Mock vision service: http://{args.host}:{args.port}")
    print(f"Scenario: {args.scenario}")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("Stopping mock vision service.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
