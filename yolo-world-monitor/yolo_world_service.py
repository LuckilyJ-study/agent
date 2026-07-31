from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Sequence


SERVICE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SERVICE_ROOT / "YOLO-World"
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "pretrain"
    / "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
)
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT / "weights" / "yolo_world_v2_1_s_640.pth"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def bbox_center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2)


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def is_gripper_label(label: str) -> bool:
    normalized = canonical(label)
    return any(
        token in normalized
        for token in ("gripper", "end effector", "robot hand", "夹爪")
    )


@dataclass
class TrackState:
    track_id: int
    label: str
    bbox_xyxy: list[float]
    center: tuple[float, float]
    motion: tuple[float, float] = (0.0, 0.0)
    falling_streak: int = 0


class SimpleTemporalTracker:
    """Small dependency-free tracker for the integration prototype.

    It matches detections by label and image-space proximity. The generated
    motion flags are hints for StructuredActionMonitor, not safety guarantees.
    """

    def __init__(
        self,
        *,
        maximum_center_distance_px: float = 160.0,
        falling_delta_px: float = 5.0,
        gripper_proximity_px: float = 180.0,
        follow_motion_tolerance_px: float = 18.0,
        minimum_follow_motion_px: float = 2.0,
    ) -> None:
        self.maximum_center_distance_px = maximum_center_distance_px
        self.falling_delta_px = falling_delta_px
        self.gripper_proximity_px = gripper_proximity_px
        self.follow_motion_tolerance_px = follow_motion_tolerance_px
        self.minimum_follow_motion_px = minimum_follow_motion_px
        self._tracks: dict[str, list[TrackState]] = {}
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(self, detections: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw in detections:
            label = str(raw.get("label") or "").strip()
            box = raw.get("bbox_xyxy")
            if not label or not isinstance(box, list) or len(box) != 4:
                continue
            grouped.setdefault(canonical(label), []).append(dict(raw))

        next_tracks: dict[str, list[TrackState]] = {}
        enriched: list[dict[str, Any]] = []
        for label_key, items in grouped.items():
            previous = list(self._tracks.get(label_key, []))
            used_previous: set[int] = set()
            for raw in items:
                box = [float(value) for value in raw["bbox_xyxy"]]
                center = bbox_center(box)
                match_index = self._best_match(box, center, previous, used_previous)
                if match_index is None:
                    state = TrackState(
                        track_id=self._next_track_id,
                        label=str(raw["label"]),
                        bbox_xyxy=box,
                        center=center,
                    )
                    self._next_track_id += 1
                else:
                    used_previous.add(match_index)
                    prior = previous[match_index]
                    motion = (center[0] - prior.center[0], center[1] - prior.center[1])
                    falling_streak = (
                        prior.falling_streak + 1
                        if motion[1] >= self.falling_delta_px
                        and motion[1] >= abs(motion[0]) * 0.5
                        else 0
                    )
                    state = TrackState(
                        track_id=prior.track_id,
                        label=str(raw["label"]),
                        bbox_xyxy=box,
                        center=center,
                        motion=motion,
                        falling_streak=falling_streak,
                    )
                next_tracks.setdefault(label_key, []).append(state)
                item = dict(raw)
                item["bbox_xyxy"] = box
                item["track_id"] = state.track_id
                if state.falling_streak >= 2:
                    item["falling"] = True
                item["_center"] = state.center
                item["_motion"] = state.motion
                enriched.append(item)

        self._tracks = next_tracks
        self._annotate_gripper_following(enriched)
        for item in enriched:
            item.pop("_center", None)
            item.pop("_motion", None)
        return enriched

    def _best_match(
        self,
        box: list[float],
        center: tuple[float, float],
        previous: list[TrackState],
        used_previous: set[int],
    ) -> int | None:
        best_index: int | None = None
        best_score = -1.0
        for index, track in enumerate(previous):
            if index in used_previous:
                continue
            distance = math.dist(center, track.center)
            if distance > self.maximum_center_distance_px:
                continue
            score = bbox_iou(box, track.bbox_xyxy) + (
                1.0 - distance / self.maximum_center_distance_px
            )
            if score > best_score:
                best_index = index
                best_score = score
        return best_index

    def _annotate_gripper_following(self, detections: list[dict[str, Any]]) -> None:
        grippers = [item for item in detections if is_gripper_label(str(item["label"]))]
        if not grippers:
            return
        for item in detections:
            if is_gripper_label(str(item["label"])):
                continue
            center = item["_center"]
            gripper = min(grippers, key=lambda value: math.dist(center, value["_center"]))
            if math.dist(center, gripper["_center"]) > self.gripper_proximity_px:
                continue
            target_motion = item["_motion"]
            gripper_motion = gripper["_motion"]
            if max(math.dist((0.0, 0.0), target_motion), math.dist((0.0, 0.0), gripper_motion)) < self.minimum_follow_motion_px:
                continue
            motion_difference = math.dist(target_motion, gripper_motion)
            item["following_gripper"] = (
                motion_difference <= self.follow_motion_tolerance_px
            )


class YoloWorldDetector:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        *,
        device: str,
        score_threshold: float,
        max_detections: int,
    ) -> None:
        if not config_path.is_file():
            raise FileNotFoundError(f"YOLO-World config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"YOLO-World checkpoint not found: {checkpoint_path}")
        for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "third_party" / "mmyolo"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        try:
            import cv2
            import torch
            from mmengine.config import Config
            from mmengine.dataset import Compose
            from mmdet.apis import init_detector
            from mmdet.utils import get_test_pipeline_cfg
        except ImportError as error:
            raise RuntimeError(
                "YOLO-World dependencies are not installed in this environment. "
                "Install the cloned repository in a dedicated environment first."
            ) from error

        selected_device = device
        if device == "auto":
            selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg = Config.fromfile(str(config_path))
        cfg.work_dir = str(SERVICE_ROOT / "work_dirs")
        cfg.load_from = str(checkpoint_path)
        self.model = init_detector(
            cfg,
            checkpoint=str(checkpoint_path),
            # Avoid constructing the LVIS test dataset only to obtain a
            # visualization palette. The HTTP service performs inference only.
            palette="random",
            device=selected_device,
        )
        pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        self.pipeline = Compose(pipeline_cfg)
        self.cv2 = cv2
        self.torch = torch
        self.device = selected_device
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    def detect(self, frame_bgr: Any, labels: Sequence[str]) -> list[dict[str, Any]]:
        if not labels:
            return []
        image_rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        texts = [[label] for label in labels] + [[" "]]
        data_info = {"img": image_rgb, "img_id": 0, "texts": texts}
        data_info = self.pipeline(data_info)
        batch = {
            "inputs": data_info["inputs"].unsqueeze(0),
            "data_samples": [data_info["data_samples"]],
        }
        with self.torch.no_grad():
            output = self.model.test_step(batch)[0]
        instances = output.pred_instances
        instances = instances[instances.scores.float() > self.score_threshold]
        if len(instances.scores) > self.max_detections:
            indices = instances.scores.float().topk(self.max_detections)[1]
            instances = instances[indices]
        values = instances.cpu().numpy()
        detections: list[dict[str, Any]] = []
        for box, label_index, score in zip(
            values["bboxes"], values["labels"], values["scores"]
        ):
            index = int(label_index)
            if index < 0 or index >= len(labels):
                continue
            detections.append(
                {
                    "label": labels[index],
                    "confidence": float(score),
                    "bbox_xyxy": [float(value) for value in box],
                }
            )
        return detections


class FrameSource:
    IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

    def __init__(self, source: str) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "opencv-python and numpy are required by the vision service."
            ) from error
        self.cv2 = cv2
        self.source = source
        path = Path(source)
        self._static_image = None
        self._capture = None
        if path.is_file() and path.suffix.casefold() in self.IMAGE_SUFFIXES:
            encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            self._static_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if self._static_image is None:
                raise RuntimeError(f"Could not read image source: {path}")
            self.description = str(path.resolve())
            return
        capture_source: int | str = int(source) if source.isdigit() else source
        self._capture = cv2.VideoCapture(capture_source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera/video source: {source}")
        self.description = str(source)

    def read(self) -> Any | None:
        if self._static_image is not None:
            return self._static_image.copy()
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if ok:
            return frame
        if not self.source.isdigit():
            self._capture.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok:
                return frame
        return None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()


class VisionRuntime:
    def __init__(
        self,
        detector: YoloWorldDetector,
        source: FrameSource,
        *,
        labels: Sequence[str],
        camera_id: str,
        max_fps: float,
        frame_buffer_size: int,
    ) -> None:
        self.detector = detector
        self.source = source
        self.camera_id = camera_id
        self.interval_seconds = 1.0 / max_fps
        self.frames: deque[dict[str, Any]] = deque(maxlen=frame_buffer_size)
        self.tracker = SimpleTemporalTracker()
        self._labels = self._normalize_labels(labels)
        self._generation = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    @staticmethod
    def _normalize_labels(labels: Iterable[str]) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for raw in labels:
            label = str(raw).strip()
            key = canonical(label)
            if not label or key in seen:
                continue
            selected.append(label)
            seen.add(key)
        if len(selected) > 32:
            raise ValueError("At most 32 target labels may be configured.")
        return selected

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="yolo-world-inference",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.source.close()

    def configure(self, labels: Iterable[str]) -> dict[str, Any]:
        selected = self._normalize_labels(labels)
        if not selected:
            raise ValueError("At least one target label is required.")
        with self._lock:
            if selected != self._labels:
                self._labels = selected
                self._generation += 1
                self.frames.clear()
                self.tracker.reset()
            return {"targets": list(self._labels), "generation": self._generation}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            frames = [dict(frame) for frame in self.frames]
            labels = list(self._labels)
            error = self.last_error
        return {
            "available": bool(frames),
            "source": "yolo_world_s",
            "timestamp": frames[-1]["timestamp"] if frames else utc_now(),
            "configured_targets": labels,
            "camera_id": self.camera_id,
            "frames": frames,
            "last_error": error,
        }

    def health(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "status": "ok" if self._thread and self._thread.is_alive() else "stopped",
            "model": "YOLO-World-V2.1-S",
            "device": self.detector.device,
            "source": self.source.description,
            "configured_targets": snapshot["configured_targets"],
            "frame_count": len(snapshot["frames"]),
            "last_error": snapshot["last_error"],
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                frame = self.source.read()
                if frame is None:
                    raise RuntimeError("Camera/video source returned no frame.")
                with self._lock:
                    labels = list(self._labels)
                    generation = self._generation
                detections = self.detector.detect(frame, labels)
                with self._lock:
                    if generation != self._generation:
                        continue
                    tracked = self.tracker.update(detections)
                    self.frames.append(
                        {
                            "timestamp": utc_now(),
                            "camera_id": self.camera_id,
                            "image_size": {
                                "width": int(frame.shape[1]),
                                "height": int(frame.shape[0]),
                            },
                            "detections": tracked,
                            "signals": {},
                        }
                    )
                    self.last_error = None
            except Exception as error:
                with self._lock:
                    self.last_error = f"{type(error).__name__}: {error}"
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.interval_seconds - elapsed))


class VisionRequestHandler(BaseHTTPRequestHandler):
    runtime: VisionRuntime
    server_version = "YoloWorldMonitor/0.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, self.runtime.health())
            return
        if self.path == "/observe":
            self._send_json(200, self.runtime.snapshot())
            return
        self._send_json(404, {"error": "NOT_FOUND"})

    def do_POST(self) -> None:
        if self.path != "/configure":
            self._send_json(404, {"error": "NOT_FOUND"})
            return
        try:
            payload = self._read_json()
            targets = payload.get("targets")
            if not isinstance(targets, list):
                raise ValueError("Request body requires targets[].")
            result = self.runtime.configure(targets)
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": "INVALID_REQUEST", "message": str(error)})
            return
        self._send_json(200, {"status": "configured", **result})

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length.") from error
        if length <= 0 or length > 64 * 1024:
            raise ValueError("Request body must be between 1 byte and 64 KiB.")
        decoded = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Request body must be a JSON object.")
        return decoded

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[vision-http] {self.address_string()} {format_string % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expose YOLO-World-S detections through a local HTTP service."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or static image path.",
    )
    parser.add_argument("--camera-id", default="primary")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--max-fps", type=float, default=5.0)
    parser.add_argument("--frame-buffer-size", type=int, default=12)
    parser.add_argument(
        "--labels",
        default="pizza dough,robot gripper,pizza tray",
        help="Initial comma-separated English detection prompts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.score_threshold <= 1:
        raise SystemExit("--score-threshold must be between 0 and 1.")
    if args.max_fps <= 0:
        raise SystemExit("--max-fps must be positive.")
    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    print("Loading YOLO-World-S. The first startup may take some time.")
    detector = YoloWorldDetector(
        args.config.resolve(),
        args.checkpoint.resolve(),
        device=args.device,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
    )
    source = FrameSource(args.source)
    runtime = VisionRuntime(
        detector,
        source,
        labels=labels,
        camera_id=args.camera_id,
        max_fps=args.max_fps,
        frame_buffer_size=args.frame_buffer_size,
    )
    VisionRequestHandler.runtime = runtime
    server = ThreadingHTTPServer((args.host, args.port), VisionRequestHandler)
    runtime.start()
    print(f"Vision service: http://{args.host}:{args.port}")
    print(f"Health: http://{args.host}:{args.port}/health")
    print(f"Source: {source.description}")
    print(f"Device: {detector.device}")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("Stopping vision service.")
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
