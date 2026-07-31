from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_world_service import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    FrameSource,
    SimpleTemporalTracker,
    YoloWorldDetector,
)


DEFAULT_IMAGE = (
    Path(__file__).resolve().parent
    / "YOLO-World"
    / "demo"
    / "sample_images"
    / "bus.jpg"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one YOLO-World-S image inference.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--labels", default="bus,person")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    detector = YoloWorldDetector(
        args.config.resolve(),
        args.checkpoint.resolve(),
        device=args.device,
        score_threshold=args.score_threshold,
        max_detections=100,
    )
    source = FrameSource(str(args.image.resolve()))
    try:
        frame = source.read()
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.image}")
        detections = SimpleTemporalTracker().update(detector.detect(frame, labels))
    finally:
        source.close()
    print(f"Device: {detector.device}")
    print(f"Image: {args.image.resolve()}")
    print(f"Labels: {labels}")
    print(f"Detection count: {len(detections)}")
    print(json.dumps(detections, ensure_ascii=False, indent=2))
    return 0 if detections else 1


if __name__ == "__main__":
    raise SystemExit(main())
