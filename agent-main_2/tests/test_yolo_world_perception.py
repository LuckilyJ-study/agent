from __future__ import annotations

import io
import json
import sys
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import run_agent_simulation
from robot_agent.monitor import StructuredActionMonitor
from robot_agent.physical_target_gate import (
    PhysicalPerceptionConfigurationError,
    validate_physical_perception_provider,
)
from robot_agent.yolo_world_perception import (
    VisionServiceError,
    YoloWorldHttpPerceptionProvider,
)


class FakeYoloProvider(YoloWorldHttpPerceptionProvider):
    def __init__(self, responses):
        super().__init__(endpoint="http://vision.test", warmup_timeout_seconds=0)
        self.responses = list(responses)
        self.requests = []

    def _request_json(self, path, *, method="GET", payload=None):
        self.requests.append((path, method, payload))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class MockVisionHandler(BaseHTTPRequestHandler):
    targets = ["cup", "robot gripper"]

    def do_GET(self):
        if self.path == "/health":
            self._send(
                {
                    "status": "ok",
                    "model": "mock-yolo-world-s",
                    "device": "simulation",
                    "source": "unit-test",
                }
            )
            return
        if self.path == "/observe":
            primary = next(
                (value for value in self.targets if "gripper" not in value),
                self.targets[0],
            )
            frames = []
            for index in range(4):
                frames.append(
                    {
                        "timestamp": f"frame-{index}",
                        "camera_id": "test",
                        "image_size": {"width": 640, "height": 480},
                        "detections": [
                            {
                                "label": primary,
                                "confidence": 0.95,
                                "bbox_xyxy": [100, 100, 180, 180],
                                "track_id": 1,
                                "following_gripper": True,
                            }
                        ],
                        "signals": {
                            "target_visible": True,
                            "placement_complete": True,
                        },
                    }
                )
            self._send(
                {
                    "available": True,
                    "source": "mock_yolo_world_s",
                    "timestamp": "latest",
                    "configured_targets": list(self.targets),
                    "frames": frames,
                }
            )
            return
        self._send({"error": "NOT_FOUND"}, status=404)

    def do_POST(self):
        if self.path != "/configure":
            self._send({"error": "NOT_FOUND"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).targets = list(payload["targets"])
        self._send({"status": "configured", "targets": list(self.targets)})

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return None


class YoloWorldPerceptionTests(unittest.TestCase):
    def test_simulation_cli_exposes_safe_vision_tuning_options(self):
        args = run_agent_simulation.build_parser().parse_args(
            [
                "--vision-min-confidence",
                "0.15",
                "--vision-disable-gripper-tracking",
            ]
        )

        self.assertEqual(args.vision_min_confidence, 0.15)
        self.assertTrue(args.vision_disable_gripper_tracking)

    def test_simulation_cli_accepts_repeatable_vision_target_aliases(self):
        args = run_agent_simulation.build_parser().parse_args(
            [
                "--vision-target-alias",
                "tray=pizza tray",
                "--vision-target-alias",
                "dough=pizza dough",
            ]
        )

        self.assertEqual(
            run_agent_simulation._parse_vision_target_aliases(args.vision_target_alias),
            {"tray": ("pizza tray",), "dough": ("pizza dough",)},
        )

    def test_two_dimensional_provider_cannot_opt_into_hardware_mode(self):
        provider = YoloWorldHttpPerceptionProvider(
            endpoint="http://vision.test",
            warmup_timeout_seconds=0,
        )

        self.assertFalse(provider.hardware_ready)
        self.assertEqual(provider.localization_modes, frozenset({"bbox_2d"}))
        with self.assertRaises(PhysicalPerceptionConfigurationError):
            validate_physical_perception_provider(provider)

    def test_configures_step_target_and_gripper_without_duplicates(self):
        provider = FakeYoloProvider(
            [{"status": "configured", "targets": ["dough", "robot gripper"]}]
        )

        provider.configure_targets(["dough", "dough"])
        provider.configure_targets(["dough"])

        self.assertEqual(
            provider.requests,
            [
                (
                    "/configure",
                    "POST",
                    {"targets": ["dough", "robot gripper"]},
                )
            ],
        )

    def test_configures_logical_target_with_open_vocabulary_alias(self):
        provider = FakeYoloProvider(
            [{"status": "configured", "targets": ["tray", "pizza tray"]}]
        )
        provider.target_aliases = {"tray": ("pizza tray",)}

        provider.configure_targets(["tray"])

        self.assertEqual(
            provider.requests,
            [
                (
                    "/configure",
                    "POST",
                    {"targets": ["tray", "pizza tray", "robot gripper"]},
                )
            ],
        )

    def test_alias_detection_gets_logical_entity_id(self):
        provider = FakeYoloProvider(
            [
                {"status": "configured", "targets": ["tray", "pizza tray"]},
                {
                    "available": True,
                    "configured_targets": ["tray", "pizza tray"],
                    "frames": [
                        {
                            "detections": [
                                {
                                    "label": "pizza tray",
                                    "confidence": 0.8,
                                    "bbox_xyxy": [1, 2, 3, 4],
                                }
                            ]
                        }
                    ],
                },
            ]
        )
        provider.warmup_timeout_seconds = 0
        provider.target_aliases = {"tray": ("pizza tray",)}
        provider.configure_targets(["tray"])

        observation = provider.observe()

        self.assertEqual(
            observation["frames"][0]["detections"][0]["label"], "pizza tray"
        )
        self.assertEqual(
            observation["frames"][0]["detections"][0]["entity_id"], "tray"
        )

    def test_observation_keeps_only_monitor_contract_fields(self):
        provider = FakeYoloProvider(
            [
                {
                    "available": True,
                    "source": "yolo_world_s",
                    "timestamp": "2026-07-31T00:00:00Z",
                    "configured_targets": ["dough", "robot gripper"],
                    "frames": [
                        {
                            "timestamp": "2026-07-31T00:00:00Z",
                            "camera_id": "primary",
                            "raw_pixels": "must-not-cross-service-boundary",
                            "image_size": {"width": 640, "height": 480},
                            "detections": [
                                {
                                    "label": "dough",
                                    "confidence": 0.91,
                                    "bbox_xyxy": [10, 20, 100, 120],
                                    "track_id": 7,
                                    "following_gripper": True,
                                    "private_tensor": "drop-me",
                                }
                            ],
                            "signals": {"target_visible": True},
                        }
                    ],
                }
            ]
        )

        observation = provider.observe()

        self.assertTrue(observation["available"])
        detection = observation["frames"][0]["detections"][0]
        self.assertEqual(detection["track_id"], 7)
        self.assertTrue(detection["following_gripper"])
        self.assertNotIn("raw_pixels", observation["frames"][0])
        self.assertNotIn("private_tensor", detection)

    def test_configuration_waits_for_monitor_frame_window(self):
        provider = YoloWorldHttpPerceptionProvider(
            endpoint="http://vision.test",
            warmup_timeout_seconds=0.2,
            warmup_poll_seconds=0.001,
            warmup_minimum_frames=3,
        )
        one_frame = {
            "available": True,
            "configured_targets": ["cup", "robot gripper"],
            "frames": [{"detections": []}],
        }
        three_frames = {
            "available": True,
            "configured_targets": ["cup", "robot gripper"],
            "frames": [{"detections": []} for _ in range(3)],
        }

        with patch.object(
            provider,
            "_request_json",
            side_effect=[
                {"status": "configured", "targets": ["cup", "robot gripper"]},
                one_frame,
                three_frames,
            ],
        ) as request:
            provider.configure_targets(["cup"])

        self.assertEqual(request.call_count, 3)
        self.assertEqual(len(provider.observe()["frames"]), 3)

    def test_service_failure_becomes_unavailable_observation(self):
        provider = FakeYoloProvider([VisionServiceError("connection refused")])

        observation = provider.observe()

        self.assertFalse(observation["available"])
        self.assertEqual(observation["reason"], "VISION_SERVICE_UNAVAILABLE")
        self.assertIn("connection refused", observation["details"]["error"])

    def test_required_perception_fails_closed(self):
        monitor = StructuredActionMonitor(require_perception=True)
        step = {
            "step_id": 1,
            "action_type": "pick",
            "target": "dough",
            "expected_result": "dough is held",
        }

        result = monitor.before_action(
            step,
            {
                "available": False,
                "source": "yolo_world_http",
                "reason": "VISION_SERVICE_UNAVAILABLE",
                "frames": [],
            },
            {},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["error_type"], "PERCEPTION_FAILED")

    def test_target_directed_move_is_blocked_when_not_visible(self):
        monitor = StructuredActionMonitor(require_perception=True)
        step = {
            "step_id": 1,
            "action_type": "move_to",
            "target": "cup",
            "expected_result": "arm is near cup",
        }
        observation = {
            "available": True,
            "source": "yolo_world_http",
            "frames": [{"detections": []} for _ in range(3)],
        }

        result = monitor.before_action(step, observation, {})

        self.assertIsNotNone(result)
        self.assertEqual(result["error_type"], "TARGET_LOST")
        self.assertEqual(result["details"]["phase"], "before_action")

    def test_simulation_cli_uses_http_vision_provider_end_to_end(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockVisionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        argv = [
            "run_agent_simulation.py",
            "--planner",
            "scripted",
            "--task",
            "Put cup on table",
            "--pick-failures",
            "0",
            "--vision-endpoint",
            endpoint,
            "--vision-disable-gripper-tracking",
        ]
        try:
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                exit_code = run_agent_simulation.main()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
