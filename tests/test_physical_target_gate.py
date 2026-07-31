from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from robot_agent.physical_target_gate import (
    PhysicalPerceptionConfigurationError,
    PhysicalTargetGate,
    validate_physical_perception_provider,
)


class PhysicalTargetGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc)
        self.gate = PhysicalTargetGate(
            minimum_detection_confidence=0.7,
            max_observation_age_seconds=0.5,
            now_provider=lambda: self.now,
        )

    def test_target_action_blocks_when_perception_is_unavailable(self):
        step = self._step("pick")

        result = self.gate.before_action(
            step,
            {
                "available": False,
                "source": "yolo_world_s",
                "reason": "camera disconnected",
                "frames": [],
            },
            {},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["error_type"], "PERCEPTION_UNAVAILABLE")
        self.assertTrue(result["details"]["blocked_before_execution"])
        self.assertNotIn("perception_grounding", step["parameters"])

    def test_target_action_blocks_when_matching_target_is_not_visible(self):
        step = self._step("pick")

        result = self.gate.before_action(
            step,
            self._observation(
                [
                    self._detection(
                        entity_id="other_object",
                        label="other object",
                    ),
                    self._detection(
                        confidence=0.69,
                    ),
                ]
            ),
            {},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["error_type"], "TARGET_NOT_VISIBLE")
        self.assertEqual(
            result["details"]["target_names"],
            ["ball", "small ball"],
        )

    def test_untrusted_parameter_alias_cannot_bypass_target_matching(self):
        step = self._step("pick")
        step.pop("_trusted_target_aliases")
        step["parameters"]["target_aliases"] = ["other_object"]

        result = self.gate.before_action(
            step,
            self._observation(
                [
                    self._detection(
                        entity_id="other_object",
                        label="other object",
                    )
                ]
            ),
            {},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["error_type"], "TARGET_NOT_VISIBLE")

    def test_target_action_blocks_when_detection_is_not_localized(self):
        malformed = (
            {"bbox_xyxy": None},
            {"bbox_xyxy": [10, 20, 30, 40], "position_xyz_m": None},
            {
                "bbox_xyxy": [10, 20, 30, 40],
                "position_xyz_m": [0.4, 0.1, 0.2],
                "coordinate_frame": "camera",
            },
            {
                "bbox_xyxy": [10, 20, 30, 40],
                "position_xyz_m": [20.0, 0.1, 0.2],
                "coordinate_frame": "robot_base",
            },
        )
        for overrides in malformed:
            with self.subTest(overrides=overrides):
                step = self._step("manipulate")
                detection = self._detection()
                detection.update(overrides)

                result = self.gate.before_action(
                    step,
                    self._observation([detection]),
                    {},
                )

                self.assertIsNotNone(result)
                self.assertEqual(result["error_type"], "TARGET_NOT_LOCALIZED")
                self.assertIn(
                    "missing_or_invalid_fields",
                    result["details"]["detections"][0],
                )
                self.assertNotIn("perception_grounding", step["parameters"])

    def test_target_action_blocks_stale_matching_detection(self):
        step = self._step("place")
        stale_time = self.now - timedelta(seconds=2)

        result = self.gate.before_action(
            step,
            self._observation(
                [self._detection(timestamp=stale_time.isoformat())],
                timestamp=self.now.isoformat(),
            ),
            {},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["error_type"], "PERCEPTION_STALE")
        self.assertGreater(
            result["details"]["detections"][0]["age_seconds"],
            self.gate.max_observation_age_seconds,
        )

    def test_valid_target_grounding_passes_and_preserves_structured_fields(self):
        step = self._step("pick")
        detection = self._detection(
            entity_id="small_ball",
            label="small ball",
            confidence=0.93,
            track_id=17,
            localization_confidence=0.88,
            image=b"must not be copied",
        )

        result = self.gate.before_action(
            step,
            self._observation([detection]),
            {"connected": True},
        )

        self.assertIsNone(result)
        self.assertEqual(step["parameters"]["existing"], "kept")
        grounding = step["parameters"]["perception_grounding"]
        self.assertEqual(
            grounding,
            {
                "source": "physical_yolo",
                "observed_at": self.now.isoformat(),
                "entity_id": "small_ball",
                "label": "small ball",
                "confidence": 0.93,
                "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
                "position_xyz_m": [0.4, 0.1, 0.2],
                "coordinate_frame": "robot_base",
                "track_id": 17,
                "localization_confidence": 0.88,
            },
        )
        self.assertNotIn("image", grounding)

    def test_inspect_only_requires_available_perception(self):
        step = self._step("inspect")

        result = self.gate.before_action(
            step,
            {
                "available": True,
                "source": "physical_yolo",
                "frames": [],
            },
            {},
        )

        self.assertIsNone(result)
        self.assertNotIn("perception_grounding", step["parameters"])

    def test_inspect_blocks_when_perception_is_unavailable(self):
        result = self.gate.before_action(
            self._step("inspect"),
            {"available": False, "source": "physical_yolo"},
            {},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["error_type"], "PERCEPTION_UNAVAILABLE")

    def test_target_independent_primitives_do_not_require_localization(self):
        for action_type in (
            "move_home",
            "open_gripper",
            "close_gripper",
            "move_relative",
        ):
            with self.subTest(action_type=action_type):
                result = self.gate.before_action(
                    self._step(action_type),
                    {"available": False, "source": "null_perception"},
                    {},
                )
                self.assertIsNone(result)

    def test_provider_validation_is_opt_in_and_requires_localization_modes(self):
        class PhysicalProvider:
            hardware_ready = True
            localization_modes = {"bbox_2d", "robot_base_xyz"}

            def configure_targets(self, labels):
                del labels

        validate_physical_perception_provider(PhysicalProvider())

        class ScriptedLikeProvider:
            hardware_ready = False
            localization_modes = {"bbox_2d", "robot_base_xyz"}

            def configure_targets(self, labels):
                del labels

        with self.assertRaises(PhysicalPerceptionConfigurationError):
            validate_physical_perception_provider(ScriptedLikeProvider())

        class NoRobotBaseLocalization(PhysicalProvider):
            localization_modes = {"bbox_2d"}

        with self.assertRaisesRegex(
            PhysicalPerceptionConfigurationError,
            "robot_base_xyz",
        ):
            validate_physical_perception_provider(NoRobotBaseLocalization())

    def _step(self, action_type):
        return {
            "step_id": 1,
            "action_type": action_type,
            "target": "ball",
            "_trusted_target_aliases": ["small_ball"],
            "expected_result": "the requested action completed",
            "parameters": {
                "existing": "kept",
            },
        }

    def _observation(self, detections, *, timestamp=None):
        observed_at = timestamp or self.now.isoformat()
        return {
            "available": True,
            "source": "physical_yolo",
            "timestamp": observed_at,
            "frames": [
                {
                    "timestamp": observed_at,
                    "detections": detections,
                }
            ],
        }

    def _detection(self, **overrides):
        detection = {
            "entity_id": "small_ball",
            "label": "small ball",
            "confidence": 0.9,
            "bbox_xyxy": [10, 20, 30, 40],
            "position_xyz_m": [0.4, 0.1, 0.2],
            "coordinate_frame": "robot_base",
        }
        detection.update(overrides)
        return detection


if __name__ == "__main__":
    unittest.main()
