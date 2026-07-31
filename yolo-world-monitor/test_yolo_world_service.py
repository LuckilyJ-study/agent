from __future__ import annotations

import unittest

from yolo_world_service import SimpleTemporalTracker


class SimpleTemporalTrackerTests(unittest.TestCase):
    def test_assigns_stable_ids_and_detects_gripper_following(self):
        tracker = SimpleTemporalTracker()
        first = tracker.update(
            [
                {
                    "label": "dough",
                    "confidence": 0.9,
                    "bbox_xyxy": [100, 100, 140, 140],
                },
                {
                    "label": "robot gripper",
                    "confidence": 0.95,
                    "bbox_xyxy": [130, 90, 180, 140],
                },
            ]
        )
        second = tracker.update(
            [
                {
                    "label": "dough",
                    "confidence": 0.9,
                    "bbox_xyxy": [110, 100, 150, 140],
                },
                {
                    "label": "robot gripper",
                    "confidence": 0.95,
                    "bbox_xyxy": [140, 90, 190, 140],
                },
            ]
        )

        first_dough = next(item for item in first if item["label"] == "dough")
        second_dough = next(item for item in second if item["label"] == "dough")
        self.assertEqual(first_dough["track_id"], second_dough["track_id"])
        self.assertTrue(second_dough["following_gripper"])

    def test_marks_sustained_downward_motion_as_falling(self):
        tracker = SimpleTemporalTracker(falling_delta_px=4.0)
        frames = []
        for y in (100, 106, 112):
            frames.append(
                tracker.update(
                    [
                        {
                            "label": "dough",
                            "confidence": 0.9,
                            "bbox_xyxy": [100, y, 140, y + 40],
                        }
                    ]
                )
            )

        self.assertNotIn("falling", frames[1][0])
        self.assertTrue(frames[2][0]["falling"])


if __name__ == "__main__":
    unittest.main()
