from __future__ import annotations

import unittest
from copy import deepcopy

from robot_agent.skill_grounding import (
    SceneSkillPlanner,
    SkillPlanGroundingError,
)
from robot_agent.virtual_entities import VirtualEntityRegistry


EMPTY_SCENE = {
    "scene": {
        "scene_id": "open_world_test",
        "skill_contracts": [],
        "goal_predicates": [],
    },
    "runtime": {
        "holding": None,
        "gripper": "unknown",
        "robot_location": "home",
    },
    "objects": {},
}


class _StaticPlanner:
    last_goal = {
        "description": "test goal",
        "conditions": [],
    }

    def __init__(self, plan):
        self.plan = deepcopy(plan)

    def create_plan_with_context(self, context):
        return deepcopy(self.plan)


def _step(
    step_id: int,
    action_type: str,
    target: str,
    *,
    parameters=None,
):
    return {
        "step_id": step_id,
        "action_type": action_type,
        "target": target,
        "expected_result": f"{action_type} completed",
        "parameters": dict(parameters or {}),
    }


def _context(scene):
    return {
        "original_task": "test task",
        "world_state": {"version": 0, "values": deepcopy(scene)},
        "capabilities": [],
    }


class VirtualEntityRegistryTests(unittest.TestCase):
    def test_registry_uses_safe_stable_id_and_consumes_only_new_objects(self):
        registry = VirtualEntityRegistry()

        entity_id = registry.resolve_or_register("pick", "小球.with/path")

        self.assertIsNotNone(entity_id)
        self.assertRegex(str(entity_id), r"^virtual_entity_[0-9a-f]{12}$")
        self.assertNotIn("小球", str(entity_id))
        patch = registry.consume_patch()
        self.assertEqual(set(patch), {"objects"})
        self.assertEqual(set(patch["objects"]), {entity_id})
        self.assertEqual(
            patch["objects"][entity_id]["location"],
            "virtual_workspace",
        )
        self.assertEqual(registry.consume_patch(), {})

    def test_registry_does_not_register_unapproved_action_type(self):
        registry = VirtualEntityRegistry()

        self.assertIsNone(registry.resolve_or_register("teleport", "moon"))
        self.assertEqual(registry.consume_patch(), {})


class SceneSkillPlannerVirtualEntityTests(unittest.TestCase):
    def test_closed_world_default_still_rejects_unknown_pick_target(self):
        planner = _StaticPlanner([_step(1, "pick", "小球")])
        grounded = SceneSkillPlanner(
            planner,
            EMPTY_SCENE,
            allow_one_repair=False,
        )

        with self.assertRaisesRegex(
            SkillPlanGroundingError,
            "not present in the current scene",
        ):
            grounded.create_plan_with_context(_context(EMPTY_SCENE))

        self.assertEqual(grounded.consume_world_patch(), {})

    def test_opt_in_registers_pick_without_leaking_simulated_final_state(self):
        planner = _StaticPlanner([_step(1, "pick", "小球")])
        grounded = SceneSkillPlanner(
            planner,
            EMPTY_SCENE,
            allow_one_repair=False,
            allow_virtual_entities=True,
        )

        plan = grounded.create_plan_with_context(_context(EMPTY_SCENE))
        entity_id = plan[0]["target"]
        patch = grounded.consume_world_patch()

        self.assertRegex(entity_id, r"^virtual_entity_[0-9a-f]{12}$")
        self.assertEqual(
            plan[0]["effects"],
            [
                {
                    "path": "runtime.holding",
                    "operation": "set",
                    "value": entity_id,
                },
                {
                    "path": f"objects.{entity_id}.location",
                    "operation": "set",
                    "value": "held",
                },
            ],
        )
        self.assertEqual(set(patch), {"objects"})
        self.assertEqual(
            patch["objects"][entity_id]["location"],
            "virtual_workspace",
        )
        self.assertNotIn("runtime", patch)
        self.assertEqual(grounded.drain_world_patch(), {})

    def test_pick_and_place_register_distinct_semantic_roles(self):
        planner = _StaticPlanner(
            [
                _step(1, "pick", "红色积木"),
                _step(2, "place", "蓝色盒子"),
            ]
        )
        grounded = SceneSkillPlanner(
            planner,
            EMPTY_SCENE,
            allow_one_repair=False,
            allow_virtual_entities=True,
        )

        plan = grounded.create_plan_with_context(_context(EMPTY_SCENE))
        picked_id = plan[0]["target"]
        destination_id = plan[1]["target"]
        patch = grounded.consume_world_patch()

        self.assertNotEqual(picked_id, destination_id)
        self.assertEqual(
            patch["objects"][picked_id]["type"],
            "movable_object",
        )
        self.assertEqual(
            patch["objects"][destination_id]["type"],
            "placement_target",
        )
        self.assertEqual(
            plan[1]["effects"][-1],
            {
                "path": f"objects.{picked_id}.location",
                "operation": "set",
                "value": destination_id,
            },
        )

    def test_unknown_inspect_target_can_be_registered(self):
        planner = _StaticPlanner([_step(1, "inspect", "红色标签")])
        grounded = SceneSkillPlanner(
            planner,
            EMPTY_SCENE,
            allow_one_repair=False,
            allow_virtual_entities=True,
        )

        plan = grounded.create_plan_with_context(_context(EMPTY_SCENE))
        patch = grounded.consume_world_patch()

        self.assertEqual(plan[0]["effects"], [])
        self.assertEqual(
            patch["objects"][plan[0]["target"]]["type"],
            "observable_entity",
        )

    def test_unknown_skill_and_generic_manipulate_remain_rejected(self):
        cases = [
            (
                deepcopy(EMPTY_SCENE),
                _step(1, "teleport", "moon"),
                "not present in the current scene",
            ),
            (
                deepcopy(EMPTY_SCENE),
                _step(
                    1,
                    "manipulate",
                    "未知盒子",
                    parameters={"operation": "open"},
                ),
                "not present in the current scene",
            ),
            (
                {
                    **deepcopy(EMPTY_SCENE),
                    "objects": {
                        "known_box": {
                            "type": "container",
                            "aliases": ["已知盒子"],
                            "state": "closed",
                        }
                    },
                },
                _step(
                    1,
                    "manipulate",
                    "已知盒子",
                    parameters={"operation": "open"},
                ),
                "no reusable contract",
            ),
        ]
        for scene, step, message in cases:
            with self.subTest(action=step["action_type"], target=step["target"]):
                grounded = SceneSkillPlanner(
                    _StaticPlanner([step]),
                    scene,
                    allow_one_repair=False,
                    allow_virtual_entities=True,
                )

                with self.assertRaisesRegex(SkillPlanGroundingError, message):
                    grounded.create_plan_with_context(_context(scene))

                self.assertEqual(grounded.consume_world_patch(), {})

    def test_open_world_rejects_planner_supplied_motor_parameters(self):
        grounded = SceneSkillPlanner(
            _StaticPlanner(
                [
                    _step(
                        1,
                        "pick",
                        "小球",
                        parameters={"position_xyz_m": [0.4, 0.1, 0.2]},
                    )
                ]
            ),
            EMPTY_SCENE,
            allow_one_repair=False,
            allow_virtual_entities=True,
        )

        with self.assertRaisesRegex(
            SkillPlanGroundingError,
            "cannot contain motor-space field",
        ):
            grounded.create_plan_with_context(_context(EMPTY_SCENE))

        self.assertEqual(grounded.consume_world_patch(), {})


if __name__ == "__main__":
    unittest.main()
