from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import patch

from robot_agent.action_executors import DryRunRobotController, PolicyRegistry
from robot_agent.capabilities import CapabilityRegistry, default_capabilities
from robot_agent.demo_scenes import DEMO_WORKSPACE_SCENE, select_demo_scene
from robot_agent.grounding import (
    PlanGroundingError,
    SceneGoalTaskVerifier,
    SceneGroundedPlanner,
)
from robot_agent.planner import PlannerServiceError, QwenApiPlanner
from robot_agent.runtime import build_agent_runtime


SCENE = DEMO_WORKSPACE_SCENE


class MockQwenPlanner(QwenApiPlanner):
    def __init__(
        self,
        plan,
        goal,
        *,
        review_plans: bool = False,
        reviewed_document: dict | None = None,
    ):
        robot_capabilities = tuple(
            item for item in default_capabilities() if item.executor == "robot"
        )
        super().__init__(
            api_key="test-key",
            endpoint="https://example.invalid/chat",
            allowed_skills=tuple(item.action_type for item in robot_capabilities),
            review_plans=review_plans,
        )
        self.documents = [{"goal": goal, "steps": plan}]
        if reviewed_document is not None:
            self.documents.append(reviewed_document)
        self.calls = 0
        self.last_payload = None

    def _post_json(self, payload):
        self.last_payload = payload
        document = self.documents[min(self.calls, len(self.documents) - 1)]
        self.calls += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(document)
                    }
                }
            ]
        }


class DynamicQwenPlanningTests(unittest.TestCase):
    def test_qwen_api_falls_back_to_json_object_and_caches_compatible_mode(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "goal": {
                                    "description": "the bottle is held",
                                    "conditions": [],
                                },
                                "steps": [
                                    {
                                        "step_id": 1,
                                        "action_type": "pick",
                                        "target": "bottle",
                                        "expected_result": "the bottle is held",
                                        "status": "pending",
                                        "parameters": {},
                                    }
                                ],
                            }
                        )
                    }
                }
            ]
        }
        planner = QwenApiPlanner(
            api_key="offline-test-key",
            allowed_skills=("pick",),
            review_plans=False,
        )

        with patch.object(
            planner,
            "_post_json",
            side_effect=[
                PlannerServiceError("Qwen API request failed with HTTP 400."),
                response,
                response,
            ],
        ) as post:
            first_plan = planner.create_plan("Pick up the bottle.")
            second_plan = planner.create_plan("Pick up the bottle again.")

        self.assertEqual(first_plan[0]["target"], "bottle")
        self.assertEqual(second_plan[0]["target"], "bottle")
        payloads = [call.args[0] for call in post.call_args_list]
        self.assertEqual(payloads[0]["response_format"]["type"], "json_schema")
        self.assertEqual(payloads[1]["response_format"], {"type": "json_object"})
        self.assertEqual(payloads[2]["response_format"], {"type": "json_object"})

    def test_different_tasks_share_one_workspace_without_task_workflows(self):
        tasks = [
            "打开抽屉",
            "从抽屉中拿刷子",
            "把刷子放到工具架",
            "从抽屉中拿刷子给披萨刷酱",
        ]

        scenes = [select_demo_scene(task) for task in tasks]

        self.assertEqual(
            {scene["scene"]["scene_id"] for scene in scenes},
            {"demo_kitchen_workspace"},
        )
        self.assertNotIn("required_workflow", scenes[0]["scene"])
        self.assertGreater(len(scenes[0]["scene"]["affordances"]), 1)
        self.assertGreater(len(scenes[0]["scene"]["motions"]), 1)

    def test_open_drawer_task_uses_only_needed_affordances(self):
        result, _ = _run_grounded(
            "打开抽屉",
            _open_drawer_plan(),
            _goal("抽屉处于打开状态", ("objects.drawer.state", "open")),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.memory.completed_steps), 5)
        values = result.memory.world_state["values"]
        self.assertEqual(values["objects"]["drawer"]["state"], "open")
        self.assertIsNone(values["runtime"]["holding"])
        self.assertEqual(
            result.memory.task_verification["verification_scope"],
            "symbolic_goal",
        )

    def test_already_satisfied_goal_completes_without_invented_motion(self):
        result, controller = _run_grounded(
            "保持抽屉关闭",
            [],
            _goal("抽屉保持关闭", ("objects.drawer.state", "closed")),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.memory.completed_steps, [])
        self.assertEqual(controller.get_state()["pose"], "home")
        self.assertEqual(
            result.memory.task_verification["verification_scope"],
            "symbolic_goal",
        )

    def test_take_brush_task_is_dynamically_longer_than_open_drawer(self):
        result, controller = _run_grounded(
            "从抽屉中拿出刷子",
            _take_brush_plan(),
            _goal(
                "刷子已从抽屉取出并由夹爪持有",
                ("runtime.holding", "brush"),
                ("objects.brush.location", "held_outside_drawer"),
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.memory.completed_steps), 8)
        values = result.memory.world_state["values"]
        self.assertEqual(values["runtime"]["holding"], "brush")
        self.assertEqual(
            values["objects"]["brush"]["location"],
            "held_outside_drawer",
        )
        for actual, expected in zip(
            controller.get_state()["cartesian_pose"]["position_xyz_m"],
            [0.48, 0.10, 0.24],
        ):
            self.assertAlmostEqual(actual, expected)

    def test_place_brush_task_composes_retrieve_and_place_affordances(self):
        result, _ = _run_grounded(
            "把抽屉里的刷子放到工具架",
            _place_brush_on_rack_plan(),
            _goal(
                "刷子位于工具架且夹爪已释放",
                ("objects.brush.location", "tool_rack"),
                ("runtime.holding", None),
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.memory.completed_steps), 11)
        values = result.memory.world_state["values"]
        self.assertEqual(values["objects"]["brush"]["location"], "tool_rack")
        self.assertEqual(values["runtime"]["gripper"], "open")

    def test_spread_sauce_task_composes_all_relevant_affordances(self):
        result, _ = _run_grounded(
            "从抽屉中拿刷子给披萨刷酱",
            _spread_sauce_plan(),
            _goal(
                "披萨表面完成刷酱",
                ("objects.pizza.sauce_applied", True),
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.memory.completed_steps), 13)
        values = result.memory.world_state["values"]
        self.assertTrue(values["objects"]["pizza"]["sauce_applied"])
        self.assertEqual(
            values["runtime"]["sequence_progress"]["apply_sauce_to_pizza"],
            4,
        )

    def test_brush_cannot_be_accessed_while_drawer_is_closed(self):
        plan = [
            _gripper_step(1, "open_gripper", "home"),
            _move_step(2, "brush_grasp"),
            _gripper_step(3, "close_gripper", "brush_grasp"),
        ]
        grounded = SceneGroundedPlanner(
            MockQwenPlanner(
                plan,
                _goal("夹住刷子", ("runtime.holding", "brush")),
            ),
            SCENE,
        )

        with self.assertRaisesRegex(PlanGroundingError, "drawer.state"):
            grounded.create_plan_with_context(_context("拿刷子"))

    def test_gripper_must_be_open_before_entering_a_grasp_pose(self):
        plan = [
            _move_step(1, "drawer_handle_closed"),
            _gripper_step(2, "close_gripper", "drawer_handle_closed"),
        ]
        grounded = SceneGroundedPlanner(
            MockQwenPlanner(
                plan,
                _goal("打开抽屉", ("objects.drawer.state", "open")),
            ),
            SCENE,
        )

        with self.assertRaisesRegex(PlanGroundingError, "requires gripper='open'"):
            grounded.create_plan_with_context(_context("打开抽屉"))

    def test_fixed_handle_must_be_released_before_task_completion(self):
        plan = _open_drawer_plan()[:-1]
        grounded = SceneGroundedPlanner(
            MockQwenPlanner(
                plan,
                _goal("抽屉打开", ("objects.drawer.state", "open")),
            ),
            SCENE,
        )

        with self.assertRaisesRegex(PlanGroundingError, "still holding fixed"):
            grounded.create_plan_with_context(_context("打开抽屉"))

    def test_interaction_sequence_is_checked_without_a_task_recipe(self):
        plan = _spread_sauce_plan()
        plan[9], plan[10] = plan[10], plan[9]
        grounded = SceneGroundedPlanner(
            MockQwenPlanner(
                plan,
                _goal(
                    "披萨表面完成刷酱",
                    ("objects.pizza.sauce_applied", True),
                ),
            ),
            SCENE,
        )

        with self.assertRaisesRegex(PlanGroundingError, "next valid item is 1"):
            grounded.create_plan_with_context(
                _context("从抽屉中拿刷子给披萨刷酱")
            )

    def test_llm_supplied_effects_are_ignored(self):
        plan = _open_drawer_plan()
        plan[0]["effects"] = [
            {
                "path": "objects.pizza.sauce_applied",
                "operation": "set",
                "value": True,
            }
        ]
        grounded = SceneGroundedPlanner(
            MockQwenPlanner(
                plan,
                _goal(
                    "披萨表面完成刷酱",
                    ("objects.pizza.sauce_applied", True),
                ),
            ),
            SCENE,
        )

        with self.assertRaisesRegex(PlanGroundingError, "does not achieve"):
            grounded.create_plan_with_context(_context("打开抽屉"))

    def test_invalid_goal_predicate_is_rejected(self):
        grounded = SceneGroundedPlanner(
            MockQwenPlanner(
                _open_drawer_plan(),
                _goal("伪造目标", ("objects.drawer.magic", True)),
            ),
            SCENE,
        )

        with self.assertRaisesRegex(PlanGroundingError, "not an allowed"):
            grounded.create_plan_with_context(_context("打开抽屉"))

    def test_replanning_validates_suffix_from_current_world_state(self):
        goal = _goal(
            "刷子已取出",
            ("runtime.holding", "brush"),
            ("objects.brush.location", "held_outside_drawer"),
        )
        current_scene = copy.deepcopy(SCENE)
        current_scene["objects"]["drawer"]["state"] = "open"
        current_scene["runtime"].update(
            {
                "current_affordance_id": "drawer_handle_open",
                "cartesian_position_xyz_m": [0.25, 0.10, 0.22],
                "gripper": "open",
                "holding": None,
            }
        )
        current_scene["_agent"] = {"task_goal": goal}
        suffix = [
            _move_step(6, "brush_grasp"),
            _gripper_step(7, "close_gripper", "brush_grasp"),
            _motion_step(8, "lift_brush_from_drawer"),
        ]
        grounded = SceneGroundedPlanner(
            MockQwenPlanner(suffix, goal),
            SCENE,
        )
        context = _context("从抽屉中拿出刷子", current_scene)

        replacement = grounded.revise_from_failure(context)

        self.assertEqual(len(replacement), 3)
        self.assertEqual(
            replacement[0]["parameters"]["affordance_id"],
            "brush_grasp",
        )

    def test_replanning_cannot_change_persisted_goal(self):
        original_goal = _goal(
            "刷子已取出",
            ("runtime.holding", "brush"),
        )
        changed_goal = _goal(
            "披萨已刷酱",
            ("objects.pizza.sauce_applied", True),
        )
        current_scene = copy.deepcopy(SCENE)
        current_scene["_agent"] = {"task_goal": original_goal}
        grounded = SceneGroundedPlanner(
            MockQwenPlanner(_spread_sauce_plan(), changed_goal),
            SCENE,
        )

        with self.assertRaisesRegex(PlanGroundingError, "changed the original"):
            grounded.revise_from_failure(
                _context("从抽屉中拿出刷子", current_scene)
            )

    def test_qwen_review_pass_can_replace_a_bad_draft(self):
        final_document = {
            "goal": _goal(
                "抽屉处于打开状态",
                ("objects.drawer.state", "open"),
            ),
            "steps": _open_drawer_plan(),
        }
        planner = MockQwenPlanner(
            [
                _gripper_step(1, "open_gripper", "home"),
                _move_step(2, "brush_grasp"),
            ],
            _goal("夹住刷子", ("runtime.holding", "brush")),
            review_plans=True,
            reviewed_document=final_document,
        )
        grounded = SceneGroundedPlanner(planner, SCENE)

        plan = grounded.create_plan_with_context(_context("打开抽屉"))

        self.assertEqual(len(plan), 5)
        self.assertEqual(planner.calls, 2)
        self.assertIn(
            "Independently review",
            planner.last_payload["messages"][1]["content"],
        )

    def test_local_validation_error_can_trigger_one_qwen_repair(self):
        repaired_document = {
            "goal": _goal(
                "抽屉处于打开状态",
                ("objects.drawer.state", "open"),
            ),
            "steps": _open_drawer_plan(),
        }
        planner = MockQwenPlanner(
            [
                _gripper_step(1, "open_gripper", "home"),
                _move_step(2, "brush_grasp"),
            ],
            _goal("抽屉处于打开状态", ("objects.drawer.state", "open")),
            reviewed_document=repaired_document,
        )
        grounded = SceneGroundedPlanner(planner, SCENE)

        plan = grounded.create_plan_with_context(_context("打开抽屉"))

        self.assertEqual(len(plan), 5)
        self.assertEqual(planner.calls, 2)
        self.assertIn(
            "trusted local semantic validator rejected",
            planner.last_payload["messages"][1]["content"],
        )


def _run_grounded(task: str, plan, goal):
    planner = MockQwenPlanner(plan, goal)
    grounded = SceneGroundedPlanner(planner, SCENE)
    capabilities = CapabilityRegistry(
        item for item in default_capabilities() if item.executor == "robot"
    )
    controller = DryRunRobotController()
    runtime = build_agent_runtime(
        grounded,
        controller=controller,
        policies=PolicyRegistry(),
        dry_run=False,
        capabilities=capabilities,
        initial_world_state=SCENE,
        task_verifier=SceneGoalTaskVerifier(SCENE),
    )
    return runtime.agent.run_safe(task), controller


def _context(task: str, scene=None):
    return {
        "original_task": task,
        "world_state": {
            "version": 0,
            "values": copy.deepcopy(scene or SCENE),
        },
        "capabilities": [],
        "available_policies": [],
    }


def _goal(description: str, *conditions):
    return {
        "description": description,
        "conditions": [
            {"path": path, "operator": "eq", "value": value}
            for path, value in conditions
        ],
    }


def _open_drawer_plan():
    return [
        _gripper_step(1, "open_gripper", "home"),
        _move_step(2, "drawer_handle_closed"),
        _gripper_step(3, "close_gripper", "drawer_handle_closed"),
        _motion_step(4, "open_drawer"),
        _gripper_step(5, "open_gripper", "drawer_handle_open"),
    ]


def _take_brush_plan():
    plan = _open_drawer_plan()
    plan.extend(
        [
            _move_step(6, "brush_grasp"),
            _gripper_step(7, "close_gripper", "brush_grasp"),
            _motion_step(8, "lift_brush_from_drawer"),
        ]
    )
    return plan


def _place_brush_on_rack_plan():
    plan = _take_brush_plan()
    plan.extend(
        [
            _move_step(9, "tool_rack_approach"),
            _move_step(10, "tool_rack_place"),
            _gripper_step(11, "open_gripper", "tool_rack_place"),
        ]
    )
    return plan


def _spread_sauce_plan():
    plan = _take_brush_plan()
    for affordance_id in (
        "sauce_loading",
        "pizza_sauce_waypoint_1",
        "pizza_sauce_waypoint_2",
        "pizza_sauce_waypoint_3",
        "pizza_sauce_waypoint_4",
    ):
        plan.append(_move_step(len(plan) + 1, affordance_id))
    return plan


def _move_step(step_id: int, affordance_id: str):
    affordance = _affordance(affordance_id)
    return {
        "step_id": step_id,
        "action_type": "move_to_pose",
        "target": affordance_id,
        "executor": "robot",
        "expected_result": f"robot reaches {affordance_id}",
        "status": "pending",
        "parameters": {
            "affordance_id": affordance_id,
            **copy.deepcopy(affordance["parameters"]),
        },
    }


def _motion_step(step_id: int, motion_id: str):
    motion = _motion(motion_id)
    return {
        "step_id": step_id,
        "action_type": "move_linear",
        "target": motion_id,
        "executor": "robot",
        "expected_result": f"motion {motion_id} completes",
        "status": "pending",
        "parameters": {
            "motion_id": motion_id,
            **copy.deepcopy(motion["parameters"]),
        },
    }


def _gripper_step(step_id: int, action_type: str, affordance_id: str):
    affordance = _affordance(affordance_id)
    return {
        "step_id": step_id,
        "action_type": action_type,
        "target": "gripper",
        "executor": "robot",
        "expected_result": f"{action_type} completes at {affordance_id}",
        "status": "pending",
        "parameters": {
            "at_affordance_id": affordance_id,
            "at_position_xyz_m": copy.deepcopy(
                affordance["parameters"]["position_xyz_m"]
            ),
            "coordinate_frame": affordance["parameters"]["coordinate_frame"],
        },
    }


def _affordance(affordance_id: str):
    return next(
        item
        for item in SCENE["scene"]["affordances"]
        if item["affordance_id"] == affordance_id
    )


def _motion(motion_id: str):
    return next(
        item
        for item in SCENE["scene"]["motions"]
        if item["motion_id"] == motion_id
    )


if __name__ == "__main__":
    unittest.main()
