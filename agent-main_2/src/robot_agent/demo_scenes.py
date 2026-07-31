from __future__ import annotations

import json
from typing import Any


class SceneSelectionError(ValueError):
    pass


def _copy(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


# This is a world description, not a collection of task recipes.  The Planner
# selects and orders reusable affordances according to each user request.
DEMO_WORKSPACE_SCENE: dict[str, Any] = {
    "scene": {
        "source": "simulated_semantic_workspace",
        "scene_id": "demo_kitchen_workspace",
        "task_family": "general_robot_manipulation",
        "coordinate_frame": "robot_base",
        "units": "meters",
        "planner_notes": [
            "Infer a new plan from the current user task; there is no required task workflow.",
            "Use affordance_id for move_to_pose and at_affordance_id for gripper actions.",
            "Use motion_id for move_linear.",
            "Meet affordance and motion preconditions using other available contracts.",
            "A fixed handle must be released before the task can finish.",
        ],
        "goal_predicates": [
            {
                "path": "objects.drawer.state",
                "allowed_values": ["open", "closed"],
            },
            {
                "path": "runtime.holding",
                "allowed_values": [None, "brush"],
            },
            {
                "path": "runtime.gripper",
                "allowed_values": ["open", "closed"],
            },
            {
                "path": "runtime.current_affordance_id",
                "allowed_values": [
                    "home",
                    "drawer_handle_closed",
                    "drawer_handle_open",
                    "brush_grasp",
                    "brush_lifted",
                    "sauce_loading",
                    "pizza_sauce_waypoint_1",
                    "pizza_sauce_waypoint_2",
                    "pizza_sauce_waypoint_3",
                    "pizza_sauce_waypoint_4",
                    "tool_rack_place",
                ],
            },
            {
                "path": "objects.brush.location",
                "allowed_values": [
                    "drawer",
                    "held_inside_drawer",
                    "held_outside_drawer",
                    "tool_rack",
                ],
            },
            {
                "path": "runtime.held_state.loaded_with",
                "allowed_values": ["sauce"],
            },
            {
                "path": "objects.pizza.sauce_applied",
                "allowed_values": [True, False],
            },
            {
                "path": "runtime.events.drawer_opened",
                "allowed_values": [True, False],
            },
            {
                "path": "runtime.events.drawer_closed",
                "allowed_values": [True, False],
            },
        ],
        "affordances": [
            {
                "affordance_id": "home",
                "kind": "safe_pose",
                "entity": "robot",
                "parameters": {
                    "position_xyz_m": [0.30, 0.00, 0.35],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "drawer_handle_closed",
                "kind": "grasp_pose",
                "entity": "drawer",
                "grasp_entity": "drawer_handle",
                "grasp_movable": False,
                "requires_gripper": "open",
                "requires": [
                    {
                        "path": "objects.drawer.state",
                        "operator": "eq",
                        "value": "closed",
                    }
                ],
                "parameters": {
                    "position_xyz_m": [0.45, 0.10, 0.22],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "drawer_handle_open",
                "kind": "grasp_pose",
                "entity": "drawer",
                "grasp_entity": "drawer_handle",
                "grasp_movable": False,
                "requires_gripper": "open",
                "requires": [
                    {
                        "path": "objects.drawer.state",
                        "operator": "eq",
                        "value": "open",
                    }
                ],
                "parameters": {
                    "position_xyz_m": [0.25, 0.10, 0.22],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "brush_grasp",
                "kind": "grasp_pose",
                "entity": "brush",
                "grasp_entity": "brush",
                "grasp_movable": True,
                "requires_gripper": "open",
                "requires": [
                    {
                        "path": "objects.drawer.state",
                        "operator": "eq",
                        "value": "open",
                    },
                    {
                        "path": "objects.brush.location",
                        "operator": "eq",
                        "value": "drawer",
                    },
                ],
                "grasp_effects": [
                    {
                        "path": "objects.brush.location",
                        "operation": "set",
                        "value": "held_inside_drawer",
                    }
                ],
                "parameters": {
                    "position_xyz_m": [0.48, 0.10, 0.14],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "brush_lifted",
                "kind": "transit_pose",
                "entity": "brush",
                "requires_holding": "brush",
                "parameters": {
                    "position_xyz_m": [0.48, 0.10, 0.24],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "sauce_loading",
                "kind": "tool_contact",
                "entity": "sauce",
                "requires_holding": "brush",
                "requires": [
                    {
                        "path": "objects.brush.location",
                        "operator": "eq",
                        "value": "held_outside_drawer",
                    }
                ],
                "arrival_effects": [
                    {
                        "path": "runtime.held_state.loaded_with",
                        "operation": "set",
                        "value": "sauce",
                    }
                ],
                "parameters": {
                    "position_xyz_m": [0.40, -0.05, 0.13],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "pizza_sauce_waypoint_1",
                "kind": "path_waypoint",
                "entity": "pizza",
                "requires_holding": "brush",
                "requires_held_state": {"loaded_with": "sauce"},
                "sequence_group": "apply_sauce_to_pizza",
                "sequence_index": 1,
                "parameters": {
                    "position_xyz_m": [0.44, 0.06, 0.12],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "pizza_sauce_waypoint_2",
                "kind": "path_waypoint",
                "entity": "pizza",
                "requires_holding": "brush",
                "requires_held_state": {"loaded_with": "sauce"},
                "sequence_group": "apply_sauce_to_pizza",
                "sequence_index": 2,
                "parameters": {
                    "position_xyz_m": [0.56, 0.06, 0.12],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "pizza_sauce_waypoint_3",
                "kind": "path_waypoint",
                "entity": "pizza",
                "requires_holding": "brush",
                "requires_held_state": {"loaded_with": "sauce"},
                "sequence_group": "apply_sauce_to_pizza",
                "sequence_index": 3,
                "parameters": {
                    "position_xyz_m": [0.56, 0.18, 0.12],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "pizza_sauce_waypoint_4",
                "kind": "path_waypoint",
                "entity": "pizza",
                "requires_holding": "brush",
                "requires_held_state": {"loaded_with": "sauce"},
                "sequence_group": "apply_sauce_to_pizza",
                "sequence_index": 4,
                "arrival_effects": [
                    {
                        "path": "objects.pizza.sauce_applied",
                        "operation": "set",
                        "value": True,
                    }
                ],
                "parameters": {
                    "position_xyz_m": [0.44, 0.18, 0.12],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "affordance_id": "tool_rack_place",
                "kind": "placement_pose",
                "entity": "tool_rack",
                "requires_holding": "brush",
                "accepts_release": ["brush"],
                "release_effects": [
                    {
                        "path": "objects.brush.location",
                        "operation": "set",
                        "value": "tool_rack",
                    }
                ],
                "parameters": {
                    "position_xyz_m": [0.22, -0.20, 0.20],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "coordinate_frame": "robot_base",
                },
            },
        ],
        "motions": [
            {
                "motion_id": "open_drawer",
                "entity": "drawer",
                "from_affordance_id": "drawer_handle_closed",
                "requires_holding": "drawer_handle",
                "requires": [
                    {
                        "path": "objects.drawer.state",
                        "operator": "eq",
                        "value": "closed",
                    }
                ],
                "result_affordance_id": "drawer_handle_open",
                "effects": [
                    {
                        "path": "objects.drawer.state",
                        "operation": "set",
                        "value": "open",
                    },
                    {
                        "path": "runtime.events.drawer_opened",
                        "operation": "set",
                        "value": True,
                    },
                ],
                "parameters": {
                    "delta_xyz_m": [-0.20, 0.0, 0.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "motion_id": "close_drawer",
                "entity": "drawer",
                "from_affordance_id": "drawer_handle_open",
                "requires_holding": "drawer_handle",
                "requires": [
                    {
                        "path": "objects.drawer.state",
                        "operator": "eq",
                        "value": "open",
                    }
                ],
                "result_affordance_id": "drawer_handle_closed",
                "effects": [
                    {
                        "path": "objects.drawer.state",
                        "operation": "set",
                        "value": "closed",
                    },
                    {
                        "path": "runtime.events.drawer_closed",
                        "operation": "set",
                        "value": True,
                    },
                ],
                "parameters": {
                    "delta_xyz_m": [0.20, 0.0, 0.0],
                    "coordinate_frame": "robot_base",
                },
            },
            {
                "motion_id": "lift_brush_from_drawer",
                "entity": "brush",
                "from_affordance_id": "brush_grasp",
                "requires_holding": "brush",
                "requires": [
                    {
                        "path": "objects.brush.location",
                        "operator": "eq",
                        "value": "held_inside_drawer",
                    }
                ],
                "result_affordance_id": "brush_lifted",
                "effects": [
                    {
                        "path": "objects.brush.location",
                        "operation": "set",
                        "value": "held_outside_drawer",
                    }
                ],
                "parameters": {
                    "delta_xyz_m": [0.0, 0.0, 0.10],
                    "coordinate_frame": "robot_base",
                },
            },
        ],
    },
    "runtime": {
        "current_affordance_id": "home",
        "cartesian_position_xyz_m": [0.30, 0.00, 0.35],
        "gripper": "unknown",
        "holding": None,
        "held_state": {},
        "sequence_progress": {},
        "events": {
            "drawer_opened": False,
            "drawer_closed": False,
        },
    },
    "robot": {
        "aliases": ["robot", "机械臂", "机器人", "gripper", "夹爪"],
        "home_pose": {
            "position_xyz_m": [0.30, 0.00, 0.35],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    },
    "objects": {
        "drawer": {
            "type": "container",
            "aliases": ["drawer", "抽屉", "drawer_handle", "抽屉把手"],
            "state": "closed",
        },
        "brush": {
            "type": "tool",
            "aliases": ["brush", "刷子", "sauce_brush", "酱刷"],
            "location": "drawer",
            "contained_in": "drawer",
        },
        "sauce": {
            "type": "material",
            "aliases": ["sauce", "酱", "酱料", "番茄酱", "sauce_bowl"],
        },
        "pizza": {
            "type": "workpiece",
            "aliases": ["pizza", "披萨", "pizza_surface", "披萨表面"],
            "sauce_applied": False,
        },
        "tool_rack": {
            "type": "placement_target",
            "aliases": ["tool_rack", "工具架", "刷子架", "架子"],
        },
    },
}

# Reusable semantic skill contracts.  These describe what one registered skill
# means in this workspace; they are not a stored workflow for any user task.
# Qwen must compose the contracts needed by the current goal.
DEMO_WORKSPACE_SCENE["scene"]["skill_contracts"] = [
    {
        "contract_id": "open_drawer",
        "action_type": "manipulate",
        "target": "drawer",
        "parameters_match": {"operation": "open"},
        "preconditions": [
            {"path": "objects.drawer.state", "operator": "eq", "value": "closed"},
            {"path": "runtime.holding", "operator": "eq", "value": None},
        ],
        "effects": [
            {
                "path": "objects.drawer.state",
                "operation": "set",
                "value": "open",
            }
        ],
    },
    {
        "contract_id": "close_drawer",
        "action_type": "manipulate",
        "target": "drawer",
        "parameters_match": {"operation": "close"},
        "preconditions": [
            {"path": "objects.drawer.state", "operator": "eq", "value": "open"},
            {"path": "runtime.holding", "operator": "eq", "value": None},
        ],
        "effects": [
            {
                "path": "objects.drawer.state",
                "operation": "set",
                "value": "closed",
            }
        ],
    },
    {
        "contract_id": "pick_brush",
        "action_type": "pick",
        "target": "brush",
        "parameters_match": {},
        "preconditions": [
            {"path": "objects.drawer.state", "operator": "eq", "value": "open"},
            {"path": "runtime.holding", "operator": "eq", "value": None},
        ],
        "effects": [
            {"path": "runtime.holding", "operation": "set", "value": "brush"},
            {
                "path": "objects.brush.location",
                "operation": "set",
                "value": "held_outside_drawer",
            },
        ],
    },
    {
        "contract_id": "place_brush_on_tool_rack",
        "action_type": "place",
        "target": "tool_rack",
        "parameters_match": {},
        "preconditions": [
            {"path": "runtime.holding", "operator": "eq", "value": "brush"}
        ],
        "effects": [
            {"path": "runtime.holding", "operation": "set", "value": None},
            {
                "path": "objects.brush.location",
                "operation": "set",
                "value": "tool_rack",
            },
        ],
    },
    {
        "contract_id": "load_brush_with_sauce",
        "action_type": "manipulate",
        "target": "sauce",
        "parameters_match": {"operation": "load_tool"},
        "preconditions": [
            {"path": "runtime.holding", "operator": "eq", "value": "brush"}
        ],
        "effects": [
            {
                "path": "runtime.held_state.loaded_with",
                "operation": "set",
                "value": "sauce",
            }
        ],
    },
    {
        "contract_id": "spread_sauce_on_pizza",
        "action_type": "manipulate",
        "target": "pizza",
        "parameters_match": {"operation": "spread_sauce"},
        "preconditions": [
            {"path": "runtime.holding", "operator": "eq", "value": "brush"},
            {
                "path": "runtime.held_state.loaded_with",
                "operator": "eq",
                "value": "sauce",
            },
        ],
        "effects": [
            {
                "path": "objects.pizza.sauce_applied",
                "operation": "set",
                "value": True,
            }
        ],
    },
]


# Backward-compatible names now point to the same general workspace.  They no
# longer contain task-specific required_workflow lists.
DRAWER_BRUSH_SCENE = _copy(DEMO_WORKSPACE_SCENE)
PIZZA_SAUCE_SCENE = _copy(DEMO_WORKSPACE_SCENE)
DRAWER_BRUSH_PIZZA_SCENE = _copy(DEMO_WORKSPACE_SCENE)


def select_demo_scene(task: str) -> dict[str, Any]:
    normalized = task.lower()
    aliases = [
        str(alias).lower()
        for item in DEMO_WORKSPACE_SCENE["objects"].values()
        for alias in item.get("aliases", [])
    ]
    aliases.extend(
        str(alias).lower()
        for alias in DEMO_WORKSPACE_SCENE["robot"].get("aliases", [])
    )
    aliases.extend(["回到初始位", "回到原位", "home"])
    if any(alias in normalized for alias in aliases):
        return _copy(DEMO_WORKSPACE_SCENE)
    raise SceneSelectionError(
        "No entity in the built-in workspace matches this task. Supply --scene-file "
        "with objects, reusable affordances, motions, and allowed goal predicates."
    )
