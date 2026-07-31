from __future__ import annotations

from .events import record_event
from .motion import parse_relative_motion_target
from .observation import ObservationSource, build_structured_observation, summarize_observation
from .state import RobotState


def build_pi05_input(
    state: RobotState,
    observation_source: ObservationSource | None = None,
) -> dict:
    """Translate a scheduled skill into Pi0.5 language and observation input."""
    step = state["current_step"]
    if step["skill"] == "move_relative":
        motion = parse_relative_motion_target(step["target"])
        task_text = f"Move the arm {motion.direction} by {motion.distance_cm:g} centimeters."
    else:
        prompts = {
            "pick": f"Pick up the {step['target']}.",
            "place": f"Place the held object onto the {step['target']}.",
            "move_home": "Move the arm to its safe home pose.",
            "inspect": f"Inspect whether the {step['target']} is in the expected state.",
            "open_gripper": "Open the gripper.",
            "close_gripper": "Close the gripper.",
        }
        task_text = prompts[step["skill"]]
    observation = build_structured_observation(step, task_text, source=observation_source)
    update: dict = {"pi05_task_text": task_text, "pi05_observation": observation}
    update.update(
        record_event(
            state,
            "pi05.input_prepared",
            f"Prepared Pi05 observation: task='{task_text}' "
            f"state_dim={len(observation['state'])} cameras={sorted(observation['images'])}",
            step_id=step["id"],
            data={"observation_summary": summarize_observation(observation)},
        )
    )
    return update
