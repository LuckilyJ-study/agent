from __future__ import annotations

from .planner import QwenApiPlanner


def main() -> None:
    planner = QwenApiPlanner()
    plan = planner.create_plan("Make a pizza by picking dough and placing it on a tray.")

    print("Qwen API smoke test passed.")
    print("Planner returned steps:")
    for step in plan:
        print(
            f"- step {step['id']}: {step['skill']}({step['target']})"
            f" -> {step['expected_result']}"
        )


if __name__ == "__main__":
    main()
