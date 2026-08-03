"""Verify the LIBERO bridge without a policy model or physical robot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
SOURCE_ROOT = WORKSPACE / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robot_agent.libero_integration import (
    LIBERO_ACTION_DIM,
    LiberoActionChunkController,
    LiberoHttpClient,
    LiberoPerceptionProvider,
    validate_libero_action_schema,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reset LIBERO, read observations, and send bounded zero actions."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8770")
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--zero-steps", type=int, default=5)
    parser.add_argument("--skip-reset", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.zero_steps <= 100:
        print("--zero-steps must be between 0 and 100.")
        return 2
    client = LiberoHttpClient(args.endpoint)
    try:
        health = client.health()
        validate_libero_action_schema(health)
        if not args.skip_reset:
            client.reset(args.init_state_index)
        perception = LiberoPerceptionProvider(client)
        observation = perception.observe()
        action_result = None
        if args.zero_steps:
            controller = LiberoActionChunkController(
                client,
                action_schema_confirmed=True,
            )
            action_result = controller.execute_action_chunk(
                [[0.0] * LIBERO_ACTION_DIM for _ in range(args.zero_steps)]
            )
            observation = perception.observe()
    except Exception as error:
        print(f"LIBERO smoke test failed: {error}")
        return 1

    images = dict(observation.get("images") or {})
    print("LIBERO bridge smoke test passed.")
    print(f"Task: {health.get('task_language')}")
    print(f"State dimension: {len(observation.get('state') or [])}")
    print(
        "Encoded cameras: "
        + ", ".join(
            f"{name}={len(value)} chars" for name, value in sorted(images.items())
        )
    )
    if action_result is not None:
        print(
            "Zero-action result: "
            f"{action_result.get('reason')} "
            f"({action_result.get('steps_executed')}/{args.zero_steps} steps)"
        )
    print(f"LIBERO task success: {bool(observation.get('success', False))}")
    print("No policy model and no physical robot were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
