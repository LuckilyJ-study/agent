# LIBERO integration

The Agent and LIBERO are connected through a small HTTP bridge. This keeps the
legacy LIBERO Python 3.8 / MuJoCo dependencies separate from the Agent process
and ensures that only one owner thread calls the simulator.

## Boundaries

```text
Agent Planner -> policy route -> policy model -> 7-D action chunk
    -> Agent ActionChunkGuard -> LIBERO HTTP bridge -> env.step(action)

LIBERO cached RGB/proprioception -> Agent perception / policy input
LIBERO BDDL success predicate -> Agent final task verifier
```

LIBERO's default `OSC_POSE` action is a normalized seven-dimensional command:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

Do not connect a Pi05 output merely because it also has seven values. Camera
layout, state normalization, coordinate frames, action semantics, gripper sign,
and action scaling must all match or be adapted explicitly.

## Start the bridge

Create the dedicated environment described by `LIBERO-master/README.md`, then
run the service from `agent-main_2`:

```powershell
conda activate libero
python .\libero_bridge_service.py `
  --libero-root "D:\vscode会话\latex格式更改\LIBERO-master" `
  --benchmark libero_object `
  --task-id 0 `
  --init-state-index 0
```

The first task suite should be `libero_object`, because its single-object
pick-and-place tasks are easier to diagnose than `libero_10` multi-stage tasks.

## Verify the environment boundary

In another terminal, from `agent-main_2`:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\run_libero_smoketest.py --zero-steps 5
```

This resets the task, reads both camera views and proprioception, validates the
declared action schema, and sends five bounded zero actions. It does not use a
policy model and cannot complete the benchmark task.

## Policy requirement

This LIBERO checkout contains BDDL tasks and initial states, but no trained
policy checkpoint. Actual task execution requires one of these:

1. A LIBERO-trained policy loaded from a checkpoint and wrapped as an Agent
   `PolicyBackend`.
2. Pi05 plus a verified LIBERO observation/action adapter. Direct raw connection
   is intentionally blocked until its output schema is confirmed.

For the first closed-loop experiment, expose the exact LIBERO task language as
one `manipulate` policy step and let the policy run until `success=true` or a
step budget is exhausted. Skill-level `pick`/`place` decomposition can be added
after the policy can reliably execute those sub-instructions.
