# Constraint Integration

This directory is an isolated bridge between the current RoboAssemblyBench
assembly pipeline and the experimental `constraint_detection/` project.

The demo pipeline imports it only when `--runtime-constraint-monitor` is
enabled. The monitor is passive and fail-open: it records metrics but never
changes actions or task terminal state.

## Purpose

The target integration flow is:

```text
recipe phase
  -> atomic skill resolves a target pose
  -> optional trajectory precheck
  -> original controller executes
  -> optional runtime collision monitor observes rollout steps
  -> violations are logged or later used to fail/replan an episode
```

## Why This Is Separate

`constraint_detection/` originally demonstrates collision checking with
Franka robots. The main reproduction task uses dual UR5e + Robotiq 2F-85.
Directly importing demo assumptions into the main pipeline would mix robot
link names, capsule geometry, controller types, object registration, and
allowed assembly contacts. This directory keeps that adapter explicit and
reversible.

## Files

```text
constraint_integration/
|-- __init__.py
|-- models.py             # configurable Franka / UR5e collision models
|-- pipeline.py           # fail-open episode lifecycle adapter
|-- runtime_monitor.py    # passive rollout-time collision monitor
|-- precheck_adapter.py   # robot-agnostic trajectory precheck adapter
`-- README.md
```

## Runtime Monitor

`RuntimeConstraintMonitor` reuses collision geometry from:

```text
constraint_detection/src/collision.py
```

It replaces the hard-coded Franka model with configurable models from
`models.py` and:

- creates Isaac Sim prim readers lazily,
- supports root-mounted and wrist-mounted Robotiq 2F-85 prim layouts,
- refreshes unheld box obstacles from `task.get_tracked_object_states()`,
- checks self, inter-arm, robot-environment, and optional ground collisions,
- returns JSON-serializable reports,
- records missing prims, missing object geometry, and monitor errors,
- never mutates task state or stops a rollout.

Tracked objects are registered only when metric collision dimensions are
known. USD scale factors are not treated as physical box dimensions.

The runtime hook is in:

```text
toolkits/factory_dual_franka_assembly/generate_demos.py
```

`RuntimeConstraintEpisodeHook` owns one monitor per episode. It observes only
after `env.step()`, attaches the finalized report to the episode metrics, and
resets before the next episode. Initialization, observation, and finalization
all fail open.

### Command-Line Use

The monitor is disabled by default. Enable it with:

```text
--runtime-constraint-monitor
--constraint-check-stride 64
--constraint-threshold 0.03
--constraint-include-ground
--constraint-ignore-pair left_inner_finger:block_0
```

`--constraint-ignore-pair` may be repeated. Matching is symmetric, so `A:B`
also ignores `B:A`. Omitting `--constraint-threshold` uses the UR5e model
default of 0.03 m. The command-line stride default remains 8 to match the
initial interface contract, but the first server rollout was validated with an
explicit stride of 64 because stride 8 perturbed this timing-sensitive task.

The final episode dictionary gains one field only:

```text
runtime_constraint_monitor
```

It contains check counts and timings, aggregate violation counts, violations by
kind, minimum signed clearance, detailed events, registered links, missing
prim/object diagnostics, and fail-open errors. Each event includes `step`,
`kind`, `entity_a`, `entity_b`, `distance`, `threshold`, `pos_a`, and `pos_b`.
The last two fields are the nearest points used by the distance check. Stored
events are capped while aggregate counts continue to grow.

The monitor never modifies actions, `terminated`, `success`, `status`, or
`terminal_reason`.

## Trajectory Precheck

`LinearPosePrechecker` implements an execution-time precheck without assuming a
Franka Lula configuration. It expects callbacks supplied by the skill layer:

```python
report = prechecker.check(
    start_position=current_tcp,
    target_position=target_tcp,
    target_orientation=target_quat,
    solve_ik=solve_ik_callback,
    forward_kinematics=fk_callback,
    warm_start=current_q,
)
```

The future hook point is:

```text
toolkits/factory_dual_franka_assembly/plumbers_block_ur5e_skills.py
```

The first integration intentionally does not call this prechecker from the
rollout path.

## Validation Guidance

1. Record a disabled-monitor baseline for fixed recipe/seed pairs.
2. Enable the monitor with the same recipe, seeds, and scene profile.
3. Confirm both UR5e robots register all expected links.
4. Compare success, terminal reason, phase transitions, steps, and run time.
5. Inspect monitor timing, missing geometry, errors, and static false positives.
6. Calibrate capsule radii, threshold, stride, and allowed contact pairs before
   increasing the batch size.

Collision checks add frame time and may perturb a physics-sensitive rollout
even though the monitor is logically passive. Use the largest stride that
still provides useful coverage and always compare enabled and disabled runs on
the same seeds.

For the first fixed-seed server validation, stride 64 completed successfully
with 229 checks over 14,688 observed steps. Collision computation took 6.35 s
in total (27.7 ms average per check), both robots registered all 10 expected
links, and no prim, object-geometry, or monitor errors were reported.

## Current Status

Runtime monitoring is connected behind an explicit, default-off command-line
flag. It imports only `constraint_detection.src.collision`; it does not import
OpenVLA, Transformers, or the action checker. Trajectory precheck and
enforcement remain disconnected from the rollout path.
