# Constraint Integration

This directory is an isolated bridge between the current RoboAssemblyBench
assembly pipeline and the experimental `constraint_detection/` project.

All integrations are explicit, passive, and fail-open: they record metrics but
never change actions or task terminal state.

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
|-- contact_policy.py     # phase-aware allowed-contact classification
|-- pipeline.py           # fail-open episode lifecycle adapter
|-- runtime_monitor.py    # passive rollout-time collision monitor
|-- precheck_adapter.py   # robot-agnostic trajectory precheck adapter
|-- precheck_pipeline.py  # sequence/stage precheck episode lifecycle
|-- sequence_precheck.py  # full-recipe symbolic state simulation
|-- stage_precheck.py     # commanded-joint interpolation + UR5e Lula FK
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
- covers the Robotiq outer/inner fingers, knuckles, and palm envelope,
- refreshes unheld box obstacles from `task.get_tracked_object_states()`,
- loads untracked static collider bounds, including fixtures and optical boards,
- checks inter-arm, robot-environment, and optional ground collisions,
- separates proximity candidates, expected assembly contact, and abnormal
  capsule overlap,
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

The detector threshold is a broad candidate distance: positive signed
clearance up to 0.03 m is recorded as `proximity`, but is not a collision.
Surface overlap at or below 0 m is classified as `collision`. During grasp,
insert, and release phases, contact between the active robot end effector and
the phase target object is classified as `allowed_contact`.

The report contains check counts and timings, candidate and classification
counts, abnormal collision counts, minimum signed clearance, detailed
collision events, sampled proximity/allowed-contact audit events, registered
links, missing prim/object diagnostics, and fail-open errors. Each event
includes `step`, `phase`, `kind`, `entity_a`, `entity_b`, `distance`,
`threshold`, `classification`, `classification_reason`, `active_robot`,
`active_object`, `pos_a`, and `pos_b`. Stored audit events are capped while
aggregate counts continue to grow.

The monitor never modifies actions, `terminated`, `success`, `status`, or
`terminal_reason`.

## Passive Prechecks

Two default-off checks run before `env.step()`:

- `AssemblySequencePrechecker` executes the complete `phase_specs` list in a
  symbolic state machine. It validates robot/object references, end-effector
  occupancy, object ownership, attach-before-carry ordering, double grasps,
  and release ownership. It does not step Isaac Sim.
- `StageTrajectoryPrechecker` samples the current-to-commanded UR5e joint
  segment, calls the controller's existing Lula FK for each arm link, and
  reuses `CollisionDetector` for capsule-vs-object, optional ground, and
  dual-arm checks.

Enable both with:

```text
--assembly-sequence-precheck
--stage-trajectory-precheck
--stage-precheck-stride 64
--stage-precheck-waypoints 8
```

Reports are written to `assembly_sequence_precheck` and
`stage_trajectory_precheck`. Exceptions are recorded in `monitor_error` and
the original action is still passed unchanged to `env.step()`.

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

The corrected fixed-seed server validation completed successfully with 229
checks over 14,688 observed steps. It registered 16 links per robot and two
static scene obstacles, with no missing prim, geometry, or monitor errors.
The report contains 9 abnormal inter-arm overlaps around steps 8448-8576, 115
positive-clearance proximity samples, and 3 allowed target contacts. See
`DYNAMIC_CONTACT_VALIDATION.md` for the video and three-camera review.

## Current Status

Runtime monitoring, full-sequence logical precheck, and stage trajectory
precheck are connected behind explicit, default-off flags. None imports
OpenVLA or Transformers, and none enforces a stop or replan.

The fixed seed-0 validation completed the original assembly successfully in
15,014 steps while both prechecks were enabled. The sequence report covered 36
phases and 41 normalized actions with no logical errors. The stage report
performed 235 checks across 470 arm segments and 1,880 synchronized waypoint
sets, recorded 112 passive proximity events, and reported no monitor errors.
