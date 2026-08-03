# Constraint Checking Toolkit

`toolkits/constraint_checking` contains the constraint-checking implementation used by
the RoboAssemblyBench dual-UR5e assembly pipeline. The original assembly entry remains:

```text
toolkits/factory_dual_franka_assembly/generate_demos.py
```

All three checks are passive by default: they record reports in episode metrics without
changing actions, termination, success status, or the original assembly policy.

## Three Checks

| Check | Timing | Purpose | Metric key |
| --- | --- | --- | --- |
| Assembly sequence precheck | Once before rollout | Normalize the complete phase list and validate ordering, ownership, grasp/release, and handoff logic | `assembly_sequence_precheck` |
| Stage trajectory precheck | Before sampled `env.step()` calls | Interpolate the current-to-commanded joint segment and check sampled robot configurations | `stage_trajectory_precheck` |
| Runtime collision monitor | After sampled `env.step()` calls | Measure inter-arm and robot-environment clearances and classify allowed contact, proximity, and collision events | `runtime_constraint_monitor` |

The sequence precheck performs a lightweight symbolic state simulation. It does not run a
second physics rollout. The stage precheck reuses the capsule collision geometry to predict
whether sampled configurations along the next commanded segment are safe. The runtime
monitor reads the actual simulated state after execution.

## V2 Collision Model

The V2 model approximates both UR5e + Robotiq 2F-85 robots with link capsules. Environment
and assembly objects are represented by oriented or axis-aligned boxes obtained from task
state and USD bounds. The detector supports:

- self-independent dual-arm checks between registered robot links;
- robot-to-environment and robot-to-assembly-part distance checks;
- optional ground checks;
- signed surface clearance and nearest-point reporting;
- phase-aware contact classification for normal grasp, carry, insert, and release contact;
- symmetric user-defined ignore pairs.

The geometry detector produces candidates. `integration/contact_policy.py` then labels each
candidate as `allowed_contact`, `proximity`, or abnormal `collision`. This separation keeps
normal assembly contact rules out of the low-level geometry implementation.

## Layout

```text
toolkits/constraint_checking/
|-- detector/       # Generic rule, capsule collision, scene, and precheck algorithms
|-- integration/    # RoboAssemblyBench models, policies, hooks, and episode reports
|-- demos/          # Standalone detector demonstrations
|-- docs/           # Validation notes and experiment records
|-- requirements.txt
`-- run.sh
```

The assembly call path is:

```text
generate_demos.py
  -> PassivePrecheckEpisodeHook
       -> AssemblySequencePrechecker
       -> StageTrajectoryPrechecker
  -> env.step()
  -> RuntimeConstraintEpisodeHook
       -> RuntimeConstraintMonitor
       -> AssemblyContactPolicy
       -> CollisionDetector
  -> episode metrics JSON and video overlays
```

## Running the Assembly Entry

Run from the repository root with the configured Isaac Sim Python environment. The project
root must be importable; setting `PYTHONPATH` explicitly also makes `--help` work under a
plain Python process.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ISAAC_SIM_ROOT=/path/to/isaacsim
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python toolkits/factory_dual_franka_assembly/generate_demos.py \
  --recipes fabrica_plumbers_block_ur5e_right_base_prepare \
  --num-demos 1 \
  --start-seed 0 \
  --max-trials 1 \
  --headless \
  --runtime-constraint-monitor \
  --constraint-check-stride 32 \
  --assembly-sequence-precheck \
  --stage-trajectory-precheck \
  --stage-precheck-stride 64 \
  --stage-precheck-waypoints 8 \
  --output-dir outputs/constraint_checking
```

If the Isaac Sim installation requires a launcher that creates `SimulationApp` before the
project script, invoke this same file through that launcher. The entry path and arguments do
not change.

## Constraint Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--runtime-constraint-monitor` | off | Enable passive runtime collision monitoring |
| `--constraint-check-stride N` | `8` | Run one runtime check every N simulation steps |
| `--constraint-threshold M` | model default (`0.03 m`) | Candidate proximity clearance |
| `--constraint-include-ground` | off | Include the ground plane |
| `--constraint-ignore-pair A:B` | none | Ignore a symmetric entity substring pair; repeatable |
| `--assembly-sequence-precheck` | off | Enable complete sequence logic precheck |
| `--stage-trajectory-precheck` | off | Enable sampled stage trajectory precheck |
| `--stage-precheck-stride N` | `64` | Run one stage check every N simulation steps |
| `--stage-precheck-waypoints N` | `8` | Number of interpolated points per checked segment |

All switches default to off, preserving the baseline assembly behavior.

## Episode Metrics

`assembly_sequence_precheck` includes whether the sequence is feasible, phase and normalized
action counts, and `errors`/`warnings`.

`stage_trajectory_precheck` includes observed steps, checks, checked segments and waypoints,
violations by kind, minimum distance, timing, and `monitor_error`.

`runtime_constraint_monitor` includes robot model and thresholds, observed steps, check timing,
candidate and violation totals, classifications, nearest clearances, detailed events,
allowed-contact records, missing prim/object geometry diagnostics, and `monitor_error`.

Reports are JSON serializable. Fail-open errors are recorded in `monitor_error` and do not
change the rollout result.

## Tests

From the repository root:

```bash
python -m pytest -q \
  --confcutdir=tests/toolkits/constraint_checking \
  tests/toolkits/constraint_checking

python -m pytest -q \
  --confcutdir=tests/toolkits \
  tests/toolkits/test_factory_dual_franka_assembly.py

PYTHONPATH="$PWD" python toolkits/factory_dual_franka_assembly/generate_demos.py --help
```

The first suite covers geometry, the UR5e model, contact policy, runtime monitoring, both
prechecks, episode hooks, serialization, fail-open behavior, and worker argument propagation.

## Dependency Boundaries

The integration modules delay Isaac Sim and USD imports until a simulation-backed operation
needs them. Importing the package in ordinary unit tests does not initialize Isaac Sim.
`detector/vla_client.py` is an optional demonstration component and is not imported by the
assembly runtime path, so OpenVLA, Transformers, and their model weights are not required for
the three assembly checks.

## Legacy Compatibility

New code should use:

```python
from toolkits.constraint_checking.integration.pipeline import RuntimeConstraintEpisodeHook
from toolkits.constraint_checking.detector.collision import CollisionDetector
```

The legacy paths remain as thin forwarding modules for existing users:

```python
from constraint_integration.pipeline import RuntimeConstraintEpisodeHook
from constraint_detection.src.collision import CollisionDetector
```

There is only one implementation. Legacy modules and Demo entries forward to
`toolkits/constraint_checking` and are retained temporarily for migration compatibility.

See `docs/dynamic_contact_validation.md` for the V2 dynamic-contact validation record.
