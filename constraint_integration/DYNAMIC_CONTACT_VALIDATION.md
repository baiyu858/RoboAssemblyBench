# Dynamic Allowed-Contact Validation

## Scope

- Recipe: `fabrica_plumbers_block_ur5e_right_base_prepare`
- Seed: `0`
- Runtime check stride: `64`
- Video frame stride: `8`
- Mode: passive and fail-open

## False-Positive Cause

The legacy overlay treated every detector candidate below `0.03 m` as a
collision. At step 8384, the left arm still had positive surface clearance
from blocks 0 and 3, but those proximity candidates were displayed as red
warnings. The first-pass distal capsule radii also produced artificial
inter-arm overlap around steps 8448-8576 even though the source video showed
no physical contact.

## Changes

1. Candidate distance and collision overlap now have separate meanings:
   positive clearance is `proximity`; clearance at or below zero is
   `collision`.
2. Phase context is read from `get_current_phase_spec()`. During grasp,
   insertion, and release, active end-effector contact with the active target
   object is classified as `allowed_contact`.
3. Attached or grasped payloads remain excluded from environment obstacles
   during transport.
4. UR5e/Robotiq distal capsule radii were narrowed:

| Capsule | Before | After |
| --- | ---: | ---: |
| wrist 1 to wrist 2 | 0.045 m | 0.040 m |
| wrist 2 to wrist 3 | 0.045 m | 0.035 m |
| wrist 3 to gripper base | 0.050 m | 0.040 m |
| gripper base to inner finger | 0.025 m | 0.020 m |

Configured symmetric ignore pairs remain available, but no fixed
frame/recipe-specific ignore rule was added for this fix.

## Fixed-Seed Result

| Metric | Legacy run | Optimized run |
| --- | ---: | ---: |
| Assembly success | yes | yes |
| Episode steps | 14,688 | 14,688 |
| Runtime checks | 229 | 229 |
| Detector candidates | 52 | 35 |
| Allowed assembly contacts | not separated | 1 |
| Positive-clearance proximity events | not separated | 34 |
| Abnormal collision events | all 52 were displayed | 0 |
| Minimum candidate clearance | -0.019462 m | +0.000538 m |
| Missing prims / geometry / monitor errors | 0 / 0 / 0 | 0 / 0 / 0 |

At the original false-positive interval:

- Step 8384 contains three positive-clearance object proximity samples. The
  minimum is `0.009532 m`, so none is a collision.
- Step 8448 has a minimum inter-arm clearance of `0.000809 m`.
- Step 8512 has a minimum inter-arm clearance of `0.000538 m`.
- No event from this interval is written to the collision-only `events` list.

## Tests

```text
constraint integration: 37 passed
assembly regression:     12 passed
git diff --check:         passed
```

The generated three-camera videos each contain 1,834 frames at 30 FPS
(61.13 seconds). The overlay reads only the collision-only `events` list, so
this successful episode contains no red collision banner. Proximity and
allowed-contact samples remain in `collect_results.json` for audit.

## Server Artifacts

```text
/data/pxb/outputs/fabrica_plumbers_block_ur5e_dynamic_contacts_seed0_v3/
  collect_results.json
  episode_0000_live_videos/
  episode_0000_annotated_videos/
```
