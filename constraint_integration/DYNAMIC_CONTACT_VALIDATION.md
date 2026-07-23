# Dynamic Contact Validation

## Scope

- Recipe: `fabrica_plumbers_block_ur5e_right_base_prepare`
- Seed: `0`
- Runtime check stride: `64`
- Video frame stride: `8`
- Mode: passive and fail-open

## Superseded Conclusion

The first contact-precision pass incorrectly concluded that the close-arm
interval around 35 seconds was a false positive. It narrowed the distal UR5e
capsules until all overlap disappeared. Review of the source video showed real
arm contact, so that calibration and its zero-collision conclusion are
superseded.

## Corrected Model

1. Conservative wrist envelopes are restored:

| Capsule | Radius |
| --- | ---: |
| wrist 1 to wrist 2 | 0.045 m |
| wrist 2 to wrist 3 | 0.045 m |
| wrist 3 to gripper base | 0.050 m |

2. The Robotiq model includes base, inner/outer knuckles, inner/outer fingers,
   and a palm capsule between the two outer-knuckle roots.
3. Untracked collider objects are loaded from the live USD stage. The fixed
   seed run registered `optical_board` and `fabrica_fixture`.
4. Positive surface clearance remains `proximity`; signed clearance at or below
   zero is `collision`.
5. Active end-effector contact with the active target during grasp, insert, and
   release is `allowed_contact`.

## Fixed-Seed Result

| Metric | Corrected run |
| --- | ---: |
| Assembly success | yes |
| Episode steps | 14,688 |
| Runtime checks | 229 |
| Detector candidates | 127 |
| Allowed assembly contacts | 3 |
| Positive-clearance proximity events | 115 |
| Abnormal collision events | 9 |
| Minimum signed clearance | -0.019462 m |
| Monitor compute time | 13.105 s |
| Missing prims / geometry / monitor errors | 0 / 0 / 0 |

All nine abnormal events are inter-arm overlaps at steps 8448, 8512, and 8576
during `left_move_above_block_4`. They cover wrist-to-wrist and
wrist-to-gripper pairs. The first video warning appears at about 35.2 seconds.

## 22-Second Review

The front camera makes the two end effectors appear to overlap around 22
seconds. A diagnostic run with a temporary `0.15 m` candidate threshold found:

- closest inter-arm capsule clearance: `0.061547 m` at step 5376;
- closest active-arm/object clearance: `0.064939 m` to block 0 at step 5312.

The left- and right-wrist cameras confirm that the end effectors are separated
in depth. This interval is therefore not written to the collision-only event
list and receives no red overlay. The wide threshold was diagnostic only; the
runtime default remains `0.03 m`.

## Tests

```text
constraint integration: 39 passed
assembly regression:     12 passed
git diff --check:         passed
```

The three annotated videos contain 1,834 frames each at 30 FPS. Their overlay
reads only the collision event list and shows the pair, phase, step, and overlap
depth.

## Server Artifacts

```text
/data/pxb/outputs/fabrica_plumbers_block_ur5e_contact_precision_v4_seed0/
  collect_results.json
  episode_0000_live_videos/
  episode_0000_annotated_videos/
  inspection_22s/
  annotation_checks/
```
