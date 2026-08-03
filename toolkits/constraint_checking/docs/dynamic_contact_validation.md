# Dynamic Contact Validation

## Scope

- Recipe: `fabrica_plumbers_block_ur5e_right_base_prepare`
- Seed: `0`
- Runtime check stride: `32`
- Video frame stride: `8`
- Mode: passive and fail-open

## Superseded Conclusion

Two earlier conclusions are superseded:

1. The first contact-precision pass incorrectly removed the real close-arm
   collision around 35 seconds by narrowing the distal UR5e capsules.
2. A later stride-64 run incorrectly classified the 22-second interval as
   camera occlusion. Frame-by-frame review showed that the passive arm was
   displaced, while stride 64 skipped the brief contact window.

## Corrected Model

1. Conservative wrist envelopes are restored:

| Capsule | Radius |
| --- | ---: |
| wrist 1 to wrist 2 | 0.045 m |
| wrist 2 to wrist 3 | 0.045 m |
| wrist 3 to gripper base | 0.072 m |

2. The Robotiq model includes base, inner/outer knuckles, inner/outer fingers,
   and a palm capsule between the two outer-knuckle roots. The wider 2F-85 base
   uses a 72 mm envelope; the former 50 mm envelope missed the brief contact.
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
| Episode steps | 14,610 |
| Runtime checks | 456 |
| Detector candidates | 85 |
| Allowed assembly contacts | 7 |
| Positive-clearance proximity events | 64 |
| Abnormal collision events | 14 |
| Minimum signed clearance | -0.042644 m |
| Monitor compute time | 26.116 s |
| Missing prims / geometry / monitor errors | 0 / 0 / 0 |

The report contains 13 inter-arm events and one robot-object event. Both
visually confirmed intervals are preserved:

- step 5408, `left_move_above_block_3`: two inter-arm events, first displayed
  at about 22.5 seconds;
- steps 8576-8768, `left_move_above_block_4`: the later arm/part and inter-arm
  collision interval around 35.7-36.5 seconds.

## Stride Review

- Stride 64 sampled before and after the brief contact and missed it.
- Stride 8 captured steps 5384-5440 in detail, but increased monitor work to
  1,254 checks and the rollout timed out in a later phase.
- Stride 32 sampled step 5408, retained both collision intervals, reduced the
  work to 456 checks, and completed the full assembly successfully.

Stride 32 is therefore the validated setting for this recipe. Stride 8 remains
useful for short diagnostic runs where maximum temporal resolution matters.

## Tests

```text
constraint integration: 39 passed
assembly regression:     12 passed
git diff --check:         passed
```

The three annotated videos contain 1,824 frames each at 30 FPS. Their overlay
reads only the collision event list and shows the pair, phase, step, and overlap
depth.

## Server Artifacts

```text
outputs/fabrica_plumbers_block_ur5e_contact_precision_v8_stride32_seed0/
  collect_results.json
  episode_0000_live_videos/
  episode_0000_annotated_videos/
  inspection_22s/
  annotation_checks/
```
