# RoboAssemblyBench

RoboAssemblyBench is a reproduction branch of InternUtopia focused on atomic-skill based robotic assembly. The current checkpoint contains a dual-UR5e + Robotiq 2F-85 Fabrica plumbers-block task:

`fabrica_plumbers_block_ur5e_right_base_prepare`

The task stages part 2 with the right arm, then uses the left arm to place part 0 into the staged part-2 slot, stack part 3, and insert parts 4 and 1 into the remaining holes.

## Quick Preview

The linked rollout is a physically validated checkpoint: it completes without a
timeout or recovery. Grasp completion requires strict bilateral finger geometry,
blocked gripper closure, and bounded object motion. Placement completion requires
object-pose convergence, and release does not snap parts to their targets.

Videos are stored with the reproduction assets:

- Front view: https://huggingface.co/datasets/baiyu858/InternUtopia-repro-assets/resolve/main/outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/episode_0000_live_videos/observation_images_front.mp4
- Left wrist: https://huggingface.co/datasets/baiyu858/InternUtopia-repro-assets/resolve/main/outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/episode_0000_live_videos/observation_images_left_wrist.mp4
- Right wrist: https://huggingface.co/datasets/baiyu858/InternUtopia-repro-assets/resolve/main/outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/episode_0000_live_videos/observation_images_right_wrist.mp4

After restoring assets, the same files appear under:

```bash
outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/episode_0000_live_videos/
```

## Environment

This checkpoint is meant to be reproduced locally with NVIDIA Isaac Sim and Conda.

Required:

- Ubuntu 20.04/22.04 with an NVIDIA GPU and working driver
- NVIDIA Isaac Sim 5.1.0
- Conda
- Git LFS, for the external asset bundle

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate internutopia311
pip install -e .
```

Set Isaac Sim location if your install is not in the default path used on the development machine:

```bash
export ISAAC_SIM_ROOT=/path/to/isaac-sim
export PYTHONNOUSERSITE=1
```

## Assets

Large binary assets are kept outside GitHub in the Hugging Face dataset:

https://huggingface.co/datasets/baiyu858/InternUtopia-repro-assets

Restore them into the repository root while preserving relative paths:

```bash
git lfs install
git clone https://huggingface.co/datasets/baiyu858/InternUtopia-repro-assets /tmp/InternUtopia-repro-assets

mkdir -p roboassemblybench/assets/Fabrica outputs
rsync -a /tmp/InternUtopia-repro-assets/roboassemblybench/assets/Fabrica/ roboassemblybench/assets/Fabrica/
rsync -a /tmp/InternUtopia-repro-assets/outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/ outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/
```

The required asset paths are:

```text
roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001/
roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/
roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/ur5e_robotiq_2f85_wrist_mount_task.usda
roboassemblybench/assets/Fabrica/canonical_7_bundles/task_bundles/
roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json
```

## Run

Open the task scene in Isaac Sim UI without running the policy:

```bash
bash roboassemblybench/scripts/view_fabrica_plumbers_block_ur5e_right_base_prepare_scene_ui.sh
```

Generate one demo:

```bash
bash roboassemblybench/scripts/generate_fabrica_plumbers_block_ur5e_right_base_prepare_demo.sh
```

For a lower-resource headless run:

```bash
HEADLESS=1 NUM_DEMOS=1 MAX_TRIALS=1 LIVE_VIDEO_FRAME_STRIDE=8 \
  bash roboassemblybench/scripts/generate_fabrica_plumbers_block_ur5e_right_base_prepare_demo.sh \
  --skip-episode-steps
```

`--skip-episode-steps` keeps result metrics and live videos but omits the large
per-step observation/action list from `episode_0000.json`.

Outputs are written to:

```text
outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/
outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/collect_results.json
outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/episode_0000.json
outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo/episode_0000_live_videos/
```

All seven canonical Fabrica assemblies use one shared staged-task compiler.
Replace `<task>` with `beam`, `car`, `cooling_manifold`, `duct`, `gamepad`,
`plumbers_block`, or `stool_circular`:

```bash
bash roboassemblybench/scripts/view_fabrica_canonical_ur5e_scene_ui.sh <task>
bash roboassemblybench/scripts/generate_fabrica_canonical_ur5e_demo.sh <task>
```

These commands randomize the pickup layout, assembly targets, table color, and
background color by default. The optical board always remains at its fixed world
pose. Pickup layouts move by 5-12 cm and assembly targets move by 5-15 cm in XY;
constraint sampling keeps the fixture and final assembly on the fixed board and
keeps all generated TCP targets inside the configured UR5e reach envelope. Set
`DOMAIN_RANDOMIZATION=0` for the nominal layout.
The generator defaults to `SKIP_EPISODE_STEPS=1` to keep memory bounded while
still executing the complete rollout and recording videos. Set it to `0` only
when the large per-step JSON trace is required.

## Dataset And ACT

The position-randomized collector translates two groups independently in XY by
up to 2 cm: the complete start-parts group and the assembly targets. The optical
board remains fixed.
Each accepted episode contains three `640x480` RGB streams, a 16D dual-arm
Cartesian state, and a 16D next-sample absolute Cartesian action. Physics and
control run at 240 Hz; cameras and dataset samples are synchronized at 30 Hz
with `frame_stride=8` and `rendering_interval=7`.

Collect exactly 2,000 successful episodes with one memory-guarded Isaac worker:

```bash
bash roboassemblybench/scripts/collect_fabrica_plumbers_block_2k.sh
```

The worker is rejected and retried if available system memory remains below
`1.5 GiB`; the systemd service also caps the complete pipeline at `12 GiB`.

The collector uses only four prevalidated, near-nominal layouts, pinned in the
recipe as seeds `4906`, `485`, `34`, and `12`. The unique episode seed remains
independent, so 2,000 episodes can be indexed without treating repeated layouts
as duplicate episodes. Before formal collection, all four layouts must pass
qualification. A failure writes
`qualification_status.json` and stops without recording formal data; the same
failed fingerprint is not restarted automatically. After qualification, the
collector assigns the four layouts round-robin, is resumable, rejects failed,
misaligned, or out-of-contract episodes, and writes:

```text
outputs/fabrica_plumbers_block_ur5e_right_base_prepare_2k_raw_v3/
```

For the complete unattended workflow, run the pipeline watcher instead of the
standalone collector. It runs qualification, resumes collection after transient
resource exits, waits for all 2,000 successes, exports LeRobot v3, trains ACT,
then evaluates 50 randomized episodes using the automatic task success detector.
It does not restart a failed recipe qualification. The collection lock prevents
a second Isaac worker from being launched:

```bash
bash roboassemblybench/scripts/run_fabrica_plumbers_block_act_pipeline.sh
```

For multi-day collection, install the memory-limited user service. It survives
terminal closure and resumes on user login:

```bash
bash roboassemblybench/scripts/install_fabrica_plumbers_block_pipeline_service.sh
systemctl --user status roboassemblybench-fabrica-pipeline.service
journalctl --user -fu roboassemblybench-fabrica-pipeline.service
```

Monitor both processes:

```bash
cat outputs/fabrica_plumbers_block_ur5e_right_base_prepare_2k_raw_v3/qualification_status.json
tail -f outputs/fabrica_plumbers_block_ur5e_right_base_prepare_2k_raw_v3/collector.log
jq . outputs/fabrica_plumbers_block_pipeline/pipeline_state.json
```

The validated ACT environment uses Python 3.11, LeRobot 0.4.4 dataset schema
v3.0, PyTorch 2.7.0 with CUDA 12.8, and PyAV 15.1.0. The stage entry points are:

```text
roboassemblybench/scripts/export_fabrica_plumbers_block_lerobot_v3.py
roboassemblybench/scripts/train_fabrica_plumbers_block_act.sh
roboassemblybench/scripts/evaluate_fabrica_plumbers_block_act.sh
```

## Task Design

The current task is configured as layered YAML recipes:

```text
fabrica_plumbers_block_ur5e_right_base_prepare
  extends fabrica_plumbers_block_ur5e_wrist_mount
    extends fabrica_plumbers_block_ur5e
      extends fabrica_plumbers_block
```

Important files:

```text
roboassemblybench/tasks/fabrica_plumbers_block_ur5e_right_base_prepare/recipe.yaml
roboassemblybench/tasks/fabrica_plumbers_block_ur5e_wrist_mount/recipe.yaml
toolkits/factory_dual_franka_assembly/plumbers_block_ur5e_skills.py
toolkits/factory_dual_franka_assembly/ur5e_skill_api.py
roboassemblybench/scripts/generate_fabrica_plumbers_block_ur5e_right_base_prepare_demo.sh
roboassemblybench/scripts/view_fabrica_plumbers_block_ur5e_right_base_prepare_scene_ui.sh
```

The logical robot names remain `franka_left` and `franka_right` for compatibility with the original factory task code, but both are instantiated as `UR5eRobot`. The wrist-mount recipe replaces the previous gripper setup with a Robotiq 2F-85 asset fixed under `wrist_3_link`.

The task uses local atomic skills registered in recipe metadata:

```text
ur5e_move_above_part
ur5e_retreat_vertical
ur5e_preshape_gripper
ur5e_descend_to_grasp
ur5e_close_gripper
ur5e_move_part_to_staging
ur5e_move_part_to_table_hover
ur5e_hold_part_end
```

All of them route through:

```text
toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills:UR5eAssemblyAtomicSkillAdapter
```

Planner code can compile typed calls through
`toolkits.factory_dual_franka_assembly.ur5e_skill_api:UR5eAssemblySkillAPI`
instead of constructing recipe phases or adapter paths directly.

Current generic safeguards include joint-space IK tracking, IK branch-jump and
wrist-flip limiting, bounded per-step joint targets, shared-workspace arm
clearance, TCP-frame object slip checks, strict dual-finger physical-contact gates,
object-relative approach-axis hover poses, multi-sample interior contact checks,
and object-pose convergence before placement completion. Force probes are queried
only for recipes that require them; Isaac 5.1 force values are not used as a
substitute for bilateral grasp geometry. Physical attachment
filters only gripper/object collisions, and every release uses
`snap_on_open: false`.

The task-specific grasp poses remain recipe data rather than hard-coded policy
branches. In particular, part 1 is grasped on its shaft with an object-local TCP
offset while the gripper keeps a world-frame vertical orientation. This pattern
allows new objects to adjust grasp geometry independently of placement geometry.

## Add A New Assembly Task

Use the current task as the template:

1. Add a new task folder under `roboassemblybench/tasks/<task_name>/`.
2. Start its `recipe.yaml` with `extends: fabrica_plumbers_block_ur5e_wrist_mount` if it uses the same dual-UR5e + Robotiq setup.
3. Define task objects, world targets, and ordered `phases`.
4. Compile planner calls with `UR5eAssemblySkillAPI`, or register reusable local skills in `metadata.local_skills` with `UR5eAssemblyAtomicSkillAdapter`.
5. Prefer YAML parameters over code changes for new pick/place variants: `object`, `grasp_relative_position`, `grasp_relative_orientation`, `approach_clearance`, `gripper_openness`, `target_object_target`, `offset`, `cartesian_servo`, `position_tolerance`, `require_target_object_pose_convergence`, `attach`, and `release`. Insert `retreat_vertical` between a placement and the next cross-workspace pickup.
6. Add a wrapper script in `roboassemblybench/scripts/` that calls `roboassemblybench/scripts/generate_demos.py` with the new recipe name.

Keep direct Cartesian IK disabled unless a specific task has been validated with `allow_direct_arm_ik_controller: true`; the default joint-space guarded path is the safer reusable setting for UR5e pick/place skills.

For a canonical Fabrica bundle, use the smaller generic interface instead of
copying a long phase list: extend `_fabrica_canonical_ur5e.yaml` and set
`fabrica_canonical.assembly`. The compiler reads
`canonical_7_bundles/canonical_tasks.json`, stages the base, converts the
official Panda grasp candidates to Robotiq 2F-85 object-relative grasps,
selects every moving-part grasp from UR5e reach, orientation continuity,
insertion-axis alignment, gripper sweep clearance, and full-pose UR5e IK at
pickup approach, pickup, lift, assembly clearance, every insertion waypoint, and
final placement. The IK path must remain warm-start connected and above the
shared UR5e Jacobian-manipulability threshold, so a mathematically reachable but
near-singular grasp is rejected before rollout. The compiler then
reverses the official disassembly tree into assembly order and follows each
insertion path. No task name or part ID is handled by a policy branch. A new
canonical task supplies only assets, part bounds, assembly relations, grasp
candidates, and insertion paths; the compiler rejects incomplete or unreachable
task data before simulation. The optical board remains fixed and is never
position-randomized.
Regenerate the JSON only in the `fabrica` environment:

```bash
PYTHONPATH=third_part/Fabrica conda run -n fabrica \
  python roboassemblybench/scripts/build_fabrica_canonical_metadata.py
```

## Repository Layout

```text
environment.yml                         # exported Conda environment
internutopia_extension/                 # simulator robot/task extensions
roboassemblybench/tasks/                # assembly task recipes
roboassemblybench/scripts/              # UI and demo-generation entry points
toolkits/factory_dual_franka_assembly/  # policy, scene builder, and local skills
```

The GitHub repository contains code and lightweight configuration. Reproduction assets and preview videos are in the Hugging Face dataset listed above.
