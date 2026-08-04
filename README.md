# RoboAssemblyBench

RoboAssemblyBench is a reproduction branch of InternUtopia focused on atomic-skill based robotic assembly. The current checkpoint contains a dual-UR5e + Robotiq 2F-85 Fabrica plumbers-block task:

`fabrica_plumbers_block_ur5e_right_base_prepare`

The task stages part 2 with the right arm, then uses the left arm to place part 0 into the staged part-2 slot, stack part 3, and insert parts 4 and 1 into the remaining holes.

## Quick Preview

The linked rollout is the latest physically validated checkpoint: it completes in
14,523 simulation steps with zero timeout or recovery events. Parts 0, 3, 4, and
1 are held by strict force-confirmed contact on both gripper fingers; placement
completion requires object-pose convergence and release does not snap parts to
their targets.

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
roboassemblybench/scripts/generate_fabrica_plumbers_block_ur5e_right_base_prepare_demo.sh
roboassemblybench/scripts/view_fabrica_plumbers_block_ur5e_right_base_prepare_scene_ui.sh
```

The logical robot names remain `franka_left` and `franka_right` for compatibility with the original factory task code, but both are instantiated as `UR5eRobot`. The wrist-mount recipe replaces the previous gripper setup with a Robotiq 2F-85 asset fixed under `wrist_3_link`.

The task uses local atomic skills registered in recipe metadata:

```text
ur5e_move_above_part
ur5e_descend_to_grasp
ur5e_close_gripper
ur5e_move_part_to_staging
ur5e_move_part_to_table_hover
ur5e_hold_part_end
```

All of them route through:

```text
toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills:UR5ePlumbersBlockAtomicSkillAdapter
```

Current generic safeguards include joint-space IK tracking, IK branch-jump and
wrist-flip limiting, bounded per-step joint targets, shared-workspace arm
clearance, TCP-frame object slip checks, strict dual-finger force-contact gates,
and object-pose convergence before placement completion. Physical attachment
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
4. Register reusable local skills in `metadata.local_skills` with the same `UR5ePlumbersBlockAtomicSkillAdapter`.
5. Prefer YAML parameters over code changes for new pick/place variants: `object`, `target_object_target`, `target_orientation`, `target_orientation_frame`, `offset`, `grasp_tcp_offset`, `grasp_tcp_offset_frame`, `cartesian_servo`, `cartesian_position_step`, `max_joint_step`, `position_tolerance`, `require_target_object_pose_convergence`, `attach`, and `release`.
6. Add a wrapper script in `roboassemblybench/scripts/` that calls `roboassemblybench/scripts/generate_demos.py` with the new recipe name.

Keep direct Cartesian IK disabled unless a specific task has been validated with `allow_direct_arm_ik_controller: true`; the default joint-space guarded path is the safer reusable setting for UR5e pick/place skills.

## Repository Layout

```text
environment.yml                         # exported Conda environment
internutopia_extension/                 # simulator robot/task extensions
roboassemblybench/tasks/                # assembly task recipes
roboassemblybench/scripts/              # UI and demo-generation entry points
toolkits/factory_dual_franka_assembly/  # policy, scene builder, and local skills
```

The GitHub repository contains code and lightweight configuration. Reproduction assets and preview videos are in the Hugging Face dataset listed above.
