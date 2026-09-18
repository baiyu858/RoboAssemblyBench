# AutoMate: from official trajectories to a policy dataset

This directory vendors NVIDIA Isaac Lab's official AutoMate implementation at
commit `37ddf626871758333d6ed89cf64ad702aef127d0` (`v2.3.2`). It is intended for
Isaac Sim 5.1 and Python 3.11. The upstream task code and BSD-3-Clause headers
are retained; RoboAssemblyBench adds local-asset resolution and launchers that
select an assembly without rewriting source files.

RoboAssemblyBench registers scoped environment IDs so that a stock Isaac Lab
checkout cannot overwrite the vendored registration:

- `RoboAssemblyBench-AutoMate-Assembly-Direct-v0`
- `RoboAssemblyBench-AutoMate-Disassembly-Direct-v0`

The complete data flow is:

```text
official assets and disassembly trajectories
                    |
                    v
       one PPO assembly policy per assembly
                    |
                    v
 policy rollout with third-person and wrist cameras
                    |
                    v
       successful raw episodes and sidecars
                    |
                    v
                 LeRobot
```

There are three different data products. Do not treat them as interchangeable:

1. `disassemble_traj.json` is the scripted end-effector trajectory used by the
   imitation reward during PPO training.
2. `Assembly.pth` is the trained RL-Games policy checkpoint for one assembly.
3. A policy dataset is a set of successful policy rollouts with synchronized
   state, action, RGB/depth, randomization, and task metadata. The official
   disassembly JSON is not this final dataset.

## 1. Create the Python environments

### 1.1 Prerequisites

- Linux and an NVIDIA driver that can run Isaac Sim 5.1;
- a standalone Isaac Sim 5.1 installation;
- Conda or Miniconda, Git, Git LFS, `ffmpeg`, and `ffprobe`;
- enough local storage for the 1.56 GB AutoMate asset mirror, checkpoints, raw
  videos/depth, and the final LeRobot dataset.

Accept the NVIDIA Omniverse EULA yourself before using Isaac Sim. On a headless
machine, set `OMNI_KIT_ACCEPT_EULA=YES` only after it has been accepted.

The examples below assume the repository root is the current directory:

```bash
export REPO_ROOT="$PWD"
export ISAAC_SIM_ROOT=/absolute/path/to/isaac-sim-5.1.0
test -x "$ISAAC_SIM_ROOT/python.sh"
```

### 1.2 Isaac Sim and Isaac Lab environment

Create a Python 3.11 environment and install this repository:

```bash
conda create -n automate python=3.11 -y
conda activate automate
source "$ISAAC_SIM_ROOT/setup_conda_env.sh"
python -m pip install --upgrade pip
python -m pip install -e "$REPO_ROOT"
```

Clone the exact Isaac Lab release expected by this vendored copy. `IsaacLab/`
is intentionally ignored by Git.

```bash
git clone --branch v2.3.2 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git "$REPO_ROOT/IsaacLab"
ln -s "$ISAAC_SIM_ROOT" "$REPO_ROOT/IsaacLab/_isaac_sim"
cd "$REPO_ROOT/IsaacLab"
./isaaclab.sh --install rl_games
```

If `_isaac_sim` already exists, verify that it resolves to the intended Isaac
Sim 5.1 installation instead of replacing it blindly.

Check the runtime before starting a long job:

```bash
cd "$REPO_ROOT/IsaacLab"
./isaaclab.sh -p -c \
  "import gymnasium, isaaclab, rl_games, torch, warp; print(torch.cuda.is_available())"
./isaaclab.sh -p -c \
  "import roboassemblybench.library.automate as a; print(a.ASSEMBLY_TASK_ID)"
nvidia-smi
```

### 1.3 Separate LeRobot conversion environment

Keep conversion dependencies out of the Isaac Sim environment. The Fabrica
export path in this repository is tested with LeRobot 0.4.4:

```bash
conda create -n automate-lerobot python=3.11 -y
conda activate automate-lerobot
python -m pip install --upgrade pip
python -m pip install "lerobot==0.4.4" opencv-python numpy
```

## 2. Download and verify the 100 official tasks

The official asset bundle is mirrored under
`roboassemblybench/library/automate/assets/AutoMate`. It contains 100
plug/socket assemblies, grasp metadata, disassembly distances, OBJ/USD assets,
and one official `disassemble_traj.json` per assembly. The 1.56 GB directory is
ignored by Git; `automate_assets_manifest.json` records its source and objects.

Download or resume it with:

```bash
cd "$REPO_ROOT"
env HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= \
  python roboassemblybench/library/automate/download_assets.py --workers 8
```

Create the authoritative assembly list from the checked-in manifest and verify
that every task has its training inputs:

```bash
mkdir -p outputs/automate
python - <<'PY' > outputs/automate/assembly_ids.txt
import json
from pathlib import Path

manifest = json.loads(
    Path("roboassemblybench/library/automate/automate_assets_manifest.json").read_text()
)
ids = sorted(
    {
        item["relative_path"].split("/", 1)[0]
        for item in manifest["objects"]
        if item["relative_path"].endswith("/disassemble_traj.json")
    }
)
assert len(ids) == 100, len(ids)
print("\n".join(ids))
PY

while read -r assembly_id; do
  task_dir="roboassemblybench/library/automate/assets/AutoMate/$assembly_id"
  test -s "$task_dir/plug.usd"
  test -s "$task_dir/socket.usd"
  test -s "$task_dir/plug.obj"
  test -s "$task_dir/socket.obj"
  test -s "$task_dir/disassemble_traj.json"
done < outputs/automate/assembly_ids.txt
```

The downloaded official trajectories are sufficient for policy training. To
regenerate 100 scripted trajectories for one assembly with the official
disassembly environment:

```bash
cd "$REPO_ROOT/IsaacLab"
TERM=xterm ./isaaclab.sh -p \
  "$REPO_ROOT/roboassemblybench/library/automate/generate_disassembly.py" \
  --assembly-id 00015 \
  --output-dir "$REPO_ROOT/outputs/automate/00015/disassembly" \
  --num-envs 32 \
  --num-trajectories 100 \
  --seed 0 \
  --headless \
  --device cuda:0
```

The output is
`outputs/automate/00015/disassembly/00015_disassemble_traj.json`. Each item
contains `fingertip_centered_pos`, `fingertip_centered_quat`, `arm_dof_pos`,
the initial grasp pose, and the plug pose/rotation. This is scripted
disassembly, not behavior cloning. PPO still produces the assembly policy;
Soft-DTW against these trajectories contributes an imitation reward.

## 3. Train the 100 assembly policies

### 3.1 Smoke test one task

Use the official trajectory shipped with assembly `00015`:

```bash
cd "$REPO_ROOT/IsaacLab"
TERM=xterm ./isaaclab.sh -p \
  "$REPO_ROOT/roboassemblybench/library/automate/run_assembly.py" \
  --mode train \
  --assembly-id 00015 \
  --disassembly-json \
    "$REPO_ROOT/roboassemblybench/library/automate/assets/AutoMate/00015/disassemble_traj.json" \
  --num-envs 128 \
  --max-iterations 2 \
  --seed 0 \
  --headless \
  --device cuda:0 \
  agent.params.config.device=cuda:0 \
  agent.params.config.full_experiment_name=automate_00015_smoke
```

After this succeeds, run the official 1500-epoch configuration by changing
`--max-iterations 2` to `--max-iterations 1500`.

### 3.2 Train all tasks

The following scheduler runs at most one task per listed GPU and writes a
separate log/checkpoint directory for every assembly. Set `GPU_IDS` to the
physical devices available on the machine.

```bash
cd "$REPO_ROOT"
GPU_IDS=(0 1 2 3 4 5 6 7)
mkdir -p outputs/automate/train_logs

job_index=0
while read -r assembly_id; do
  while [ "$(jobs -pr | wc -l)" -ge "${#GPU_IDS[@]}" ]; do
    wait -n
  done

  gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  run_name="automate_${assembly_id}_seed0"
  trajectory="$REPO_ROOT/roboassemblybench/library/automate/assets/AutoMate/$assembly_id/disassemble_traj.json"
  log="$REPO_ROOT/outputs/automate/train_logs/${run_name}.log"

  (
    cd "$REPO_ROOT/IsaacLab"
    TERM=xterm ./isaaclab.sh -p \
      "$REPO_ROOT/roboassemblybench/library/automate/run_assembly.py" \
      --mode train \
      --assembly-id "$assembly_id" \
      --disassembly-json "$trajectory" \
      --num-envs 128 \
      --max-iterations 1500 \
      --seed 0 \
      --headless \
      --device "cuda:$gpu" \
      "agent.params.config.device=cuda:$gpu" \
      "agent.params.config.full_experiment_name=$run_name"
  ) >"$log" 2>&1 &

  job_index=$((job_index + 1))
done < outputs/automate/assembly_ids.txt
wait
```

The best checkpoint for task `<ID>` is expected at:

```text
IsaacLab/logs/rl_games/Assembly/automate_<ID>_seed0/nn/Assembly.pth
```

Do not begin bulk collection merely because a `.pth` file exists. Evaluate
every task and freeze a checkpoint manifest containing assembly ID, absolute
checkpoint path, SHA-256, training seed, epoch, and success rate. For one task:

```bash
assembly_id=00015
checkpoint="$REPO_ROOT/IsaacLab/logs/rl_games/Assembly/automate_${assembly_id}_seed0/nn/Assembly.pth"

cd "$REPO_ROOT/IsaacLab"
TERM=xterm ./isaaclab.sh -p \
  "$REPO_ROOT/roboassemblybench/library/automate/run_assembly.py" \
  --mode play \
  --assembly-id "$assembly_id" \
  --checkpoint "$checkpoint" \
  --num-envs 128 \
  --seed 10000 \
  --eval-output "$REPO_ROOT/outputs/automate/$assembly_id/evaluation.h5" \
  --headless \
  --device cuda:0 \
  agent.params.config.device=cuda:0
```

The evaluation HDF5 contains `held_asset_pose`, `fixed_asset_pose`, and
`success`. Select and freeze checkpoints before dataset generation so a retry
cannot silently switch policies.

## 4. Generate 200,000 policy episodes

### 4.1 Exact quota

In this plan, "200k samples" means **200,000 successful episodes**, not
200,000 frames:

```text
100 assemblies x 5 visual randomization profiles x 400 successes = 200,000
```

The five profiles match the Fabrica `task7rand5` convention:

- `object_distractors`
- `texture`
- `lighting`
- `table_color`
- `scene`

Layout/pose randomization still receives a new deterministic seed for every
attempt. A failed rollout is retained only in diagnostics and does not count
toward the 400-success cell quota. The collector must persist both
`layout_seed` and `visual_seed`; retries must never reuse an accepted episode
identity.

### 4.2 Camera installation

Each environment needs two synchronized Isaac Lab cameras:

- `observation.images.front`: a fixed third-person camera that sees the Franka
  gripper, plug, socket, and their approach corridor;
- `observation.images.wrist`: a camera parented to `panda_hand`, aimed along
  the insertion axis with the fingers and plug tip visible.

Both cameras must capture RGB and metric depth at the dataset frame boundary.
Use 30 dataset FPS, a 120 Hz simulation/control loop, and `frame_stride=4`, or
record the actual values if another rate is deliberately selected. State,
action, RGB, and depth for one frame must represent the same simulation step.
Store camera intrinsics, extrinsics/mount transform, source resolution, depth
scale, and the exact preprocessing operation in episode metadata.

Do not use the viewer video produced by `run_assembly.py --video` as training
data. It is a presentation video, has no aligned state/action/depth stream,
and is not a substitute for scene cameras.

### 4.3 Raw episode contract

Use the same field categories and timing rules as the Fabrica raw recorder,
adapted to one Franka arm. Do not fake a dual-arm record by duplicating the
wrist image or zero-padding a second robot.

Each successful episode directory must contain:

```text
episode_<index>_cartesian_raw/
  metadata.json
  trajectory.npz
  annotations/episode_<index>.json
  videos/observation_images_front.mp4
  videos/observation_images_wrist.mp4
  sensors/depth/observation_images_front.u16.bshuf.zst
  sensors/depth/observation_images_wrist.u16.bshuf.zst
```

`trajectory.npz` must carry the Fabrica-equivalent arrays:

| Array | AutoMate shape | Meaning |
| --- | ---: | --- |
| `observation_state` | `(T, 8)` | EEF position, quaternion, gripper opening |
| `action` | `(T, 8)` | next-sample absolute EEF target and gripper target |
| `policy_action` | `(T, 6)` | original AutoMate delta position/rotation action |
| `joint_state` | `(T, 7)` | Franka arm joint position |
| `joint_velocity` | `(T, 7)` | Franka arm joint velocity |
| `joint_effort` | `(T, 7)` | Franka arm joint effort |
| `wrist_wrench` | `(T, 6)` | force/torque, plus an availability flag in metadata |
| `collision_signal` | `(T, 4)` | collision/contact summary compatible with Fabrica semantics |
| `simulation_step` | `(T,)` | source simulator step for alignment checks |
| `phase_index`, `phase_step` | `(T,)` | assembly phase labels |
| `subtask_index`, `substage_index` | `(T,)` | task-progress labels |
| `waiting_state`, `handoff_state` | `(T,)` | zero for this single-arm task |

`metadata.json` must include at least:

- schema version, assembly ID, task text, episode index, success, and frame
  count;
- policy checkpoint path and SHA-256;
- episode, layout, and visual seeds plus the randomization profile/result;
- state/action names and action semantics;
- simulation FPS, dataset FPS, frame stride, rendering interval, and a
  `camera_state_action_aligned=true` assertion;
- RGB video paths/shapes and depth stream paths, dtype, compression, scale,
  and count;
- robot joint/body names, camera calibration, and software/asset provenance.

Use the output layout below so collection is resumable and auditable:

```text
outputs/automate/policy_dataset/
  checkpoint_manifest.json
  collection_manifest.json
  <assembly_id>/<profile>/shards/<shard>/batches/<batch>/episode_*_cartesian_raw/
```

The collection manifest is authoritative. An episode counts only after all
files are closed, frame counts match, success is true, and its metadata path is
atomically added to the manifest.

### 4.4 Current implementation boundary

This directory currently implements and has been exercised for official asset
download, scripted disassembly generation, PPO training, checkpoint evaluation,
and viewer-video recording. It does **not yet contain** the synchronized
two-camera policy collector or the single-arm AutoMate-to-LeRobot exporter
described in sections 4.2-4.3. Therefore the commands in sections 1-3 are
runnable now, while the 200k job must not be launched until those two entry
points and their smoke tests are added.

The implementation should reuse the validation, atomic-manifest, RGB video,
metric-depth, and streaming export machinery in:

- `roboassemblybench/datasets/cartesian_episode.py`
- `roboassemblybench/scripts/export_fabrica_plumbers_block_lerobot_v3.py`

It must add AutoMate-specific single-arm schema handling rather than weakening
the existing Fabrica dual-arm checks.

Before scaling to 200k, require this sequence:

1. one checkpoint, one profile, one successful episode;
2. verify two RGB streams and two depth streams visually and numerically;
3. validate all trajectory shapes and simulation-step alignment;
4. convert that episode to LeRobot and read it back;
5. run 100 episodes for one task and test resume after interruption;
6. only then schedule the `100 x 5 x 400` production matrix.

The final LeRobot dataset should expose the same semantic features as the raw
episode (`observation.state`, `action`, joint state/velocity/effort, wrist
wrench, collision signal, progress labels, and both camera videos), preserve
metric depth as indexed sidecars, and copy checkpoint/randomization provenance
into its conversion manifest.

## License and provenance

The vendored files retain upstream BSD-3-Clause headers. See
`provenance.json` for the exact source commit and compatibility information.
