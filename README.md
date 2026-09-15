# RoboAssemblyBench：双 UR5e 数据生成、训练与仿真评测

本分支 `ur5e-batch-generation-20260826` 面向 **两台 UR5e + 两个 Robotiq 2F-85** 的 Fabrica 长程装配任务。本文从一台新机器开始，说明如何安装环境、准备资产、验证七个任务、生成带域随机化的数据、转换为 LeRobot v3、训练 ACT，并在 Isaac Sim 中做闭环模型评测。

当前支持的七个 canonical 任务是：

| 短名 | Recipe | 零件数 |
| --- | --- | ---: |
| `beam` | `fabrica_beam_ur5e_staged` | 5 |
| `car` | `fabrica_car_ur5e_staged` | 6 |
| `cooling_manifold` | `fabrica_cooling_manifold_ur5e_staged` | 7 |
| `duct` | `fabrica_duct_ur5e_staged` | 8 |
| `gamepad` | `fabrica_gamepad_ur5e_staged` | 6 |
| `plumbers_block` | `fabrica_plumbers_block_ur5e_staged` | 5 |
| `stool_circular` | `fabrica_stool_circular_ur5e_staged` | 9 |

> 仓库中仍有 `toolkits/factory_dual_franka_assembly/`、`build_dual_franka_assembly_*` 和内部 robot key `franka_left` / `franka_right`。这些是为旧数据和 API 保留的兼容名称，不代表当前机器人是 Franka。UR5e recipe 中的实际类型是 `UR5eRobot`，USD 是 UR5e + Robotiq 2F-85。不要在本分支的数据生产中使用 `*_franka_*` recipe 或脚本。

## 1. 端到端流程

```text
安装 Isaac/LeRobot 环境并下载资产
                  |
                  v
静态 recipe 检查 -> Isaac 环境/oracle rollout 验收
                  |
                  v
Stage 1：求解成功装配并记录 position 组 RGB-D/轨迹
                  |
                  v
Stage 2：复用 Stage 1 成功轨迹，重放五种视觉域
                  |
                  v
42 个 raw 子集（7 tasks x 6 profiles）
                  |
                  v
LeRobot v3 转换与数据契约检查
                  |
                  v
ACT 训练 -> 独立 policy server -> Isaac Sim 闭环评测
```

六个数据 profile 都保留受约束的位置随机化：

| Profile | 变化内容 |
| --- | --- |
| `position` | pickup/fixture 与 assembly target 的位置；光学底板固定 |
| `object_distractors` | `position` + 桌面视觉干扰物 |
| `texture` | `position` + 桌面、墙面、地面纹理 |
| `lighting` | `position` + 灯光数量、位置、强度和颜色 |
| `table_color` | `position` + 桌面颜色 |
| `scene` | `position` + 工厂背景、背景偏移和轻微 yaw |

`mixed` 可用于人工抽查，但不属于均衡数据集的六个正式标签。Stage 2 只重放 Stage 1 已成功的控制轨迹，不重新做 IK、长程规划或 RoboBrain 推理。

## 2. 目录和主要入口

```text
roboassemblybench/
  assets/Fabrica/                         # Fabrica、UR5e、夹爪和装配资产
  tasks/_shared/_fabrica_canonical_ur5e.yaml
  tasks/fabrica_*_ur5e_staged/            # 七任务 recipe
  datasets/cartesian_episode.py           # raw 数据和同步观测契约
  scripts/
    validate_fabrica_canonical_ur5e.py    # oracle 环境验收
    generate_fabrica_canonical_ur5e_demo.sh
    collect_fabrica_plumbers_block_2k.py  # 可复用的单任务采集器
    collect_fabrica_7tasks_50k_twostage_multigpu.sh
    replay_fabrica_successful_trajectories.py
    export_fabrica_lerobot_v3.py
    convert_fabrica_10w_lerobot_v3.py
    train_fabrica_plumbers_block_act.sh
    evaluate_fabrica_plumbers_block_act.sh
    run_fabrica_plumbers_block_act_pipeline.sh
```

## 3. 机器要求

- Linux + NVIDIA GPU，驱动可运行 Isaac Sim 5.1.0；无显示器机器可以 headless 运行。
- NVIDIA Isaac Sim standalone 5.1.0，安装目录中有 `isaac-sim.sh`、`python.sh`、`setup_conda_env.sh`。
- Conda/Miniconda、Git LFS、`ffmpeg`、`ffprobe`。
- 大规模采集建议为每个 Isaac worker 预留至少 32 GiB 系统内存，并先实测显存占用。
- raw RGB-D 很大，正式输出必须放在独立数据盘，不要写进 Git 工作树。

以下所有命令都从仓库根目录执行：

```bash
git clone --branch ur5e-batch-generation-20260826 \
  https://github.com/baiyu858/RoboAssemblyBench.git
cd RoboAssemblyBench
```

## 4. 安装两个隔离环境

### 4.1 Isaac Sim / 数据生成环境

推荐让安装脚本按 Isaac Sim 自带 Python 创建环境：

```bash
bash setup_conda.sh
```

脚本询问环境名时输入 `internutopia311`。随后补齐当前分支的锁定依赖并重新安装本仓库：

```bash
conda env update -n internutopia311 -f environment.yml
conda activate internutopia311
```

`setup_conda.sh` 已执行 `pip install -e .`，并会在 conda 激活钩子中 source Isaac Sim 的 `setup_conda_env.sh`。每次换 Isaac Sim 安装位置后都要重新生成或修正这个钩子。

如果批处理直接使用 standalone Python，再准备运行时依赖目录：

```bash
export ISAAC_SIM_ROOT=/absolute/path/to/isaac-sim-5.1.0
test -x "$ISAAC_SIM_ROOT/python.sh"

mkdir -p .runtime_python
"$ISAAC_SIM_ROOT/python.sh" -m pip install --target .runtime_python \
  ikpy==3.4.2 httpx==0.25.2 zstandard rsl-rl-lib
```

验证 Isaac 环境：

```bash
conda run -n internutopia311 env PYTHONNOUSERSITE=1 python -c \
  "import internutopia, internutopia_extension, isaacsim; print('runtime imports ok')"

conda run -n internutopia311 env PYTHONNOUSERSITE=1 python -c \
  "from isaacsim import SimulationApp; app=SimulationApp({'headless': True}); \
import omni.replicator.core; from pxr import UsdGeom; print('Isaac rendering ok'); app.close()"
```

### 4.2 LeRobot / ACT 环境

数据导出、训练和 policy server 使用独立环境，避免 PyTorch/CUDA 依赖污染 Isaac：

```bash
conda create -n roboassemblybench-act python=3.11 -y
conda run -n roboassemblybench-act python -m pip install \
  "lerobot==0.4.4" "pyarrow>=23" av pytest
```

本分支已验证的参考组合是 Python 3.11、LeRobot 0.4.4、PyTorch 2.7.0 + CUDA 12.8 和 PyArrow 25.0.0。PyTorch 应按本机驱动/CUDA 安装；最终必须满足：

```bash
conda run -n roboassemblybench-act python -c \
  "import torch, lerobot, pyarrow; \
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION; \
print(torch.__version__, lerobot.__version__, pyarrow.__version__, CODEBASE_VERSION)"
```

最后一项必须是 `v3.0`，并且 `conda run -n roboassemblybench-act lerobot-train --help` 能正常退出。

## 5. 下载和验证资产

复现七任务 UR5e 数据至少需要 canonical metadata、UR5e workcell bundle、兼容目录中的随机纹理，以及 Simple Warehouse 场景：

```bash
conda run -n internutopia311 python \
  roboassemblybench/scripts/download_repro_assets_from_hf.py \
  --repo-id baiyu858/InternUtopia-repro-assets \
  --include 'roboassemblybench/assets/Fabrica/canonical_7_bundles/**' \
  --include 'roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/**' \
  --include 'roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001/**' \
  --include 'roboassemblybench/assets/isaac_sim_5.1/Isaac/Environments/Simple_Warehouse/**'
```

第三个目录名虽然含 `franka`，本分支只从中复用 PackingTable 纹理，不会加载 Franka 机器人。

```bash
export INTERNUTOPIA_ASSETS_PATH="$PWD/internutopia/assets"
export ISAAC_ASSETS_ROOT="$PWD/roboassemblybench/assets/isaac_sim_5.1"

test -f roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json
test -f roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/ur5e_robotiq_2f85_wrist_mount_task.usda
test -f roboassemblybench/assets/isaac_sim_5.1/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd

(cd roboassemblybench/assets/Fabrica/canonical_7_bundles && sha256sum -c SHA256SUMS)
```

## 6. 数据生成前的环境验收

### 6.1 不启动 Isaac 的静态检查

```bash
conda run -n internutopia311 python -m pytest -q \
  --confcutdir=tests/toolkits \
  tests/toolkits/test_fabrica_canonical_staged_tasks.py \
  tests/toolkits/test_factory_dual_franka_physical_contact.py \
  tests/toolkits/test_factory_dual_franka_assembly_profiles.py \
  tests/toolkits/test_gripper_controller.py
```

这里的个别测试文件仍沿用历史 `dual_franka` 命名；测试断言的当前 recipe 是 UR5e。

### 6.2 单任务可视化 smoke

先生成一条 `car` oracle demo：

```bash
OUTPUT_DIR=outputs/ur5e_car_smoke \
NUM_DEMOS=1 START_SEED=0 MAX_TRIALS=1 HEADLESS=1 \
  bash roboassemblybench/scripts/generate_fabrica_canonical_ur5e_demo.sh car \
  --randomization-profile position \
  --worker-layout-seeds 0
```

需要 GUI 检查场景、相机和碰撞体时：

```bash
bash roboassemblybench/scripts/view_fabrica_canonical_ur5e_scene_ui.sh car
```

### 6.3 七任务 oracle/环境评测

在同一个 Isaac worker 中依次验证七个 recipe，保存成功、终止阶段、碰撞和资源统计：

```bash
conda run -n internutopia311 python \
  roboassemblybench/scripts/validate_fabrica_canonical_ur5e.py \
  --tasks beam car cooling_manifold duct gamepad plumbers_block stool_circular \
  --seed 0 --layout-seed 0 \
  --record-live-video \
  --output-dir outputs/fabrica_canonical_ur5e_validation
```

验收文件是 `outputs/fabrica_canonical_ur5e_validation/validation_summary.json`。正式采集前应满足：

- `num_completed == 7` 且 `num_successful == 7`；
- `resource_abort == null`，worker 无超时；
- `runtime_collision_total` 和 `stage_precheck_collision_total` 符合当前任务约束；
- 视频中两台机器人确实为 UR5e、三个相机有效、零件没有穿模或飞散。

再加 `--domain-randomization` 并更换 `--layout-seed`，至少抽测若干未见布局。环境/oracle 不通过时不要开始大批量采集，也不要用训练模型的失败掩盖场景问题。

## 7. 生成一条可训练的 raw 数据

下面的命令会记录同步的三视角 RGB-D、状态、动作和长程标注，而不仅是演示视频：

```bash
export ISAAC_SIM_ROOT=/absolute/path/to/isaac-sim-5.1.0

conda run -n internutopia311 python \
  roboassemblybench/scripts/collect_fabrica_plumbers_block_2k.py \
  --recipe fabrica_car_ur5e_staged \
  --output-dir outputs/ur5e_car_raw_smoke \
  --num-episodes 1 --max-attempts 32 --batch-size 1 \
  --randomization-profile position --unique-layout-seeds \
  --dataset-fps 10 --dataset-frame-stride 8 --rendering-fps 80 \
  --require-extended-observations --require-visual-quality
```

成功 episode 目录以 `_cartesian_raw` 结尾；`collection_manifest.json` 是权威清单。每帧主要包含：

- `observation.state`：双臂绝对 Cartesian pose + gripper，共 16 维；
- `action`：下一采样时刻的双臂绝对 Cartesian target + gripper，共 16 维；
- front、left wrist、right wrist 三路 RGB，及同相机的毫米尺度 `uint16` 深度；
- 双臂 joint position/velocity/effort、EEF pose、wrist wrench、碰撞信号和 availability mask；
- phase、subtask、substage、waiting/handoff 状态、seed、layout seed、recipe fingerprint；
- 仿真、控制、相机和 state/action 对齐的 timing contract。

## 8. 七任务两阶段批量生成

先只检查调度参数，不启动 Isaac：

```bash
OUTPUT_ROOT=/data/ur5e/twostage_smoke \
GPU_IDS=0 TARGET_PER_TASK=1 EXPORT_LEROBOT=0 PIPELINE_DRY_RUN=1 \
  bash roboassemblybench/scripts/collect_fabrica_7tasks_50k_twostage_multigpu.sh
```

然后实际生成七任务 × 六组 × 每组 1 条：

```bash
export ISAAC_SIM_ROOT=/absolute/path/to/isaac-sim-5.1.0

ROBOT_PLATFORM=ur5e \
OUTPUT_ROOT=/data/ur5e/twostage_smoke \
GPU_IDS=0 GPU_WORKERS_PER_GPU=1 INITIAL_GPU_WORKERS=1 \
PIPELINE_REPLAY_GPU_IDS=0 PIPELINE_REPLAY_WORKERS_PER_GPU=1 \
TARGET_PER_TASK=1 EXPORT_LEROBOT=0 \
ISAAC_PYTHON="$ISAAC_SIM_ROOT/python.sh" \
  bash roboassemblybench/scripts/collect_fabrica_7tasks_50k_twostage_multigpu.sh
```

检查 smoke 全部成功后，再换新输出目录启动正式任务。以下是双 GPU 的保守起点：

```bash
ROBOT_PLATFORM=ur5e \
OUTPUT_ROOT=/data/ur5e/fabrica_ur5e_twostage_80hz \
GPU_IDS=0,1 GPU_WORKERS_PER_GPU=1 INITIAL_GPU_WORKERS=1 \
PIPELINE_REPLAY_GPU_IDS=0,1 PIPELINE_REPLAY_WORKERS_PER_GPU=1 \
TARGET_PER_TASK=1430 \
ISAAC_PYTHON="$ISAAC_SIM_ROOT/python.sh" \
LEROBOT_PYTHON=/absolute/path/to/roboassemblybench-act/bin/python \
  bash roboassemblybench/scripts/collect_fabrica_7tasks_50k_twostage_multigpu.sh
```

总 episode 数为 `7 × 6 × TARGET_PER_TASK`：

- `TARGET_PER_TASK=1`：42 条 smoke；
- `TARGET_PER_TASK=1190`：49,980 条；
- `TARGET_PER_TASK=1430`：60,060 条；
- `TARGET_PER_TASK=2381`：100,002 条。

文件名中的 `50k` 是历史名称，实际配额始终以启动日志中的计算结果为准。正式运行建议先保持 1 worker/GPU，观察至少一轮完整 episode 的 RAM、VRAM、成功率和重启情况后再提高并发。

只采指定任务时使用逗号分隔的 `FABRICA_TASKS`：

```bash
FABRICA_TASKS=beam,car TARGET_PER_TASK=10 \
OUTPUT_ROOT=/data/ur5e/beam_car_smoke GPU_IDS=0 \
ISAAC_PYTHON="$ISAAC_SIM_ROOT/python.sh" EXPORT_LEROBOT=0 \
  bash roboassemblybench/scripts/collect_fabrica_7tasks_50k_twostage_multigpu.sh
```

两台机器最稳妥的方式是按 `FABRICA_TASKS` 分配互不重叠的任务，并给每台机器独立 `OUTPUT_ROOT`；不要让两个进程并发写同一个 shard。后续用冻结清单统一选择和转换数据。

### 输出结构与断点续跑

```text
$OUTPUT_ROOT/
  stage1/<task>/shards/shard_*/
    collection_manifest.json
    batches/*/episode_*_cartesian_raw/
  rendered/<task>/<visual_profile>/shards/shard_*/
    replay_manifest.json
    batches/*/episode_*_cartesian_raw/
  rendered/<task>/<profile>/lerobot_v3/
  logs/<machine>/
```

采集、replay 和导出都按 manifest 判断完成状态。同一配置下重跑原命令会继续未完成 shard；不要修改 recipe 后强行复用旧 fingerprint。

查看进度：

```bash
TARGET_PER_TASK=1430 \
  bash roboassemblybench/scripts/monitor_fabrica_7tasks_twostage.sh \
  /data/ur5e/fabrica_ur5e_twostage_80hz

tail -f /data/ur5e/fabrica_ur5e_twostage_80hz/logs/*/*.log
watch -n 5 nvidia-smi
```

临时停止 Stage 2 而不破坏 Stage 1：

```bash
bash roboassemblybench/scripts/pause_fabrica_stage2.sh \
  /data/ur5e/fabrica_ur5e_twostage_80hz
```

## 9. 转换为 LeRobot v3

### 9.1 转换一个 raw 子集

```bash
conda run -n roboassemblybench-act python \
  roboassemblybench/scripts/export_fabrica_lerobot_v3.py \
  --input-dir /data/ur5e/fabrica_ur5e_twostage_80hz/stage1/car \
  --output-dir /data/lerobot/ur5e/car/position \
  --repo-id local/roboassemblybench_fabrica_ur5e_car_position \
  --encoder-threads 2 --vcodec h264 --resume
```

导出器只接受 raw schema、timing、三相机、16D state/action 和扩展观测均一致的成功 episode。`--resume` 会沿已有 conversion manifest 继续。

### 9.2 转换已冻结的 10 万条数据

若数据盘已有 `task7rand5/dataset_manifest.json` 冻结清单，使用 42 子集转换器：

```bash
ACT_PYTHON="$(conda run -n roboassemblybench-act python -c 'import sys; print(sys.executable)')"
ACT_SITE_PACKAGES="$(conda run -n roboassemblybench-act python -c 'import site; print(site.getsitepackages()[0])')"

"$ACT_PYTHON" \
  roboassemblybench/scripts/convert_fabrica_10w_lerobot_v3.py \
  --data-root /data/ur5e \
  --output-root /data/lerobot/ur5e \
  --code-root "$PWD" \
  --python "$ACT_PYTHON" \
  --local-site-packages "$ACT_SITE_PACKAGES" \
  --lerobot-site-packages "$ACT_SITE_PACKAGES" \
  --workers 8 --encoder-threads 2 --attempts 3 --prepare
```

`--prepare` 冻结 source selection；正式转换中不要一边补数据一边反复改变这份选择。转换失败且检测到不完整 metadata 或损坏 Parquet 时，原目录会被重命名为 `.quarantine-*` 保留，而不是直接删除。

每个子集只有同时存在以下文件才算完成：

```text
meta/info.json
roboassemblybench_conversion_manifest.json
.roboassemblybench_export_complete
```

总览保存在 `/data/lerobot/ur5e/conversion_run_summary.json`，应满足 `failed_subsets == []` 且 `converted_episodes == target_episodes`。

### 9.3 数据集验收

先运行导出回归测试：

```bash
conda run -n roboassemblybench-act python -m pytest -q \
  --confcutdir=tests/toolkits \
  tests/toolkits/test_roboassemblybench_lerobot_v3_export.py
```

再用 LeRobot 实际加载目标数据集：

```bash
export DATASET_ROOT=/data/lerobot/ur5e/car/position
conda activate roboassemblybench-act
python - <<'PY'
import json, os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(os.environ['DATASET_ROOT'])
info = json.loads((root / 'meta/info.json').read_text())
manifest = json.loads((root / 'roboassemblybench_conversion_manifest.json').read_text())
assert (root / '.roboassemblybench_export_complete').is_file()
assert info['total_episodes'] == manifest['total_episodes']
dataset = LeRobotDataset(repo_id=manifest['repo_id'], root=root, video_backend='pyav')
sample = dataset[0]
assert tuple(sample['observation.state'].shape) == (16,)
assert tuple(sample['action'].shape) == (16,)
for key in ('observation.images.front', 'observation.images.left_wrist', 'observation.images.right_wrist'):
    assert sample[key].ndim == 3
print('episodes=', dataset.meta.total_episodes, 'frames=', dataset.meta.total_frames)
PY
```

至少随机抽取若干 episode 检查 RGB、深度、state/action、phase 边界和 seed；仅有 `meta/info.json` 不等于数据完整。

## 10. ACT 基线训练

当前仓库完整封装的训练与在线评测基线是 `fabrica_plumbers_block_ur5e_right_base_prepare`。七任务数据都可以转换为相同的 LeRobot v3 基础 schema，但当前脚本不会自动把 42 个子集合并，也没有为其余六任务提供独立成功判定的 policy evaluator。

对一个已验收的 plumbers-block 数据集训练：

```bash
ACT_ENV=roboassemblybench-act \
DATASET_ROOT=/data/lerobot/ur5e/plumbers_block_dataset \
DATASET_REPO_ID=local/roboassemblybench_fabrica_plumbers_block_ur5e \
OUTPUT_DIR=/data/checkpoints/plumbers_block_act \
BATCH_SIZE=4 NUM_WORKERS=2 STEPS=100000 SAVE_FREQ=10000 \
WANDB_ENABLE=false \
  bash roboassemblybench/scripts/train_fabrica_plumbers_block_act.sh
```

训练输入是三路 RGB + 16D `observation.state`，监督目标是 16D `action`。默认 ACT 参数为 `chunk_size=100`、`n_action_steps=25`、AMP 开启。先用 `STEPS=10` 做显存和 dataloader smoke，再启动正式训练。

断点续训：

```bash
RESUME=true \
DATASET_ROOT=/data/lerobot/ur5e/plumbers_block_dataset \
DATASET_REPO_ID=local/roboassemblybench_fabrica_plumbers_block_ur5e \
OUTPUT_DIR=/data/checkpoints/plumbers_block_act \
  bash roboassemblybench/scripts/train_fabrica_plumbers_block_act.sh
```

脚本从 `checkpoints/last/pretrained_model/train_config.json` 恢复。不要把不同数据 schema、相机键或 recipe 的 checkpoint 写进同一 `OUTPUT_DIR`。

## 11. 模型在环境中的闭环评测

评测由两个隔离进程组成：LeRobot 环境加载 ACT 并提供本机 RPC；Isaac 环境接收动作，在随机化 UR5e 场景中闭环执行。动作会检查 16D/有限值并限制单步平移、旋转和夹爪范围。

先做短链路 smoke；它只验证 checkpoint、RPC、观测编码和 Isaac 控制闭环，不代表任务成功率：

```bash
CHECKPOINT=/data/checkpoints/plumbers_block_act/checkpoints/last/pretrained_model \
OUTPUT_DIR=/data/eval/plumbers_block_act_smoke \
NUM_EPISODES=1 START_SEED=10000 \
  bash roboassemblybench/scripts/evaluate_fabrica_plumbers_block_act.sh \
  --max-steps 64
```

正式评测使用训练未见的 episode seed，并固定记录 layout seed 集合：

```bash
CHECKPOINT=/data/checkpoints/plumbers_block_act/checkpoints/last/pretrained_model \
OUTPUT_DIR=/data/eval/plumbers_block_act_50ep \
NUM_EPISODES=50 START_SEED=10000 \
LAYOUT_SEEDS="4906 485 34 12" \
  bash roboassemblybench/scripts/evaluate_fabrica_plumbers_block_act.sh
```

每个 episode 在独立 Isaac 进程中执行，已完成 seed 会被跳过，因此同一命令可以续跑。最终结果：

```text
/data/eval/plumbers_block_act_50ep/
  success_rate.json       # complete、num_successes、success_rate、seed/layout contract
  episode_results.json    # 每条 success、terminal_reason、steps、wall_seconds
  episodes/               # 每个隔离 episode 的结果
  policy_server.log
```

报告模型结果时至少同时给出：checkpoint、dataset manifest/fingerprint、评测 recipe、scene profile、episode seed 区间、layout seeds、成功数/总数、success rate，以及按 `terminal_reason` 汇总的失败类型。不要用训练 seed 评测，也不要把 `--max-steps 64` 的 smoke 当正式指标。

当前 evaluator 会启用 recipe 的域随机化，但不会自动分别输出六个 profile 的矩阵。需要比较 `position`、纹理、灯光等单因素鲁棒性时，应先扩展 evaluator 显式接收并记录 `randomization_profile`，再用同一组 episode/layout seeds 做成对比较。

## 12. 一条命令跑 2k → LeRobot v3 → ACT → 50 episode

`plumbers_block` 基线也提供可恢复的完整流水线：

```bash
conda run -n internutopia311 python \
  roboassemblybench/scripts/run_fabrica_plumbers_block_act_pipeline.py \
  --raw-dir /data/ur5e/plumbers_block_2k_raw \
  --dataset-dir /data/lerobot/ur5e/plumbers_block_2k \
  --train-dir /data/checkpoints/plumbers_block_act \
  --eval-dir /data/eval/plumbers_block_act_50ep \
  --pipeline-output-dir /data/pipeline/plumbers_block_act \
  --expected-episodes 2000 \
  --train-steps 100000 \
  --eval-episodes 50 \
  --act-env roboassemblybench-act \
  --isaac-env internutopia311
```

状态保存在 `pipeline_state.json`。流水线会校验 collection 完整性、recipe fingerprint、240/30/8 timing contract、相机/state/action 对齐，随后导出、恢复训练并在线评测。若 raw 数据由外部并行采集器负责，增加 `--external-collection`，流水线只等待权威 manifest，不会另起重复 collector。

## 13. 常见问题

- `ModuleNotFoundError: isaacsim/omni/pxr`：检查 `internutopia311` 的 activate hook 是否 source 了当前 Isaac Sim 的 `setup_conda_env.sh`。
- 启动画面为空或机器人不是 UR5e：确认使用 `fabrica_*_ur5e_staged` 和 `generate_fabrica_canonical_ur5e_demo.sh`，不要使用 Franka recipe。
- 纹理随机化缺文件：UR5e runtime 的少量 PackingTable 纹理仍来自历史 `fabrica_franka_*` 资产目录，按第 5 节下载该目录。
- worker 长时间无输出：查看对应 `logs/<machine>/*.log`，同时检查 RAM、VRAM、磁盘保留量和 cgroup 限制；不要直接提高并发。
- manifest fingerprint 不一致：说明 recipe 已变化。换新输出目录重新采集；`ALLOW_RECIPE_FINGERPRINT_MISMATCH=1` 只用于经过人工审计的旧数据迁移。
- LeRobot 目录只有 `meta/info.json`：它可能是中断的半成品。完成标志还必须包括 conversion manifest 和 `.roboassemblybench_export_complete`。
- `Parquet magic bytes not found`：保留或让 10 万条转换器 quarantine 损坏目录，再从上一个有效 checkpoint/manifest 继续。
- ACT 训练 OOM：先降低 `BATCH_SIZE` 和 `NUM_WORKERS`；不要通过更改相机键或 state/action 维度来绕过问题。
- RPC 端口占用：设置新的 `PORT`，训练服务和 Isaac evaluator 必须使用同一个值。

## 14. 正式结果验收清单

- [ ] 当前分支是 `ur5e-batch-generation-20260826`，数据命令均使用 UR5e recipe。
- [ ] 两台机器人均为 `UR5eRobot` + Robotiq 2F-85，三路相机画面正常。
- [ ] 七任务静态测试和 oracle 环境验证通过。
- [ ] smoke 数据通过扩展观测、视觉质量、timing 和 fingerprint 检查。
- [ ] Stage 1 与五个 Stage 2 profile 的 manifest 数量达到计划配额。
- [ ] 每个 LeRobot v3 子集有完整 marker，且能被 `LeRobotDataset` 随机读取。
- [ ] 训练 checkpoint、训练数据 manifest 和超参数可追溯。
- [ ] 正式评测使用未见 seed，`success_rate.json` 完整，失败原因已汇总。

进一步的实现细节见 `roboassemblybench/README.md`；顶层本文是当前 UR5e 分支的数据生产和模型评测主入口。
