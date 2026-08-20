# RoboAssemblyBench: Dual-Franka Fabrica Handoff

本分支只面向双 Franka Panda 的 Fabrica 装配、调试和数据生成。目标是让接手者能够从零恢复当前环境，继续完成七个任务的物理验收，并批量生成带域随机化的 LeRobot v3 数据。

- GitHub: https://github.com/baiyu858/RoboAssemblyBench
- Branch: `franka-fabrica-handoff`
- Reproduction assets: https://huggingface.co/datasets/baiyu858/InternUtopia-repro-assets
- Simulator: NVIDIA Isaac Sim 5.1.0
- Robot: two official Franka Panda assets with `panda_hand`

## Current Checkpoint

七个 canonical recipe 已接入同一套编译器和原子技能：

| Task | Recipe | Static compile | Full physical episode |
| --- | --- | --- | --- |
| beam | `fabrica_beam_franka_staged` | ready | needs final acceptance |
| car | `fabrica_car_franka_staged` | ready | in progress |
| cooling_manifold | `fabrica_cooling_manifold_franka_staged` | ready | needs final acceptance |
| duct | `fabrica_duct_franka_staged` | ready | needs final acceptance |
| gamepad | `fabrica_gamepad_franka_staged` | ready | needs final acceptance |
| plumbers_block | `fabrica_plumbers_block_franka_staged` | ready | needs final acceptance |
| stool_circular | `fabrica_stool_circular_franka_staged` | ready | needs final acceptance |

这里的 `ready` 表示资产、recipe、技能图、Franka 关节控制、Panda 夹爪、接触检查、相机和记录器可以构建；只有 `results.json` 明确包含 `"success": true`，并通过视频物理检查，才算完整调通。不要把“运行到后半段”或旧的 UR5e 视频当作 Franka 成功证据。

当前 `car` 使用 seed/layout seed `5412` 验证了真实 Franka 资产、`base_2` 和 `part_1` 的抓取、运输、插入、释放和锁定，随后进入 `part_3`；完整 episode 仍需继续验收。当前官方布局为：

| Role | Robot | Base position | Base orientation, WXYZ |
| --- | --- | --- | --- |
| hold/base | `franka_left` | `[0.05, 0.25, 0.998051]` | `[0.707106781, 0, 0, -0.707106781]` |
| move/assembly | `franka_right` | `[0.95, 0.25, 0.998051]` | `[0.707106781, 0, 0, -0.707106781]` |

两臂平行同向，yaw 均为 `-90 deg`，X 方向间距 `0.90 m`；pickup/board origin 为 `[0.50, 0.05, 0.0125]`，assembly origin 为 `[0.50, -0.10, 0.0125]`。该覆盖目前只写在 `car` recipe 中，其余任务继续使用共享 Franka 默认布局，除非完整路径验证证明必须覆盖。

## Code Map

- `roboassemblybench/tasks/_shared/_fabrica_canonical_franka.yaml`: Franka 平台参数、Panda home、IK 连续性、夹持和插入策略。
- `roboassemblybench/tasks/fabrica_*_franka_staged/recipe.yaml`: 七个薄任务入口，只保留 assembly、左右臂角色和必要的任务覆盖。
- `roboassemblybench/core/fabrica_canonical.py`: 从 Fabrica metadata 编译目标、grasp、phase graph、随机化和成功条件。
- `toolkits/factory_dual_franka_assembly/plumbers_block_ur5e_skills.py`: 历史文件名未改，但适配器已同时支持 6-DOF UR5e 和 7-DOF Panda。
- `internutopia_extension/tasks/factory_dual_franka_assembly_task.py`: 物理 attach、compliant insertion、接触、碰撞和完成条件。
- `toolkits/factory_dual_franka_assembly/generate_demos.py`: 单进程 rollout、实时 MP4、trajectory-only 和 replay。
- `roboassemblybench/core/domain_randomization.py`: 位置、纹理、灯光、桌面颜色、背景场景和干扰物。
- `roboassemblybench/datasets/cartesian_episode.py`: 多视角 RGB-D、机器人状态、控制和阶段标注。
- `roboassemblybench/scripts/generate_fabrica_canonical_franka_demo.sh`: 单任务复现入口。
- `roboassemblybench/scripts/collect_fabrica_franka_7tasks_50k_multigpu.sh`: 一阶段多 GPU 采集。
- `roboassemblybench/scripts/collect_fabrica_franka_7tasks_50k_twostage_multigpu.sh`: trajectory-first 两阶段加速采集。

## Requirements

推荐配置：

- Ubuntu 22.04
- NVIDIA driver compatible with Isaac Sim 5.1.0
- NVIDIA Isaac Sim standalone 5.1.0
- Conda or Miniconda
- Git LFS
- 至少 32 GiB system RAM per Isaac worker for conservative collection
- 每个 Isaac worker 至少预留 8-12 GiB VRAM，实际值按场景和相机分辨率测量
- 足够的数据盘；不要把 50k 输出写进 Git 工作树

## Install

克隆本分支：

```bash
git clone --branch franka-fabrica-handoff \
  https://github.com/baiyu858/RoboAssemblyBench.git
cd RoboAssemblyBench
```

创建测试、调度和 LeRobot 导出环境：

```bash
conda env create -f environment.yml
conda activate internutopia311
pip install -e .
```

设置 standalone Isaac Sim：

```bash
export ISAAC_SIM_ROOT=/absolute/path/to/isaac-sim-5.1.0
test -x "$ISAAC_SIM_ROOT/python.sh"
```

为 Isaac 自带 Python 准备少量运行依赖：

```bash
mkdir -p .runtime_python
"$ISAAC_SIM_ROOT/python.sh" -m pip install --target .runtime_python \
  ikpy==3.4.2 httpx==0.25.2 zstandard rsl-rl-lib
```

`pyarrow` 只在导出 Parquet/LeRobot 时需要，不再是实时 MP4 的前置依赖。它已包含在 `environment.yml` 中。

## Assets

大文件不进入 GitHub。最小 Franka 复现需要四组 HF 资产：

1. 七任务 canonical bundle 和 grasp/precedence metadata。
2. Franka plumbers-block bundle，提供当前随机纹理贴图。
3. InternUtopia Franka USD、URDF 和 Lula 配置。
4. Isaac Simple Warehouse 离线场景和视觉干扰物。

下载到仓库中的固定相对路径：

```bash
python roboassemblybench/scripts/download_repro_assets_from_hf.py \
  --repo-id baiyu858/InternUtopia-repro-assets \
  --include 'roboassemblybench/assets/Fabrica/canonical_7_bundles/**' \
  --include 'roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001/**' \
  --include 'internutopia/assets/robots/franka/**' \
  --include 'roboassemblybench/assets/isaac_sim_5.1/Isaac/Environments/Simple_Warehouse/**'
```

配置路径：

```bash
export INTERNUTOPIA_ASSETS_PATH="$PWD/internutopia/assets"
export ISAAC_ASSETS_ROOT="$PWD/roboassemblybench/assets/isaac_sim_5.1"
```

验证：

```bash
test -f internutopia/assets/robots/franka/franka.usd
test -f internutopia/assets/robots/franka/lula_franka_gen.urdf
test -f roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json
test -f roboassemblybench/assets/isaac_sim_5.1/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd

cd roboassemblybench/assets/Fabrica/canonical_7_bundles
sha256sum -c SHA256SUMS
cd ../../../..
```

`third_part/Fabrica` 只用于回查官方 planner/log，不是 staged runtime 的最小依赖。需要时单独下载：

```bash
python roboassemblybench/scripts/download_repro_assets_from_hf.py \
  --include 'third_part/Fabrica/**'
```

## Static Verification

先在不启动 Isaac 的情况下验证七任务编译和共享技能契约：

```bash
conda run -n internutopia311 python -m pytest -q \
  --confcutdir=tests/toolkits \
  tests/toolkits/test_fabrica_canonical_franka_staged_tasks.py \
  tests/toolkits/test_factory_dual_franka_physical_contact.py \
  tests/toolkits/test_factory_dual_franka_assembly_profiles.py \
  tests/toolkits/test_gripper_controller.py
```

这一步必须确认：两台 `FrankaRobot`、每臂 7 个 arm joints + 2 个 finger joints、`panda_hand`、共享 phase graph 和随机化约束均正确。

## Reproduce One Episode

默认以 80 Hz、headless、trajectory + 三视角 MP4 运行 `car`：

```bash
ISAAC_SIM_ROOT=/absolute/path/to/isaac-sim-5.1.0 \
  bash roboassemblybench/scripts/generate_fabrica_canonical_franka_demo.sh car
```

输出：

```text
outputs/fabrica_franka_validation/car/results.json
outputs/fabrica_franka_validation/car/rollout.log
outputs/fabrica_franka_validation/car/episode_000000_cartesian_raw/
outputs/fabrica_franka_validation/car/episode_0000_live_videos/
```

快速无图像验证：

```bash
RECORD_LIVE_VIDEO=0 CONTROL_FPS=80 DATASET_FPS=20 DATASET_FRAME_STRIDE=4 \
  bash roboassemblybench/scripts/generate_fabrica_canonical_franka_demo.sh car
```

保守的 240 Hz 物理验证：

```bash
CONTROL_FPS=240 DATASET_FPS=30 DATASET_FRAME_STRIDE=8 \
LIVE_VIDEO_FPS=30 LIVE_VIDEO_FRAME_STRIDE=8 \
  bash roboassemblybench/scripts/generate_fabrica_canonical_franka_demo.sh car
```

切换任务只替换最后一个参数：

```bash
for task in beam cooling_manifold duct gamepad plumbers_block stool_circular; do
  RECORD_LIVE_VIDEO=0 OUTPUT_DIR="outputs/fabrica_franka_validation/$task" \
    bash roboassemblybench/scripts/generate_fabrica_canonical_franka_demo.sh "$task"
done
```

## Acceptance Contract

每个任务必须同时满足：

1. `results.json` 包含 `success=true` 和 `success-criteria-met`。
2. metadata 中 `replay_robot_names == ["franka_left", "franka_right"]`。
3. `replay_joint_widths == [9, 9]`，总关节宽度为 18；不能只根据文件名判断机器人类型。
4. 前视角能看到两台 Panda，腕部画面不为空、不出现贯穿画面的光条。
5. 夹爪两侧与物体形成物理接触；不得隔空 attach、穿模抓取或释放时瞬移吸附。
6. 所有零件最终位姿、插入深度、静止性和 precedence 均通过。
7. 无 `local-skill-failure`、`timeout-failure`、IK branch jump 或未解释的碰撞。

查看关键日志：

```bash
grep -E '\[rollout-progress\]|\[assembly-(ik|prealign)|local-skill-failure|timeout-failure|Traceback' \
  outputs/fabrica_franka_validation/car/rollout.log | tail -100
cat outputs/fabrica_franka_validation/car/results.json
```

## Finish The Remaining Tasks

按以下顺序调试，避免为单一零件不断添加特例：

1. 先运行静态测试，确认 recipe 编译和完整 grasp/IK waypoint 集合。
2. 固定 nominal seed，关闭随机化和视频，找到第一个失败 phase。
3. 判断失败属于 reachability、IK branch、夹爪接触、碰撞、插入捕获还是 success gate。
4. 优先修改共享 Franka 参数或通用候选选择器；只有几何证据证明任务不同，才在任务 recipe 中覆盖 base pose、role 或 terminal retreat。
5. nominal full episode 通过后，至少验证 4 个 layout seeds。
6. 最后逐个打开六类随机化，并人工审核前视角和腕部 RGB-D。

不要只扩大 timeout。phase 长时间不收敛时，记录当前关节、目标关节、TCP/物体误差和接触，再决定修改路径、等待位或容差。`car` 当前最明显的性能问题是 assembly arm 在连续零件之间反复回默认 home；后续应只在真实等待、角色切换或任务结束时 park。

建议先完成 `car` 的 `part_3/part_0/part_5/part_4`，再把同一修复回归到其余六任务。单任务验收结果应保存 seed、layout seed、commit SHA、结果 JSON、前视角 MP4 和失败 phase。

## Domain Randomization

启用随机化：

```bash
DOMAIN_RANDOMIZATION=1 RANDOMIZATION_PROFILE=position LAYOUT_SEED=1001 \
  bash roboassemblybench/scripts/generate_fabrica_canonical_franka_demo.sh car
```

支持的 profile：

| Profile | Effect |
| --- | --- |
| `position` | pickup group 平移 5-12 cm，assembly targets 平移 5-15 cm |
| `object_distractors` | 位置随机化 + 桌面视觉干扰物 |
| `texture` | 位置随机化 + 桌面、墙面、地面纹理 |
| `lighting` | 位置随机化 + 多灯数量、位置、强度和颜色 |
| `table_color` | 位置随机化 + 桌面颜色 |
| `scene` | 位置随机化 + 工厂背景场景、轻微位置和 yaw |
| `mixed` | 同时启用所有配置组，用于抽样检查，不建议直接作为五类均衡数据标签 |

光学底板固定不动。fixture 与 pickup parts 作为一组移动，assembly targets 作为另一组移动；约束采样会检查桌面边界、Panda 可达范围和组内相对几何。

## Recorded Data

每个 raw episode 保存：

- front、left wrist、right wrist RGB-D；
- 双臂关节位置、速度、可用时的 effort/torque；
- TCP position/orientation、夹爪状态和专家控制指令；
- 可用时的 wrist wrench、碰撞信号及 availability mask；
- phase、subtask、substage、左右臂角色、waiting/handoff 状态；
- 前置条件、完成条件、任务目标、装配顺序和最终结果；
- randomization profile、seed、layout seed 和 recipe fingerprint。

导出单个 raw 目录为 LeRobot v3：

```bash
conda run -n internutopia311 python \
  roboassemblybench/scripts/export_fabrica_lerobot_v3.py \
  --input-dir /path/to/raw \
  --output-dir /path/to/lerobot_v3 \
  --repo-id baiyu858/roboassemblybench_fabrica_car_franka_position
```

## Batch Collection

正式采集前，先在每个 task/profile 上生成 1 条：

```bash
FRANKA_MACHINE_ID="$(hostname -s)" \
GPU_IDS=0,1 GPU_WORKERS_PER_GPU=1 \
ISAAC_PYTHON="$ISAAC_SIM_ROOT/python.sh" \
LEROBOT_PYTHON="$(conda run -n internutopia311 which python)" \
  bash roboassemblybench/scripts/collect_fabrica_franka_7tasks_50k_multigpu.sh smoke
```

只调试指定组合：

```bash
TASK_FILTER=car PROFILE_FILTER=texture,lighting TARGET_PER_SUBSET=2 \
FRANKA_MACHINE_ID="$(hostname -s)" GPU_IDS=0 \
ISAAC_PYTHON="$ISAAC_SIM_ROOT/python.sh" EXPORT_LEROBOT=0 \
  bash roboassemblybench/scripts/collect_fabrica_franka_7tasks_50k_multigpu.sh formal
```

七任务 x 五视觉 profile 的默认 formal 配额精确合计 50,000 条。这五类数据都包含受约束的位置随机化，再分别叠加干扰物、纹理、灯光、桌面颜色或背景场景。正式采集必须显式指定数据盘，安全起步使用 1 worker/GPU：

```bash
FRANKA_MACHINE_ID="$(hostname -s)" \
FRANKA_DATA_ROOT=/data/a17/baiyongjie/data/franka \
GPU_IDS=0,1 GPU_WORKERS_PER_GPU=1 BATCH_SIZE=8 \
ISAAC_PYTHON="$ISAAC_SIM_ROOT/python.sh" \
LEROBOT_PYTHON=/path/to/internutopia311/bin/python \
  bash roboassemblybench/scripts/collect_fabrica_franka_7tasks_50k_multigpu.sh formal
```

80 Hz 加速采集必须保持 timing contract：

```bash
RENDERING_FPS=80 DATASET_FPS=10 DATASET_FRAME_STRIDE=8 \
  bash roboassemblybench/scripts/collect_fabrica_franka_7tasks_50k_multigpu.sh formal
```

多个节点共享任务时设置相同 `NODE_COUNT`，每台设置不同 `NODE_INDEX` 和 `FRANKA_MACHINE_ID`。每台机器写入独立子目录，避免共享盘并发覆盖。

采集脚本按 manifest 和 LeRobot `meta/info.json` 断点续跑，重复执行同一命令不会从零开始。查看进度和资源：

```bash
find "$FRANKA_DATA_ROOT" -name '*.status' -type f -print -exec tail -n 1 {} \;
find "$FRANKA_DATA_ROOT" -name collection_manifest.json -type f | wc -l
watch -n 5 nvidia-smi
```

## Two-Stage Acceleration

推荐的大规模方案是 trajectory-first：

1. 80 Hz 只求解并保存一次成功的位置随机化 trajectory。
2. 对同一 trajectory 分别 replay `object_distractors/texture/lighting/table_color/scene`。
3. replay 生成 RGB-D 后再并行导出 LeRobot v3。

这样不会为五个视觉 profile 重复执行五次长程装配：

```bash
ROBOT_PLATFORM=franka GPU_IDS=0,1 \
INITIAL_GPU_WORKERS=1 GPU_WORKERS_PER_GPU=2 \
ISAAC_PYTHON="$ISAAC_SIM_ROOT/python.sh" \
OUTPUT_ROOT=/data/a17/baiyongjie/data/franka/twostage_50k \
  bash roboassemblybench/scripts/collect_fabrica_franka_7tasks_50k_twostage_multigpu.sh
```

两阶段脚本默认按七任务 x 六组数据 x 每组 1,430 条生成 60,060 条，其中包含额外的 `position` 轨迹组；`TARGET_PER_TASK=1190` 可生成 49,980 条，`TARGET_PER_TASK=1191` 可生成 50,022 条。一阶段脚本使用非均匀配额，才是精确的 50,000 条。先用小 `TARGET_PER_TASK` 在所有七任务上 smoke；完整物理验收前不要直接启动大批量采集。

若显存和内存稳定，再逐步将 `GPU_WORKERS_PER_GPU` 从 1 调到 2；每次只增加一个 worker，至少观察 20 分钟的 RSS、显存、episode 成功率和 worker restart。不要一开始使用脚本允许的最大并发。

## Performance Notes

- 240 Hz + live video 的 `car` 预计约 55-75 min/episode。
- 240 Hz trajectory-only 预计约 45-60 min/episode。
- 80 Hz 会自动把 phase counts 缩小到三分之一，并把 motion step 放大 3 倍；目标约 15-25 min/episode，但必须重新验证接触稳定性。
- 去掉连续同臂任务之间无意义的 park，有机会进一步降到约 10-15 min/episode。
- live video 只用于验收抽样；正式采集使用 raw camera cadence 或 trajectory-first replay。
- GPU 利用率低通常是单 Isaac 进程受 CPU/PhysX 限制。增加独立 worker 提高总吞吐，但先观察 system RAM、VRAM 和 worker 稳定性。
- 不要在同一 `ISAACSIM_PORTABLE_ROOT` 下并发启动多个 Isaac 进程。

## Handoff Checklist

- [ ] 七任务 static tests 通过。
- [ ] 七任务 nominal episode 均有 `success=true`。
- [ ] 每个任务至少 4 个位置 seeds 通过。
- [ ] 六类随机化逐类通过视觉和物理抽样。
- [ ] 视频确认双 Panda、无光条、无空抓、无穿模、无吸附放置。
- [ ] 80 Hz 与 240 Hz 结果一致性完成抽查。
- [ ] smoke batch 和 LeRobot v3 schema 通过。
- [ ] 再启动多 GPU 50k，并持续监控 manifest、失败率、内存、显存和磁盘。
