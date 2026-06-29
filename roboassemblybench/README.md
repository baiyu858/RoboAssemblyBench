# RoboAssemblyBench

`RoboAssemblyBench` is the standalone dual-arm assembly benchmark package in this repository.

The benchmark is organized as a framework instead of a Taoyuan-only demo:

- `core/`: shared loaders, scene profile resolution, task registry, and runtime-facing APIs
- `tasks/<task_name>/`: one folder per benchmark task with `recipe.yaml` and `annotation.yaml`
- `scenes/profiles/`: scene profile definitions, including lightweight procedural and asset-backed variants
- `scripts/`: benchmark entrypoints for rollout generation, dataset conversion, LeRobot export, and camera preview
- `robobrain/`: online RoboFactory-style RoboBrain planner that turns a natural-language task into
  subgoals, compositional constraints, a generated recipe, and optional demo rollout.

Current task folders:

- `tasks/peg_insertion/`
- `tasks/screw_fastening/`
- `tasks/panel_alignment/`
- `tasks/bracket_latching/`
- `tasks/connector_docking/`
- `tasks/gear_pair_mesh/`
- `tasks/nut_thread_after_hold/`
- `tasks/handover_fastener_then_insert/`
- `tasks/box_packing/`

The legacy toolkit under `toolkits/factory_dual_franka_assembly/` is kept as a compatibility layer, but new task authoring should target `roboassemblybench/`.

## internutopia311 Conda 环境安装

`internutopia311` 是运行 RoboAssemblyBench + InternUtopia + Isaac Sim 的默认 conda 环境。
这个环境不能只按普通 Python 环境创建；它必须和本机 Isaac Sim 自带的 Python 版本一致，并且在激活时
source Isaac Sim 的 `setup_conda_env.sh`，这样 `isaacsim`、`omni.*`、`pxr` 和 RTX/Replicator
渲染依赖才会被加入到 `PYTHONPATH`/`LD_LIBRARY_PATH`。

准备项：

- 一台带 NVIDIA GPU 和可用驱动的 Linux 机器；无显示器机器也可以跑 headless 渲染。
- 已安装 conda，并且当前 shell 可以使用 `conda`。
- 已安装 Isaac Sim，目录下需要有 `isaac-sim.sh`、`python.sh`、`setup_conda_env.sh`。
  换机器时使用自己的 Isaac Sim 路径；激活 `internutopia311` 后应能看到 `ISAAC_PATH` 指向该目录。
- 已拉取本仓库以及 RoboAssemblyBench/Fabrica 相关资产。

推荐安装方式是直接运行仓库根目录下的安装脚本：

```bash
cd /home/baiyu24/model/InternUtopia
bash setup_conda.sh
```

脚本会依次询问 Isaac Sim 路径和 conda 环境名。环境名提示默认是 `internutopia`，这里请输入：

```text
internutopia311
```

安装脚本实际会做这些事：

- 调用 `${ISAAC_SIM_PATH}/python.sh` 读取 Isaac Sim 自带 Python 版本；
- 创建 `conda create -n internutopia311 python=<Isaac Python version> libxcb=1.14`；
- 在 `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh` 写入 Isaac Sim 环境钩子；
- 在 `$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh` 写入还原环境变量的钩子；
- 执行 `pip install -e .`，把 InternUtopia 以 editable 形式安装到当前环境。

如果需要手动补齐渲染脚本依赖，激活环境后执行：

```bash
conda activate internutopia311
python -m pip install -e .
python -m pip install numpy pyyaml imageio imageio-ffmpeg trimesh scipy
```

其中 `pip install -e .` 会安装 `requirements/runtime.txt` 中的运行时依赖；`imageio`、
`imageio-ffmpeg`、`trimesh`、`scipy` 是 Fabrica 官方轨迹可视化渲染脚本直接用到的额外依赖。
`ffmpeg`/`ffprobe` 也建议在系统里可用，便于写出和检查 mp4：

```bash
which ffmpeg
which ffprobe
```

基础验证命令：

```bash
conda run -n internutopia311 env PYTHONNOUSERSITE=1 python -c \
  "import platform, numpy, imageio, trimesh, scipy, yaml, internutopia, internutopia_extension; \
print('python', platform.python_version()); print('python packages ok')"
```

完整 Isaac Sim 渲染验证会启动一次 headless `SimulationApp`，第一次运行可能比较慢：

```bash
conda run -n internutopia311 env PYTHONNOUSERSITE=1 python -c \
  "from isaacsim import SimulationApp; \
app = SimulationApp({'headless': True}); \
import omni.replicator.core as rep; \
from pxr import UsdGeom; \
print('isaac sim rendering imports ok'); \
app.close()"
```

如果直接在没有启动 `SimulationApp` 的短脚本里 import `pxr` 或 `omni.replicator` 失败，先不要只按普通
pip 缺包处理；优先检查 conda 激活钩子是否指向正确的 Isaac Sim：

```bash
conda activate internutopia311
cat "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo "$ISAAC_PATH"
echo "$PYTHONPATH"
```

`env_vars.sh` 里应该 source 当前 Isaac Sim 目录下的 `setup_conda_env.sh`，并设置正确的 `ISAAC_PATH`。
UR5e motion policy 配置会优先读取 `ISAAC_SIM_ROOT`，其次读取 `ISAAC_PATH`。如果路径不对，重新运行
`bash setup_conda.sh`，或手动修正该文件后重新打开 shell/重新激活环境。若报错里仍出现旧机器路径
（例如 `/home/baiyu24/APP/isaac-smi/.../ur5e_robot_description.yaml`），说明当前 shell 没有正确激活
Isaac Sim 环境变量。

## 渲染 Fabrica 官方 plumbers_block UR5e 轨迹

`roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh`
用于把 Fabrica 官方 `plumbers_block` UR5e/Robotiq 轨迹回放到 RoboAssemblyBench 的 task env 里，
并用 Isaac Sim Replicator 渲染出 mp4 和逐帧 PNG。它适合检查“官方 Fabrica 轨迹在 RoboAssemblyBench
任务环境中的视觉对齐效果”，不是物理接触仿真，也不是把轨迹重新通过 UR5e 控制器 retarget 后执行。

运行前确认这些输入都存在：

```bash
test -f roboassemblybench/tasks/fabrica_plumbers_block_ur5e/recipe.yaml
test -f roboassemblybench/scenes/profiles/taoyuan_grscenes_tabletop.yaml
test -f roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/ur5e_robotiq_2f85_task.usda
test -f roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block/traj.npy
test -f roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block/fixture/fixture.obj
test -d roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001/assets/fabrica_original_usd_sdf_margin_001/aligned/plumbers_block/parts
test -f roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001/assets/fabrica_support/optical_board.obj
test -f roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/isaac_official/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd
test -d roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets/isaac_official/Isaac/Robots/Robotiq/2F-85/parts
```

默认完整渲染命令：

```bash
cd /home/baiyu24/model/InternUtopia
bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

脚本默认使用 headless 模式，适合服务器或没有本地显示器的机器。如果要在本机直接打开 Isaac Sim GUI，
显式关闭 headless：

```bash
HEADLESS=0 \
  bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

如果要远程可视化，启用 Isaac Sim WebRTC livestream。先在运行 Isaac Sim 的机器上启动：

```bash
HEADLESS=1 WEBRTC=1 \
  bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

然后在另一台机器上打开 Isaac Sim WebRTC Streaming Client，连接运行脚本机器的 IP。局域网一般直接使用
内网 IP；公网或跨网段部署时，按 Isaac Sim 文档配置 public endpoint，并确保运行 Isaac Sim 的机器开放
`TCP 49100` 和 `UDP 47998`。同一个 Isaac Sim 实例同一时间只能连接一个 streaming client。

同样的 `WEBRTC=1` 环境变量也适用于其它 Isaac Sim 渲染入口，例如官方 traj replay、factory-scene
replay、scene preview 和主要 UR5e scene viewer。直接调用 Python 工具时使用 `--webrtc`；viewer 场景通常
同时加 `--headless`，例如 `roboassemblybench/scripts/view_task_scene.py --headless --webrtc ...`。

第一次建议先跑一个小样本，确认 Isaac Sim、路径和 mp4 编码都正常：

```bash
MAX_FRAMES=10 WIDTH=640 HEIGHT=360 STRIDE=24 \
  bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

脚本默认参数如下，都可以通过环境变量覆盖：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `CONDA_ENV` | `internutopia311` | 用哪个 conda 环境运行 Python |
| `RECIPE` | `fabrica_plumbers_block_ur5e` | 加载哪个 RoboAssemblyBench task recipe |
| `SCENE_PROFILE` | `taoyuan_grscenes_tabletop` | 加载哪个 RoboAssemblyBench scene profile |
| `SEED` | `0` | 构建 task env 时使用的随机种子 |
| `WIDTH` / `HEIGHT` | `960` / `544` | 渲染分辨率 |
| `FPS` | `30` | 输出 mp4 帧率 |
| `STRIDE` | `6` | 每隔多少个 Fabrica 源轨迹帧采样一帧渲染 |
| `MAX_FRAMES` | 空 | 限制最多渲染多少帧；空值表示渲染完整采样序列 |
| `CAMERA_OPTION` | `close` | 相机预设，可选 `close`、`front`、`far` |
| `WARMUP_STEPS` | `8` | 正式截帧前先刷新多少次渲染管线 |
| `WORLD_OFFSET` | `0.47,0,1.012` | 把 Fabrica cm 坐标轨迹映射到 task env 世界坐标时添加的米制偏移 |
| `KEEP_TASK_REPLAY_OVERLAPS` | `0` | 默认隐藏 task env 中会和回放重叠的机器人/物体；设为 `1` 保留 |
| `HEADLESS` | `1` | `1` 后台运行 Isaac Sim；`0` 使用本地 GUI，需要可用 display |
| `WEBRTC` | `0` | `1` 启用 Isaac Sim WebRTC 远程可视化；通常与 `HEADLESS=1` 一起使用 |
| `LOG_DIR` | `roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block` | Fabrica 官方轨迹目录，必须包含 `traj.npy` 和 `fixture/fixture.obj` |
| `ASSEMBLY_DIR` | `roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001/assets/fabrica_original_usd_sdf_margin_001/aligned/plumbers_block/parts` | plumbers_block 零件目录；支持当前 bundle 的 USD，也兼容旧 Fabrica OBJ 目录 |
| `ASSET_DIR` | `roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001/assets` | plumbers_block/optical board 资源根目录；缺旧式 UR5e/Robotiq OBJ 时会使用仓库内 UR5e bundle 的 USD 资源 |
| `OUTPUT` | `outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4` | 输出视频路径 |
| `FRAMES_DIR` | `outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay_frames` | 输出 PNG 帧目录 |

常用自定义示例：

```bash
CONDA_ENV=internutopia311 \
WIDTH=1280 HEIGHT=720 FPS=30 STRIDE=4 CAMERA_OPTION=front MAX_FRAMES=120 \
OUTPUT=outputs/fabrica_official_isaacsim/plumbers_block_ur5e_front_720p.mp4 \
FRAMES_DIR=outputs/fabrica_official_isaacsim/plumbers_block_ur5e_front_720p_frames \
  bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

运行时脚本会执行：

1. 用 `build_dual_franka_assembly_episode(...)` 构建 `fabrica_plumbers_block_ur5e` task env。
2. 按 `HEADLESS`/`WEBRTC` 配置启动 InternUtopia/Isaac Sim，并加载 `taoyuan_grscenes_tabletop` 场景。
3. 默认隐藏 task env 中和回放 mesh 重叠的机器人、Fabrica 零件、fixture 和 preview prim。
4. 从 `traj.npy` 读取 Fabrica 官方 body matrix 轨迹，从 `ASSEMBLY_DIR`/`ASSET_DIR` 加载零件和
   optical board，并从仓库内 UR5e bundle 加载 UR5e/Robotiq 资源；如果用户提供旧 Fabrica OBJ 目录，
   仍会优先使用 OBJ mesh。
5. 在 `/World/replay` 下创建回放 prim，按 `STRIDE` 采样源轨迹，并按 `WORLD_OFFSET` 放到任务场景中。
6. 用 Replicator `LdrColor` annotator 写出 PNG 帧，再用 `imageio`/`libx264` 编成 mp4。

默认会得到这些产物：

```text
outputs/fabrica_official_isaacsim/
  plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4
  plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.json
  plumbers_block_ur5e_official_traj_taoyuan_task_env_replay_frames/
    rgb_00000.png
    rgb_00001.png
    ...
```

其中 mp4 是可直接播放的回放视频；frames 目录保留每一帧 PNG，便于检查具体帧；同名 JSON 是本次渲染
summary，会记录：

- `mode`、`recipe`、`scene_profile`、`seed`；
- `log_dir`、`traj_path`、`assembly_dir`、`asset_dir`；
- `output_path`、`frames_dir`、`source_frame_count`、`captured_frame_count`、`written_png_count`；
- `stride`、`fps`、`camera_width`、`camera_height`、`camera_option`、`camera_position`、`camera_look_at`；
- `scene_asset_path`、`scene_asset_fallback_path`、`workspace_offset`、`world_offset_m`；
- `hidden_task_overlap_prim_paths`，即默认隐藏掉的重叠 prim；
- `limitations`，明确说明这是 kinematic visual replay，不是 Isaac PhysX 接触仿真，也不是控制器重定向执行。

检查输出是否正常：

```bash
ls -lh outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4
python -m json.tool outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.json | head -80
find outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay_frames -name '*.png' | wc -l
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of default=nokey=1:noprint_wrappers=1 \
  outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4
```

常见问题：

- `isaac-sim.sh not found`：重新运行 `bash setup_conda.sh`，输入包含 `isaac-sim.sh` 的 Isaac Sim 目录。
- `ModuleNotFoundError: No module named 'isaacsim'`：当前 shell 没有使用 `internutopia311`，或该环境没有正确安装/绑定 Isaac Sim。
- `ModuleNotFoundError: No module named 'pxr'` 或 `No module named 'omni.replicator'`：优先检查
  `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh` 是否 source 了正确的 `setup_conda_env.sh`；修正后重新激活环境。
- `FileNotFoundError` 指向 `traj.npy`、`fixture.obj` 或某个 `.obj`：说明 Fabrica 官方日志或 asset 没同步完整，按上面的
  `test -f`/`test -d` 清单逐项补齐。
- mp4 写出失败或提示 `libx264`/`ffmpeg`：在 conda 环境安装 `imageio-ffmpeg`，并确认系统 `ffmpeg` 可用。
- 渲染启动很慢：Isaac Sim 第一次 headless 启动会加载扩展和 shader cache；先用 `MAX_FRAMES=10` 做 smoke test。
- `GLFW initialization failed` 或 `failed to open the default display`：当前机器没有可用本地显示环境。使用
  `HEADLESS=1` 后台渲染，或使用 `HEADLESS=1 WEBRTC=1` 通过 Isaac Sim WebRTC Streaming Client 远程查看。
- WebRTC 客户端连不上：确认脚本输出里 `WebRTC: 1`，等待 Isaac Sim 完全启动后再连接；局域网连接使用运行脚本机器的
  IP，跨公网连接需开放 `TCP 49100` 和 `UDP 47998`。

## Online RoboBrain

RoboBrain can generate a task plan from a new natural-language instruction, validate it with
RoboChecker, compile it into a temporary RoboAssemblyBench `recipe.yaml`, and optionally call the
existing demo generator. For new task compositions it can ask the LLM for a compact skill plan
(`pick`, `place`, `insert`, `lift`, `press`, `handover`) and compile that into generated targets,
phases, attachments, and success checks instead of only replaying an existing template.

During demo generation it can also run a runtime RoboChecker that captures RGB/state observations,
validates interaction/scheduling/spatial constraints online, grounds the latest images and state
snapshots into planner hints, runs optional visual foundation-model grounding, tries a deterministic
local recipe repair, and then sends failure feedback plus recent image paths back into RoboBrain for
automatic re-planning if local repair does not recover the rollout.

Plan-only smoke run without network access:

```bash
python -m roboassemblybench.robobrain \
  "right Franka inserts a peg into the socket" \
  --mock-llm \
  --plan-only
```

Online planning with OpenAI:

```bash
export OPENAI_API_KEY=...
python -m roboassemblybench.robobrain \
  "two arms lift the steel barrier together" \
  --num-demos 1 \
  --max-trials 10 \
  --max-runtime-replans 1 \
  --record-live-video \
  --headless
```

## RoboBrain Agent App

The local Agent App provides an Ubuntu-friendly UI for the RoboBrain workflow. It shows the task
instruction, template and asset inventory, planner trace, structured plan, generated skills,
checker output, artifact files, optional Isaac demo generation, and optional LeRobot export.

Install the lightweight web dependencies if they are not already present in your environment:

```bash
pip install -r requirements/webui.txt
```

Start the app with the default `internutopia311` environment:

```bash
bash roboassemblybench/scripts/run_robobrain_agent_app.sh
```

Then open `http://127.0.0.1:7861`. By default the UI uses
`fabrica_plumbers_block_ur5e_right_base_prepare` as the executable template. If `OPENAI_API_KEY` is
not set, new jobs default to `Mock LLM` for a fast plan-only smoke run. To call the real planner,
set `OPENAI_API_KEY`, uncheck `Mock LLM`, and enable `运行仿真` only when Isaac Sim is available.
Enable `导出 LeRobot` together with demo generation to write a LeRobot-style dataset under the run
directory. UI jobs default to `roboassemblybench/outputs/robobrain_agent_app/`.

For a no-LLM walkthrough, click `Run Manual Example`. That path uses
`tasks/fabrica_plumbers_block_ur5e_right_base_prepare` directly and writes a hand-authored,
auditable trace:

- `manual_reasoning_trace.json`: task intake, template resolution, asset selection, task
  decomposition, skill formatting, static validation, simulation command plan, and LeRobot export
  plan.
- `manual_skill_steps.json`: one row per local skill with execution skill, arm, object, arm state,
  object/table target properties, position/orientation inputs, advance condition, and expected
  output.
- `manual_demo_report.md`: human-readable full-flow report for the same run.

The CLI examples above write `plan.json`, `primitive_plan.json`, `checker_report.json`, `annotation.yaml`, and
the generated `recipe.yaml` under `roboassemblybench/outputs/robobrain/<timestamp>_<task>/`. When
`--plan-only` is omitted, demo outputs are written to the run's `demo/` subdirectory. Runtime
RoboChecker artifacts are written to `runtime_feedback.json` and `runtime_observations/`.

Useful runtime flags:

```bash
--no-runtime-robochecker        # disable online checking
--no-runtime-replanning         # keep online checking, but do not re-plan after failure
--no-local-replanning           # skip deterministic local recipe repair before LLM re-plan
--no-perception-grounding       # do not summarize runtime RGB/state into the next planner prompt
--perception-visual-backend local
                                # local, owlvit, groundingdino, grounded-sam, or none
--perception-label "peg"        # add open-vocabulary detector labels
--perception-detector-model IDEA-Research/grounding-dino-tiny
                                # override the HF zero-shot detector model
--perception-sam-checkpoint /path/to/sam_vit_h.pth
                                # enable SAM masks for grounded-sam
--perception-sam2-config /path/to/sam2.yaml
--perception-sam2-checkpoint /path/to/sam2.pt
                                # enable SAM2 masks for grounded-sam
--perception-vlm-grounding      # ask the configured OpenAI vision model for semantic grounding
--runtime-checker-stride 8      # state-check interval in rollout steps
--runtime-rgb-frame-stride 24   # RGB capture interval in rollout steps
--max-runtime-replans 2         # number of feedback-driven re-plans after rollout failure
```

Visual grounding backends are optional. The default `local` backend has no model download and
produces color-region proposals. `owlvit` and `groundingdino` use the Hugging Face
`zero-shot-object-detection` pipeline. `grounded-sam` runs the detector first and then uses SAM or
SAM2 masks when `ROBOBRAIN_SAM_CHECKPOINT` or `ROBOBRAIN_SAM2_CHECKPOINT`/`ROBOBRAIN_SAM2_CONFIG`
are configured.

The local repair pass currently handles common runtime failures conservatively:

- scheduling failures: increase timeout budgets and slightly relax target-reaching tolerance;
- spatial failures: widen left/right target lanes by generating safe-lane targets;
- interaction failures: repair grasp/attach tolerances and enforce contact-gated attachment.
