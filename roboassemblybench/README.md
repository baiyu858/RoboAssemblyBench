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
  当前机器可用路径示例为 `/home/baiyu24/APP/isaac-smi`；换机器时改成自己的 Isaac Sim 路径。
- 已拉取本仓库以及 RoboAssemblyBench/Fabrica 相关资产。

推荐安装方式是直接运行仓库根目录下的安装脚本：

```bash
cd /path/to/RoboAssemblyBench
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

`env_vars.sh` 里应该 source 当前 Isaac Sim 目录下的 `setup_conda_env.sh`。如果路径不对，重新运行
`bash setup_conda.sh`，或手动修正该文件后重新打开 shell/重新激活环境。

## Fabrica 七任务零件质量

七个 canonical 任务的 46 个零件统一从 `assets/Fabrica/canonical_7_bundles/canonical_tasks.json`
读取逐件 `mass_properties`。质量按官方 OBJ 的封闭实体体积乘以 bundle 中的官方密度
`1250 kg/m³` 计算，原始厘米坐标先乘 `0.01` 转成米。圆凳 OBJ 的法线/纹理接缝仅在体积计算时
合并重复顶点；不补洞，不使用凸包或外接盒体积，也不修改仿真资产。

| 任务 | 零件编号与质量（g，展示值四舍五入） |
| --- | --- |
| beam | 0: 12.8723；1: 12.8723；2: 9.4695；3: 9.4695；6: 39.3188 |
| car | 0: 17.8907；1: 366.9239；2: 437.5361；3: 17.8907；4: 17.8907；5: 17.8907 |
| cooling_manifold | 0: 5.2643；1: 311.9919；2: 8.0075；3: 8.0074；4: 5.2643；5: 8.0075；6: 8.0075 |
| duct | 0: 305.1790；1: 82.8637；2–7: 各约 11.2395 |
| gamepad | 0: 2.5190；1: 10.8455；2: 11.5711；3: 176.0852；4: 6.2548；5: 2.5190 |
| plumbers_block | 0: 141.1946；1: 20.2292；2: 294.9544；3: 74.1420；4: 20.2292 |
| stool_circular | 0/4/6/8: 各约 9.0112；1/3/5/7: 各约 29.7105；2: 101.9735 |

通用和独立 Beam 编译器都把该质量以 kg 显式传入 `UsdObjCfg.mass`，避免加载器缺省的 1 kg；
不再将 Beam 质量人为截断到 0.25 kg。质量数据缺失或与密度、体积不一致时直接报错。
记录包含源 OBJ 的 SHA-256；完整 metadata 构建脚本也会生成质量，避免重建时丢失。
这些是均匀密度模型值，不是实物称重值；不因此放宽滑移容差或修改夹持力。
修改质量会改变物理轨迹，旧数据/旧成功率不能替代新配置的仿真验证。

只读复算与回归检查（需要本地 official OBJ 和 canonical bundle，不需要规划 pickle）：

```bash
python roboassemblybench/scripts/recompute_fabrica_part_masses.py --check
python -m pytest --confcutdir=tests/toolkits tests/toolkits/test_fabrica_mass_properties.py -q
```

需要更新 metadata 时给复算脚本显式传入 `--output <metadata.json>`，默认不写文件。

## plumbers_block 纯物理生成模式

`fabrica_plumbers_block_ur5e_staged` 的 recipe 已启用
`fabrica_canonical.continuous_physics: true`。这是在线物理轨迹生成，和下文官方轨迹 replay 不是同一条执行路径。

- 配置：`tasks/fabrica_plumbers_block_ur5e_staged/recipe.yaml`。
- 编译入口：`core/fabrica_canonical.py`；物理阶段转换：`core/plumbers_physical.py`。
- 运行时：`../internutopia_extension/tasks/factory_dual_franka_assembly_task.py`。
- 闭合、运输夹持控制：`../toolkits/factory_dual_franka_assembly/plumbers_block_ur5e_skills.py`。

5 个零件分别保留其抓取接触区域。抓取要求两指对目标零件的接触力探针有效、接触相对且持续稳定；
不创建抓取固定关节，不关闭指爪碰撞，不用夹爪开度或几何靠近代替接触力。
接触报告在 `FactoryDualFrankaAssemblyTask.set_up_scene()` 加载完机器人和任务物体后、
runner 创建本轮 PhysX 视图前统一准备。运行时使用 force-only `RigidContactView`，
`prepare_contact_sensors=False`，不再临时给掌部/零件/静态夹具添加报告 API。
这避免了开启 `BEAM_SUPPORT_CONTACT_TRACE=1` 后，运行中新增 API 导致实例化黑色掌部消失的问题。
未预配置的测力源或过滤目标会明确报告探针无效，不自动补加 API，也不退化为几何抓取成功。
预计抓取宽度只决定何时减速，未接触时继续低速闭合到配置的安全行程限制。
闭合阶段要求双指分别对目标零件形成真实相向接触，各指夹持力处于 4–12 N，
且零件位姿连续 96 步保持稳定；单指触碰、瞬时力峰值、开度到位或空夹都不能判定抓取成功。
初始化后零件不再被 fixture lock 固定，运输不再使用运动学搬运。
平移滑移以抓取成立时两指对目标零件的受力接触斑质心中点为参考：仅在登记抓取时计算一次，
转换并固定在零件坐标系中，同时保存当时的 TCP 坐标。后续比较同一零件物质点相对 TCP 的变化，
不跟随新的接触点移动、不在换阶段或解锁时重置；新的一次抓取登记才建立新参考。
接触点数据缺失、无效、不是目标配对或缓存截断时不登记抓取；已登记参考丢失时拒绝保持，不能重采样补齐。
`hold_position_slip_reference: grasp_point` 仅在 plumbers 物理模式启用；Beam 等未启用的模式保持原点判据。
平移阈值仍为 3 mm，独立相对转角阈值仍为 5°，真实接触要求不变。
纯物理搬运的旋转参考另设 `physical_rotation_speed: 0.10` rad/s（5.73°/s）、
`physical_rotation_acceleration: 0.30` rad/s²（17.19°/s²）、
`physical_rotation_lookahead: 0.010` rad（0.573°）。这些是 recipe 中可复现实验参数，不是已标定最优值。
关闭旧的角度 warm-start 累积，按实际 physics dt 在 SO(3) 上限速、限加速度；
提前为目标和预瞄边界刹车，参考状态跨搬运阶段保留，仅在接受运动命令后提交。
首帧从上一次实际发出的关节指令的 FK 开始，不能从带 IK 残差的旧目标姿态直接起转。
参考预瞄使用硬上限的 90%，剩余量留给 IK 残差；最终关节指令的 FK 仍须满足完整预瞄上限。
物理模式使用有界局部 IK，角度求解容差收紧到 1e-8 rad；求解失败时只回退平移，不改写受限旋转参考，
避免原来的 1e-3 rad IK 容差把每步小角度指令变成间歇跳变。Beam 和非物理模式不启用此分支。
局部 IK 始终以上次发出的关节命令为参考，不改用落后的实测关节作种子，
避免后级关节步长裁剪再次改写旋转参考。
跟踪异常或 IK 恢复改写了受限旋转参考时拒绝推进，不通过突然重置参考或扩大阈值继续运行。
旋转预算随限速扩大；这不修改装配成功阈值。限速对象是旋转参考，不等于实际刚体角速度或接触力矩控制。
`rotation_command_trace.jsonl`（随 `BEAM_MOTION_TRACE_PATH` 启用）记录参考速度、加速度、预瞄，以及最终 FK
预瞄、参考残差和差分速度/加速度；最终 FK 与受限参考的角度残差超过 5e-8 rad 时拒绝推进。
意外中断后不补积遗漏的时间步，并记录 `restarted_after_gap`。
`hold_trace.jsonl` 同时记录 `position_slip`（固定抓取点）和 `object_origin_position_slip`（原始模型原点），
接触轨迹和 attachment 快照包含完整 `hold_grasp_reference`，便于核对是否始终使用同一参考。
每次放下依次执行：实际张爪并卸载 → 退出 → 回到 park → 连续 120 个任务步的无夹持稳定检查。
底座稳定后只重定位后续目标，不改写底座姿态或速度。最终成功必须通过最后的无夹持稳定检查。
本次没有重新标定原有装配目标的位置/角度容差；生成器成功率不应直接当作严格装配精度。

夹爪目前使用实验起点 `kp=7500`、`kd=173`、静/动摩擦 `1.0/0.8`。
`gripper_drive_max_effort: null` 保留资产上限，启动时必须通过 native readback；不再默认写入 `25000`。
日志包含实际增益、effort 上限和指爪碰撞形状的材质绑定信息。
初始闭合参数在 `core/plumbers_physical.py` 的 `contact_policy` 中，仍需用 plumbers 零件实测标定；
代码/离线测试通过不代表 5 个零件已全部完成物理装配。

闭爪稳定判定可用 recipe 的 `fabrica_canonical.close_force_stability` 单独启用，其他任务默认不变。
当前采用完整 5 个物理步的滑动均值（240 Hz 下约 21 ms）：两侧原始力和均值都达到 4 N 才进入计数，
进入后用 0.5 N 滞回，允许夹力在 3.5–4 N 间波动；原始力低于 3.5 N 或超过 12 N 立即清零。
这会将已进入稳定带后的最低验收力降至 3.5 N，不是声称整个窗口始终大于 4 N；参数仍需物理复测。
真实目标双指接触丢失、探针无效、漏步/倒退或抓取丢失后会丢弃旧均值，必须重新建立完整窗口。
96 步连续稳定计数和固定窗口内的 1 mm / 0.02 rad 位姿检查不变，同一物理步不能重复累计。
滤波只用于判定，不改变原始力驱动控制、12 N 卸力或 40 N 上限，也不修改后续 3 mm / 5° 滑移保护。
`close_control.jsonl` 增加 `force_stability`，记录原始/滤波夹力、进入/退出门槛、窗口样本数及清零原因。
设置 `close_force_stability.enabled: false` 可回到原来逐步检查 4–12 N 的判定。

搬运阶段另用 recipe 的 `fabrica_canonical.hold_force_control`，与上述初始闭爪判定独立。
当前试验参数为：弱侧低于 6 N 触发补夹，补到 8 N 停止；软上限 14 N，卸力/暂停门槛 18 N，
硬停止上限 40 N。这些是自定义仿真实验参数，不是 Isaac 或 Robotiq 的额定力上限。
`soft_limit_action: limit_and_slow` 将轻微超限的夹力调整与机械臂暂停解耦：14–18 N 内不再补夹，
弱侧滤波力达到 8 N、瞬时力至少 6 N 且没有持续滑移时，按 0.025 rad/s 小幅卸力；
夹持已稳定时继续运动，速度随超限幅度逐渐降至 `soft_limit_motion_scale: 0.5`。
弱侧需要恢复、持续滑移或达到 18 N 时仍暂停，瞬时力超过 18 N 时卸力；超过 40 N 终止。
旧配置默认 `soft_limit_action: pause`，保留到软上限就暂停的策略；其他任务不自动启用新策略。
不能用继续合拢去强行补偿明显的接触不对中。
正常补力用 0.02 s 时间常数滤波，卸力和过载检查仍看原始力，不能让滤波隐藏尖峰。
每个物理步最多更新一次，关节闭合速度 0.025 rad/s、领先量 0.001 rad、相对原始命令的
闭合行程和累计补夹行程均限于 0.03 rad。失去目标接触、探针无效或状态缺失时直接报错。
夹力恢复期间保持上次实际发布的机械臂关节目标，不推进 IK/旋转缓存，也不重设固定抓取参考点。
初次放行或真正暂停后连续稳定 24 个物理步才允许移动，再用 48 步平滑恢复路径速度；
轻微超限不清空已有稳定/恢复计数，持续不能放行超过 480 步则失败。
固定参考点滑移在 48 步内继续增长至少 0.1 mm 时也暂停，补力目标仍为 8 N，不因滑移无限加力。
这只是早期暂停条件，原有 3 mm / 5° 拒绝条件不变。阶段切换不重置力控制状态和抓取参考。
控制器位于 `../toolkits/factory_dual_franka_assembly/physical_hold_force.py`。
`loaded_force_trace.jsonl` 随 `BEAM_HOLD_TRACE_PATH` 写出原始/滤波力、控制模式、滑移增长、
暂停步数及恢复速度比例。将 `hold_force_control.enabled` 设为 `false` 可对照原有搬运力带控制。
先保持质量、摩擦和抓取位置不变，对比静态悬持三秒与原路径搬运，不能用离线测试代替物理验证。

当前 plumbers 物理 recipe 显式启用 `fabrica_canonical.physical_idle_hold_anchor: true`。
只对空闲的 `assembly_robot`（当前为左臂）防下垂：初始化沉降阶段不启用，右臂控制始终不变。
编译阶段用 `physical_idle_hold_robot` 明确指定左臂，左臂有主动技能或运动目标的阶段不启用。
进入静置时保存最后已发布的原生关节目标；没有有效目标时只采样一次实测关节角。
静置期间持续发送同一位置目标和零目标速度，不再每步用实测角度重设目标、累积下垂。
连续静置阶段共用该目标；离开适用阶段、主动运动、负载支撑或避让接管时清除旧目标，任务重置也会清除。
plumbers 仅缓存/发送/清除目标，不使用 Beam 的负载丢失与重新解锁状态机；负载夹持仍由原有接触和夹力控制负责。
这不是关节锁定：不直接写关节状态，不改增益、力矩上限或重力，实际保持误差仍需用仿真核验。
`idle_hold_trace.jsonl` 随 `BEAM_MOTION_TRACE_PATH` 输出固定目标、实测角度和跟踪误差；
也可用 `BEAM_IDLE_HOLD_TRACE_PATH` 单独指定文件。Beam 的启用开关保持独立，其他任务不自动启用。

当前 plumbers recipe 还启用 `fabrica_canonical.physical_base_support.enabled: true`：
底座放下、独立沉降并重定位目标后，右臂重新夹住底座 `fabrica_plumbers_block_2` 的 +X 外露低台阶。
每次左臂取件前依次执行右臂接近、预张爪、侧向进入、真实双指夹紧、夹力就绪检查；不抬升底座。
空载接近采用粗到精分段：先原地抬高 TCP 至底座原点上方 0.30 m，在高处转到侧夹姿态，
横移至预接近点上方，再垂直降低到距抓取位姿 80 mm 的预接近点；最后低速侧向进入。
长段参考速度为 0.06 m/s、转向为 0.35 rad/s，降低高度为 0.025 m/s，最后侧向进入为 0.01 m/s。
参考从最后已发布关节指令的 FK 连续推进，关节速度/加速度受限，不再每步从实测位置重启微小步长。
每段在指令速度归零后再切换，避免下一段重置参考速度造成瞬时加速度突变。
粗段/降低/精接近的位置预瞄分别限制在 15/5/3 mm；超过跟踪边界直接失败，不通过改写机器人或零件状态恢复。
该分支只允许空载机械臂，真正闭爪后仍使用原来的接触、夹力和固定抓取参考点保护。
支撑接触区域限定在底座局部坐标 x=50–80 mm、y=-18.7342–21.2658 mm、z=0–20 mm，
闭合轴为局部 Y。另用独立的侧夹位姿，不改变底座搬运抓取 944 或左臂插入路径。
`grasp_center_m: [0.074, 0.0012658, 0.022]` 是倾斜指腹的参考中心，不是要求在 z=22 mm 的空气中证明接触；
30° 倾角与 143 mm 指腹距离使指腹末段覆盖低台阶侧面。它是待物理验证的几何候选，不能据此声称已经抓稳。
真实力接触仍按原有闭合门槛验证，抓取登记后的实际固定接触参考也必须落在低台阶区域内。
左臂取件、搬运、插入、张爪、退出及 park 期间，右臂持续发送已登记的固定关节目标并调节夹力。
右臂先更新本步夹持控制；夹力恢复期间左臂不推进 IK、旋转/闭合状态或阶段完成条件，
左手已经抓住零件时仍维持自己的夹力。支撑接触失效、指令缺失或底座从固定世界参考漂移超过 3 mm / 5° 则失败。
左臂已释放并退出后，右臂实际张爪、确认目标接触卸载、沿接近方向反向退出并 park，
再进入原来的无夹持稳定检查。左臂释放阶段只要求左手零件释放，不能要求此时仍被右手夹住的底座也释放。
每个插入循环均重新建立/释放支撑，最后仍须五个零件全部无夹持稳定；不创建约束、不关碰撞、不改重力或物体状态。
`plumbers_support_trace.jsonl` 随 `BEAM_HOLD_TRACE_PATH` 输出就绪状态、底座固定参考漂移和拒绝原因。
阶段编排位于 `core/plumbers_support.py`，运行期联锁位于 `../toolkits/factory_dual_franka_assembly/plumbers_support.py`。
禁用 `physical_base_support.enabled` 可与原来的单臂插入流程对照；Beam 和其他任务不自动启用。

保持控制与接入测试：

```bash
python -m pytest --confcutdir=tests/toolkits -q \
  tests/toolkits/test_physical_hold_force.py \
  tests/toolkits/test_plumbers_contact_close.py \
  tests/toolkits/test_plumbers_physical.py \
  tests/toolkits/test_plumbers_support.py \
  tests/toolkits/test_beam_idle_hold.py
```

在仓库根目录运行单任务验证（输出目录请选新的，避免覆盖已有验证结果）：

```bash
conda run --no-capture-output -n internutopia311 python \
  roboassemblybench/scripts/validate_fabrica_canonical_ur5e.py \
  --tasks plumbers_block --seed 4906 --layout-seed 4906 \
  --output-dir outputs/plumbers_physical_validation_01 \
  --worker-timeout-seconds 600 --record-live-video
```

检查 `validation_summary.json`、`validation.log` 以及生成的视频。内存保护中止、接触探针无效、
力超限、掉落或超时都不能算成功。保留默认内存保护，不要为了跑通而关闭接触检测或重新加锁。
专项离线测试：`python -m pytest --confcutdir=tests/toolkits tests/toolkits/test_plumbers_physical.py -q`。

## 从零完整复现 plumbers_block UR5e 渲染

本节给出从干净机器拉仓库到生成视频的完整流程。目标是复现：

```bash
bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

最终产物默认写到：

```text
outputs/fabrica_official_isaacsim/
  plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4
  plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.json
  plumbers_block_ur5e_official_traj_taoyuan_task_env_replay_frames/
```

### 1. 拉取代码分支

使用包含 RoboAssemblyBench 复现脚本和资产下载脚本的分支：

```bash
git clone -b codex/add-scene-lights https://github.com/baiyu858/RoboAssemblyBench.git
cd RoboAssemblyBench
```

如果已经 clone 过仓库，切换并更新同一个分支：

```bash
git fetch origin codex/add-scene-lights
git checkout codex/add-scene-lights
git pull origin codex/add-scene-lights
```

### 2. 创建并激活 conda 环境

先确认本机 Isaac Sim 目录存在这些文件：

```bash
ls /path/to/isaac-sim/isaac-sim.sh
ls /path/to/isaac-sim/python.sh
ls /path/to/isaac-sim/setup_conda_env.sh
```

然后运行安装脚本。脚本询问环境名时输入 `internutopia311`：

```bash
bash setup_conda.sh
conda activate internutopia311
```

补齐下载和渲染脚本使用到的 Python 包：

```bash
python -m pip install -e .
python -m pip install huggingface_hub requests tqdm numpy pyyaml imageio imageio-ffmpeg trimesh scipy
which ffmpeg
which ffprobe
```

基础导入检查：

```bash
conda run -n internutopia311 env PYTHONNOUSERSITE=1 python -c \
  "import numpy, imageio, trimesh, scipy, yaml, huggingface_hub, requests, tqdm, internutopia, internutopia_extension; print('python deps ok')"
```

Isaac Sim headless 检查：

```bash
conda run -n internutopia311 env PYTHONNOUSERSITE=1 python -c \
  "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); import omni.replicator.core as rep; from pxr import UsdGeom; print('isaac sim ok'); app.close()"
```

### 3. 下载 Hugging Face 大资产

大目录不进 GitHub，按原目录结构放在 Hugging Face dataset：

```text
https://huggingface.co/datasets/baiyu858/InternUtopia-repro-assets
```

国内网络推荐使用镜像站：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python roboassemblybench/scripts/download_repro_assets_from_hf.py \
  --repo-id baiyu858/InternUtopia-repro-assets
```

如果需要代理，先设置代理再下载：

```bash
# 按本机代理端口调整；不需要代理时不要设置这两行。
export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HF_ENDPOINT=https://hf-mirror.com

python roboassemblybench/scripts/download_repro_assets_from_hf.py \
  --repo-id baiyu858/InternUtopia-repro-assets
```

脚本会把 HF 上的资产直接下载回仓库根目录，不需要解压 zip。默认同步这些路径：

```text
third_part/Fabrica/
third_part/factory_dual_franka_peg_transfer/
IsaacLab/
recordings/
manifest.json
```

如果只想先验证 Fabrica 相关资产，可以只下载对应路径：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python roboassemblybench/scripts/download_repro_assets_from_hf.py \
  --repo-id baiyu858/InternUtopia-repro-assets \
  --include 'third_part/Fabrica/**' \
  --include 'manifest.json'
```

### 4. 验证资产是否齐全

先检查 GitHub 仓库内随代码分发的默认渲染资产。这些是
`render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh` 默认读取的路径：

```bash
test -f roboassemblybench/tasks/fabrica_plumbers_block_ur5e/recipe.yaml
test -f roboassemblybench/scenes/profiles/taoyuan_grscenes_tabletop.yaml
test -f roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block/traj.npy
test -f roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block/fixture/fixture.obj
test -d roboassemblybench/assets/Fabrica/official_replay_assets/fabrica/plumbers_block
test -f roboassemblybench/assets/Fabrica/official_replay_assets/optical_board.obj
test -d roboassemblybench/assets/Fabrica/official_replay_assets/ur5e/visual
test -d roboassemblybench/assets/Fabrica/official_replay_assets/robotiq_85/visual
```

再检查 HF 下载的大资产。下面这些文件用于和本机完整目录保持一致，也能覆盖其他 Fabrica/IsaacGym
脚本的复现需求：

```bash
test -f manifest.json
test -f third_part/Fabrica/logs/codex_plumbers_block_ur5e_official/plumbers_block/traj.npy
test -f third_part/Fabrica/isaacgym/IsaacGym_Preview_4_Package.tar.gz
test -f third_part/Fabrica/isaacgym/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so
test -f third_part/Fabrica/simulation/build/lib.linux-x86_64-cpython-39/redmax_py.cpython-39-x86_64-linux-gnu.so
test -d third_part/factory_dual_franka_peg_transfer
test -d IsaacLab
test -d recordings
```

如果某条 `test` 命令失败，说明代码分支或 HF 资产没有同步完整。先重新拉分支，再重新执行资产下载命令。

### 5. 先做小样本 smoke test

无显示器或远程服务器建议显式设置 `HEADLESS=1`。先渲染 10 帧，确认环境、资产路径和 mp4 编码都正常：

```bash
HEADLESS=1 MAX_FRAMES=10 WIDTH=640 HEIGHT=360 STRIDE=24 \
  bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

检查输出：

```bash
ls -lh outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4
python -m json.tool outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.json | head -80
find outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay_frames -name '*.png' | wc -l
```

### 6. 运行完整渲染

smoke test 正常后运行完整命令：

```bash
HEADLESS=1 \
  bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

有本地图形界面并希望打开 Isaac Sim GUI 时可以去掉 `HEADLESS=1`，脚本默认是 GUI 模式：

```bash
bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

完整渲染结束后用 `ffprobe` 检查视频流：

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of default=nokey=1:noprint_wrappers=1 \
  outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4
```

### 7. 复现一致性注意事项

- 不要把 `LOG_DIR` 改到不匹配的 Fabrica 日志目录。默认脚本使用
  `roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block`。
- 不要手动打开 `KEEP_TASK_REPLAY_OVERLAPS=1` 做复现对比。默认值 `0` 会隐藏 task env 里和回放 mesh
  重叠的机器人、fixture 和零件，避免视觉上出现两套机器人或两套组件。
- 默认 `WORLD_OFFSET=0.47,0,1.012` 是当前 plumbers_block UR5e replay 对齐到 task env 的偏移。
  改动它会直接导致轨迹和桌面/夹具/零件错位。
- 这个脚本是 kinematic visual replay，不是 PhysX 接触仿真，也不是控制器重定向执行。不要把“机械臂乱飞”
  当成控制器策略问题优先排查，先确认代码分支、默认资产路径、`KEEP_TASK_REPLAY_OVERLAPS` 和
  `WORLD_OFFSET` 是否和上面一致。

## 下载 Hugging Face 大资产

GitHub 分支只保存可直接随仓库分发的代码、小型配置和必要 replay 资产。完整本机复现还可能需要
`third_part/Fabrica`、`third_part/factory_dual_franka_peg_transfer`、`IsaacLab` 或 `recordings` 等大目录；
这些目录不适合直接放进普通 GitHub 仓库，超过 GitHub 单文件/仓库体量限制时放在 Hugging Face：

```bash
python roboassemblybench/scripts/download_repro_assets_from_hf.py \
  --repo-id baiyu858/InternUtopia-repro-assets
```

国内网络建议使用 HF 镜像端点：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python roboassemblybench/scripts/download_repro_assets_from_hf.py \
  --repo-id baiyu858/InternUtopia-repro-assets
```

如果当前 `huggingface_hub` 版本对 `hf-mirror.com` 的 metadata 校验失败，下载脚本会自动退回到
镜像站 resolve URL 的流式下载；用户侧仍然使用上面的命令即可。

默认会把 HF dataset repo 中的这些路径下载回仓库根目录：

```text
third_part/Fabrica/
third_part/factory_dual_franka_peg_transfer/
IsaacLab/
recordings/
```

如果只需要 Fabrica 第三方目录，可以限制 include：

```bash
python roboassemblybench/scripts/download_repro_assets_from_hf.py \
  --repo-id baiyu858/InternUtopia-repro-assets \
  --include 'third_part/Fabrica/**'
```

维护者上传本机大资产时使用：

```bash
python roboassemblybench/scripts/upload_repro_assets_to_hf.py \
  --repo-id baiyu858/InternUtopia-repro-assets
```

上传需要先提供 Hugging Face token，例如设置 `HF_TOKEN` 或运行 `huggingface-cli login`。如果需要尝试镜像端点，
同样可以设置 `HF_ENDPOINT=https://hf-mirror.com`；如果镜像写入失败，切回默认 Hugging Face endpoint 后重试。

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
test -d roboassemblybench/assets/Fabrica/official_replay_assets/fabrica/plumbers_block
test -f roboassemblybench/assets/Fabrica/official_replay_assets/optical_board.obj
test -d roboassemblybench/assets/Fabrica/official_replay_assets/ur5e/visual
test -d roboassemblybench/assets/Fabrica/official_replay_assets/robotiq_85/visual
```

默认完整渲染命令：

```bash
cd /path/to/RoboAssemblyBench
bash roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_task_env_isaacsim.sh
```

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
| `LOG_DIR` | `roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block` | Fabrica 官方轨迹目录，必须包含 `traj.npy` 和 `fixture/fixture.obj` |
| `ASSEMBLY_DIR` | `roboassemblybench/assets/Fabrica/official_replay_assets/fabrica/plumbers_block` | plumbers_block 零件 OBJ 目录 |
| `ASSET_DIR` | `roboassemblybench/assets/Fabrica/official_replay_assets` | UR5e、Robotiq、optical board 等共享 mesh 根目录 |
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
2. 启动 InternUtopia/Isaac Sim headless 环境，并加载 `taoyuan_grscenes_tabletop` 场景。
3. 默认隐藏 task env 中和回放 mesh 重叠的机器人、Fabrica 零件、fixture 和 preview prim。
4. 从 `traj.npy` 读取 Fabrica 官方 body matrix 轨迹，从 `ASSEMBLY_DIR`/`ASSET_DIR` 加载零件、UR5e、
   Robotiq 和 optical board mesh。
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
`UR5e assembly Template` as the executable reference. If `OPENAI_API_KEY` is
not set, new jobs default to `Mock LLM` for a fast plan-only smoke run. To call the real planner,
set `OPENAI_API_KEY`, uncheck `Mock LLM`, and enable `运行仿真` only when Isaac Sim is available.
Enable `导出 LeRobot` together with demo generation to write a LeRobot-style dataset under the run
directory. UI jobs default to `roboassemblybench/outputs/robobrain_agent_app/`.

For a no-LLM walkthrough, click `Run Manual Example`. That path uses the local UR5e assembly
reference task directly and writes a hand-authored,
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
