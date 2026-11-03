# plumbers_block 物理夹持与高精度插入修复说明

本文记录相对于 `c37e31b`（PR #12 合并后的 main）的代码变化、当前配置和验证结果。已有的纯物理抓取、固定抓取参考点和基础安全检测继续使用；本次在此基础上加入通用插入 RL、接触与夹持修复、持续支撑及运行性能改进。

**当前结论：白色件的抓取、抬升、RL 插入和首次释放检查通过；两根销各有局部续跑成功记录。最近一次从初始场景运行的完整装配仍未通过，原因是白色件释放后持续漂移。** 局部续跑、夹持就位和完整装配成功是不同的验证范围。

## 1. 修复内容与代码入口

| 问题 | 本次修改 | 主要文件 |
| --- | --- | --- |
| 低速抬升时夹爪移动、零件近乎不动，随后触发滑移 | 初始化保留零件稳定化门槛；首次负载运动前把该零件门槛改为 0，后续不重新开启。探针不再尝试运行时切换场景稳定化标志 | `toolkits/factory_dual_franka_assembly/manipulation_stabilization.py`、`roboassemblybench/core/plumbers_physical.py`、`internutopia_extension/tasks/factory_dual_franka_assembly_task.py` |
| 读取静态桌板时改变了碰撞表示 | plumbers 静态桌板使用三角网格碰撞；移除场景层中冲突的 SDF API。读取静态物体位姿使用只读包装，不再把桌板提升成刚体 | `roboassemblybench/core/fabrica_canonical.py`、`internutopia_extension/objects/usd_object.py`、`toolkits/factory_dual_franka_assembly/static_object_pose.py` |
| 单侧夹力已高，另一侧仍弱，闭合控制却不再补夹 | 闭合阶段按弱侧决定预紧，强侧过载仍先卸力；保持阶段加入原始夹力骤降响应和可选的滑移预紧 | `toolkits/factory_dual_franka_assembly/plumbers_block_ur5e_skills.py`、`toolkits/factory_dual_franka_assembly/physical_hold_force.py` |
| 薄件尚有间隙却被名义闭合预设挡住 | 区分名义闭合预设和物理最小开度；plumbers 显式允许剩余闭合行程，仍受机械端点、命令领先量和夹力保护约束 | `internutopia_extension/configs/robots/ur5e.py`、`internutopia_extension/controllers/gripper_controller.py`、`roboassemblybench/core/plumbers_physical.py` |
| 紫色件抓取易碰夹具，或指腹未充分贴合 | 校准原生指腹中心；紫色件抓取中心偏移 20 mm；闭合前按目标及实测位姿检查整段夹爪闭合轨迹与夹具间隙 | `roboassemblybench/core/grasp_clearance.py`、`roboassemblybench/core/plumbers_physical.py` |
| 白色件负载保持等待 8 N，无法进入正常抬升 | 增加逐移动零件的夹持配置；仅 `@rl` 的白色件改为下限 4 N、恢复目标 5 N | `roboassemblybench/core/plumbers_physical.py`、任务 `recipe.yaml` |
| 后续插入期间底座或已装件缺少支撑 | 右手跨插入阶段持续支撑；紫色件装好后，先由左手临时支撑，右手撤离低台阶并改夹紫色/绿色连接处，再执行两销插入 | `roboassemblybench/core/plumbers_support.py`、`roboassemblybench/core/plumbers_junction.py`、`toolkits/factory_dual_franka_assembly/compound_grasp.py` |
| 规划终点提前释放、插入受力后完全锁住 | 收紧就位门槛；引入残差 RL 与导纳控制；有效夹持下允许受限卸载，软力矩限提前限制顶压 | `roboassemblybench/core/rl/`、`roboassemblybench/core/plumbers_physical.py` |
| 第二销孔轴偏心，顶压及抓取滑移 | 同时约束紫色件、底座的四个孔截面；不满足孔轴包络时先校正横向和倾斜，限制继续送入 | `roboassemblybench/core/rl/axis_alignment.py`、`runtime.py`、任务 `recipe.yaml` |
| 运输转向预算耗尽、最后撤离或回 home 不完整 | 旋转预算计入预瞄制动约束；最终空手撤离保留已发布保持偏置；双臂以实测关节误差完成 home 检查 | `roboassemblybench/core/plumbers_physical.py`、`roboassemblybench/core/plumbers_support.py`、`toolkits/factory_dual_franka_assembly/empty_wrist.py` |
| 调试采集、逐帧渲染及视频编码拖慢运行 | 默认关闭全量轨迹日志；录像时每 16 个物理步渲染；有界队列后台 FFmpeg 编码；记录各阶段实际耗时 | `roboassemblybench/scripts/validate_fabrica_canonical_ur5e.py`、`toolkits/factory_dual_franka_assembly/generate_demos.py`、`render_schedule.py`、`streaming_video.py`、`rollout_performance.py` |

## 2. 稳定化与接触的具体处理

### 2.1 初始化稳定化与操作阶段分开

零件初始 `physxRigidBody:stabilizationThreshold` 为 `0.001`。该属性用于数值稳定化，不是成功率评测的位置公差，也不是夹持力阈值。连续对照发现，它会在本任务的低速抬升中使零件近乎停留原位，夹爪则继续运动。

当前编译器在底座及每个移动零件的首次负载运动入口添加 `object_stabilization_thresholds`，将对应零件的门槛设为 `0`，并核对属性读回。初始化沉降仍使用原值，抬升、搬运和释放之后保持 `0`。此操作不写零件位姿、速度或抓取参考点，不改变重力、摩擦、碰撞或普通睡眠设置。

入口快照续跑时按当前 recipe 恢复截至入口阶段的稳定化设置。快照复位会恢复物理状态，因此局部续跑与从初始场景连续运行仍须分别验证。

### 2.2 场景设置必须在首次物理初始化前应用

`physics_scene` 支持显式设置 `enable_stabilization` 和 `enable_external_forces_every_iteration`。配置由 `SimConfig` 传入，在 World 创建后、首次 reset 前应用；共享同一场景的任务必须请求相同设置，非法字段或读回失败直接报错。

`@rl` 同时开启上述两项。零件操作阶段仍通过逐刚体门槛排除数值冻结，接触探针只读取预先准备好的报告。静态接触过滤器展开为实际碰撞子路径，动态对象按刚体 owner 去重，避免容器路径漏报或重复计力。

## 3. 夹持控制和支撑交接

### 3.1 当前夹力配置

下表的力均按每侧夹指计算；连接处支撑将同一侧在紫色件和绿色底座上的真实载荷汇总，同时分别要求两物体接触有效。

| 使用范围 | 弱侧下限 | 恢复目标 | 软限 / 卸力阈值 / 硬限 |
| --- | ---: | ---: | --- |
| 通用负载保持、底座低台阶支撑、销及紫色移动件 | 6 N | 8 N | 14 / 18 / 40 N |
| `@rl` 白色 `part_0` 负载保持 | 4 N | 5 N | 14 / 18 / 40 N |
| 换抓期间左手临时支撑 | 4 N | 6 N | 继承其支撑配置 |
| 右手紫色/绿色连接处支撑 | 8 N | 10 N | 16 / 20 / 40 N |

**5 N 是白色件恢复夹持时的目标，不是实际夹力上限。** 指令领先、接触形变和外载荷会影响实测夹力；最近一次搬运末段约为 9.82 / 9.84 N。初始白色件闭合仍使用 4 N 门槛，未改成此前试验失败的 6/8 N 闭合条件。

夹力平滑、上下限滞回、24 步稳定确认、48 步恢复渐进和 480 步恢复超时继续使用。原始夹力突然低于下限的一半时提前触发恢复，避免滤波掩盖快速失力。可选 `slip_preload` 在早期滑移增长时恢复原目标，不提高目标夹力。

`max_closing_travel: null` 关闭累计正向补夹指令预算；累计量仍记录。相对抓取位置的 `max_seek`、物理闭合端点、每步速度、命令领先、夹力硬限及 3 mm / 5° 固定参考点滑移保护继续生效。

两销可在探针有效、仅接触缺失、间隙不超过 0.5 mm 时进入最多 24 个物理步的补夹恢复。该状态不被当作有效抓取，机械臂不继续运输；出现滑移、异常或超时仍失败，主动张爪不触发恢复。

### 3.2 插入期间保持支撑，释放后单独验证

右手先夹绿色底座外露低台阶，完成白色件及紫色件装配后，换抓到两者连接处。额外接触体保留自由动力学，不创建固定约束；控制器分别检查其真实接触和固定相对位姿漂移。

中间 `*_supported_settle` 检查已释放装配件的目标位姿及相对底座稳定窗口：连续 120 步，漂移不超过 0.5 mm / 0.5°，右手支撑同时有效。这里不把主动支撑下的高频接触速度波动当作无外力静止。

最终 `*_unloaded_settle` 仍要求所有零件释放，线速度不超过 0.01 m/s、角速度不超过 0.1 rad/s，并满足位姿及稳定窗口。双臂最终还须回到配置的 home，实测关节误差不超过 0.025 rad、持续 24 步；张爪和撤离不替代 home 检查。

## 4. 通用残差 RL 与高精度插入

实现统一放在 `roboassemblybench/core/rl/`，不再保留 `toolkits/factory_dual_franka_assembly/part0_rl` 包。`InsertionTask` 描述被插物、孔座、执行臂、支撑臂、接触对象、入口和目标；`InsertionBridge` 隔离原生环境。当前可运行的后端是 Isaac Sim / UR5e + Robotiq，尚未迁移成 Isaac Lab 并行训练环境。

控制链路为：几何引导提供基础对齐和送入，策略输出 6 维小幅平移/转角残差，导纳和接触保护约束最终指令，再由 IK 驱动关节。策略使用 55 维仿真状态，PPO 当前在 CPU 更新；不是从图像端到端学习整个装配流程。

任务 recipe 的 `precision_insertion` 原生声明插入段，编译器检查阶段连续性、引用和重叠，只替换指定窗口，保留后续张爪、撤离和稳定验证。

| 物件 | 当前执行方式 | 关键设置 |
| --- | --- | --- |
| 白色 `part_0` | 反向接近 + 残差 RL | 松爪后总转角公差 3°；夹持就位转角仍为 1° |
| 紫色 `part_3` | 物理伺服 | 位置 ≤1.5 mm、轴向 ≤1 mm、横向 ≤0.75 mm、姿态 ≤3°，连续 48 步后释放 |
| 第一销 `part_1` | 独立权重的残差 RL | 手腕力矩软限 1.2 N·m、硬限 2 N·m |
| 第二销 `part_4` | 独立权重的残差 RL + 四截面孔轴约束 | 软限 1.2 N·m、执行硬限 2.2 N·m；允许轴偏心为 0.25−0.08=0.17 mm |

两销释放后将轴线倾斜与绕自身轴自转分开检查：倾斜公差 5°、轴向自转公差 15°。夹持就位及抓取滑移门槛不随之放宽。位置误差按固定 CAD 特征点计算，避免允许自转时把偏置坐标原点的运动误判为插入位移。

`force_relief` 下，夹持有效时允许沿反力方向做限速、限幅卸载，禁止继续正向顶压；软力矩限在硬停止前限制送入和探索。接触无效、弱侧夹力不足、滑移或原始力矩硬限仍阻止运动。

新权重保存任务协议、控制器参数和版本；不匹配任务或权重校验失败时在启动前报错。现有试训权重不构成收敛、随机布局泛化或跨零件共用策略的证据。`baseline` 模式关闭学习残差，保留同一几何引导及导纳，用于评估 RL 本身的贡献。

## 5. Recipe 合并与运行性能

只维护 `roboassemblybench/tasks/fabrica_plumbers_block_ur5e_staged/recipe.yaml`，通过 profile 复用公共配置：

| 选择器 | 用途 |
| --- | --- |
| `fabrica_plumbers_block_ur5e_staged` | 默认正向物理规划流程 |
| `fabrica_plumbers_block_ur5e_staged@reverse_part0` | 白色件反向插入规划对照 |
| `fabrica_plumbers_block_ur5e_staged@rl` | 三个局部 RL 段、持续支撑及连接处换抓 |
| `fabrica_plumbers_block_ur5e_staged@fast_empty` | 仅将高位空载长距离段从 0.06 提到 0.10 m/s |
| `fabrica_plumbers_block_ur5e_staged@fast_rotation` | 负载旋转从 0.10/0.30 提到 0.15 rad/s / 0.45 rad/s² |

`recipe_profiles.py` 支持继承和旧任务名别名。回放保存 profile 选择器，避免把 RL 轨迹重新加载成默认规划 recipe。旋转提速对照保留当前基线 16145 步的阶段预算，避免同时改变速度和允许用时。

物理步长保持 `1/240 s`。验证脚本录像默认 stride 16，对应每 16 个物理步渲染一次、15 fps 视频；框架内部 `rendering_interval` 表示额外跳过的步数，所以实际周期 16 对应值 15。降低渲染频率不降低接触、安全及物理控制的检查频率。

`stream` 编码以容量受限队列把 RGB 帧送入后台 FFmpeg，不先落盘全部 PNG；队列满时施加背压，不静默丢帧，编码失败显式报错。`png` 后端可作为显式对照。原始数据集采样仍保留相机与动作的时序约束。

默认移除全量 `BEAM_*` 调试采集环境变量，需要时通过 `--full-debug-traces` 显式保留。必要的接触测力、夹力恢复和滑移保护仍执行；`rollout_performance.json` 汇总阶段运行、控制、渲染及录像耗时，不能只用视频长度推断仿真运行速度。

## 6. 已有原生验证和当前未解决问题

### 6.1 稳定化对照

同一初始状态连续运行，原门槛对照在约 3.009 mm 滑移时失去抓取，底座仅上升约 0.178 mm；逐步唤醒未解决。将底座操作阶段门槛设为 0 后，首段抬升约 88.515 mm，最大缓存滑移约 0.093 mm。

本地记录：`outputs/plumbers_lift_cause_20261102_175617/root_cause_report.json`。这是冻结问题的隔离对照，不代表之后所有接触稳定性问题都已修复。

### 6.2 最新完整流程尝试

记录：`outputs/plumbers_white_force45_20261102_214730/verification_results.json`，seed / layout seed 均为 4906，从初始场景运行 112 阶段图。

| 指标 | 结果 |
| --- | --- |
| 白色件实际抬升 | 88.445 mm |
| 白色件搬运及插入期间最大采样抓取点滑移 | 0.207 mm |
| 同期最大相对转角 | 1.220° |
| 白色件 RL 就位、松爪、首次支撑稳定检查 | 通过 |
| 最终停止 | 第 48098 步，`assemble_01_part_3_supported_settle` 超时 |
| 导致失败的对象 | 已释放的白色 `part_0` |
| 白色件固定特征点误差 | 8.723 mm，允许 1.118 mm |
| 白色件总转角误差 | 21.438°，允许 3° |
| 紫色件自身位置 / 转角误差 | 0.087 mm / 0.074°，自身位姿通过 |
| 两销与最终双臂 home | 此次尚未执行到 |

白色件在紫色件插入前就已漂移到约 7.05 mm / 19.70°，不能把全部漂移归因于最后插入顶压。当前需继续定位释放后的白色件与底座、桌板之间的长期接触稳定性。最新结果未证明问题已解决，也未用放宽成功门槛将其判为通过。

三视角原视频位于上述目录的 `episode_0000_live_videos/`；`white_released_pose_history.json` 保存漂移出现的阶段顺序。

### 6.3 两销局部验证

| 范围 | 本地记录 | 结果及边界 |
| --- | --- | --- |
| 第一销入口快照续跑 | `outputs/plumbers_release5deg_20261031_160616/verification_results.json` | 插入、释放及 120 步支撑稳定通过；位置误差约 0.331 mm、总转角约 2.863° |
| 第二销入口快照续跑 | `outputs/plumbers_axis_grip_20261101_152617/verification_results.json` | RL 就位及释放通过；最大孔轴偏心约 0.074 mm、最大采样抓取滑移约 0.658 mm；随后撤离在该历史版本仍失败 |
| 最终双臂 home 局部回放 | `outputs/plumbers_terminal_home_20261101_160802/verification_results.json` | 双臂实测关节 home 检查通过；不代表全部装配成功条件通过 |

这些结果来自不同局部回放，不能拼成一次连续完整装配成功。`outputs/` 中的报告和视频是本地实验产物，不随本次代码提交上传。

## 7. 权重、复现和提交范围

提交前相关回归共 **684 项通过**，覆盖 RL 核心、recipe 编译、夹力控制、稳定化、接触报告、抓取净空、双臂支撑、释放角度、home 及录像性能；另有两条现存 Pydantic 弃用警告。Python 源码语法检查与 `git diff --check` 通过。本次整理提交没有重新运行整段 Isaac Sim，原生验证结论来自第 6 节列出的既有记录。

运行 `@rl` 前需另行安装三个试训权重。它们位于被忽略的 `checkpoints/`，本次不把模型二进制推送到 GitHub。

| 文件（相对于仓库根目录） | recipe 中要求的 SHA-256 |
| --- | --- |
| `checkpoints/plumbers_block/part0_physical_rl/policy.pt` | `cb50cfd29b6999866c1c2ec0792010fafdb7cee76c6207594bab8b7b9cf838f9` |
| `checkpoints/plumbers_block/part1_physical_rl/policy.pt` | `a9618a5654aa36c003c58e955e54190e4fa5aa088dbd3a167f773525437dc457` |
| `checkpoints/plumbers_block/part4_physical_rl/policy.pt` | `dadaaf8c9a261172c85b2f264b9f83c2502afcdbb765f7c4ab2834f62f0cd440` |

在已配置 Isaac Sim 的 Conda 环境中，从仓库根目录运行：

```bash
PLUMBERS_RUN_DIR="$PWD/outputs/plumbers_rl_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir -- "$PLUMBERS_RUN_DIR"

conda run --no-capture-output -n internutopia311 \
  python roboassemblybench/scripts/validate_fabrica_canonical_ur5e.py \
  --tasks fabrica_plumbers_block_ur5e_staged@rl \
  --seed 4906 --layout-seed 4906 \
  --output-dir "$PLUMBERS_RUN_DIR" \
  --record-live-video --live-video-fps 15 \
  --live-video-frame-stride 16 --live-video-encoder stream
```

该命令执行当前 recipe；成功与否需读取结果文件中的任务判定和停止阶段。缺少权重会在场景启动前报错，系统可用内存保护保持 1.5 GiB。资格验证、采集入口、训练、基线、局部续跑及完整流程的详细用法见 [RL 核心库说明](../roboassemblybench/core/rl/README.md)；历史诊断细节见 [实验记录](plumbers_part0_physical_rl.md)。

本次提交包括运行代码、配置和文档；不提交 `/tests/` 的任何变更，不提交 `outputs/`、视频、权重或 `.chart-data-*` 临时 PPT 文件。本地回归验证使用了保留在工作区的测试，其中部分为未提交的新测试；GitHub 上的测试文件不是本地完整验证集。
