# Fabrica 七任务自动成功率评测：实现交接

适用基线：`main@2739e27`，两台 UR5e + Robotiq 2F-85，Isaac Sim 中的七个 `fabrica_*_ur5e_staged` 任务。本文件是评测设计与代码交接，不代表七任务物理评测器已经实现或通过实机仿真验证。

## 先说结论

**可以复用数据生成器的目标、零件清单、位姿误差计算和逐件诊断；不能直接把数据生成器的 `success` 当成七任务模型的物理装配成功率。**

建议同时输出两个不混用的指标：

1. **兼容成功率（legacy）**：复用当前逐件目标位姿、已释放、静止条件和连续成功步数，便于与现有数据生成结果对照。
2. **物理成功率（strict，主指标）**：在上述条件之外，要求零件由真实接触支撑，结束时机械臂已退出，且整个评测过程中没有固定抓取关节、零件位置锁、运动学搬运或运行时位姿/速度写入。七任务最终装配结构须保持稳定。

若评测允许固定关节和位置锁，应明确把结果称为“辅助仿真成功率”，不要称为物理抓取成功率。

## 七任务范围与装配关系

源数据为 [`canonical_tasks.json`](../roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json)。表中 `A→B` 表示元数据里的 `socket_part=A, move_part=B`，用作最终装配关系，不要求策略严格按表中顺序操作。

| 任务 | recipe 名称中的任务段 | 零件数 | 底座 | 期望装配关系 |
| --- | --- | ---: | ---: | --- |
| Beam | `beam` | 5 | 6 | 6→2，6→3，3→1，2→0 |
| Car | `car` | 6 | 2 | 2→1，1→3/0/5/4 |
| Cooling manifold | `cooling_manifold` | 7 | 1 | 1→3/6/5/4/2/0 |
| Duct | `duct` | 8 | 0 | 0→1，1→7/6/3/4/5/2 |
| Gamepad | `gamepad` | 6 | 3 | 3→0/1/2/5/4 |
| Plumbers block | `plumbers_block` | 5 | 2 | 2→0/3，3→1/4 |
| Stool circular | `stool_circular` | 9 | 2 | 2→3/7/5/1，3→6，7→4，1→0，5→8 |

具体 recipe 路径统一为 `roboassemblybench/tasks/fabrica_<任务段>_ur5e_staged/recipe.yaml`。七个 recipe 的 `collection.layout_seeds` 都列出 `4906、485、34、12`；这些是现有采集布局，不应自动视为未见过的泛化测试布局。Beam 另有连续物理配置，其他六个任务当前仍走固定连接抓取。

## 现有代码在哪里

| 环节 | 位置 | 评测同学需要知道的事 |
| --- | --- | --- |
| 官方零件与装配图 | [`canonical_tasks.json`](../roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json) | 七任务的底座、零件、`move_part/socket_part` 和资产来源。 |
| 配方入口 | [`tasks/`](../roboassemblybench/tasks/) 中七个 `fabrica_*_ur5e_staged/recipe.yaml` | 任务名、场景、种子和任务专用参数。 |
| 配方加载及指纹 | [`task_registry.py`](../roboassemblybench/core/task_registry.py) 的 `load_task_recipe` | 继承 YAML、场景合并、编译七任务，生成 `recipe_fingerprint`。评测须记录该指纹。 |
| 目标和成功条件编译 | [`fabrica_canonical.py`](../roboassemblybench/core/fabrica_canonical.py) 的 `compile_fabrica_canonical_recipe`、`_compile_targets_and_phases` | 生成 `targets`、`phases`、逐件 `success` 条件。非 Beam 抓取阶段使用 `fixed_joint`；释放阶段可锁定位姿。 |
| Beam 专用编译 | [`beam_physical_compiler.py`](../roboassemblybench/core/beam_physical_compiler.py) | Beam 使用纯物理抓取，但其阶段及最终验证与另外六任务不同。 |
| 创建仿真任务 | [`scene_builder.py`](../toolkits/factory_dual_franka_assembly/scene_builder.py) 的 `build_dual_franka_assembly_episode/batch` | 把 `success` 变成 `success_criteria`；`policy_evaluation_mode=True` 用于模型闭环评测。 |
| 逐件判定与诊断 | [`factory_dual_franka_assembly_task.py`](../internutopia_extension/tasks/factory_dual_franka_assembly_task.py) 的 `_check_success`、`_success_diagnostics` | 现有可复用的位姿、释放、静止检查。`calculate_metrics` 输出诊断与终止原因。 |
| 自动抓取交互 | 同一任务类的 `_policy_evaluation_interactions` | 策略评测模式默认自动抓取/释放，并复用生成阶段的抓取配置；不是纯粹被动的检测器。 |
| 单任务 ACT 闭环评测 | [`evaluate_fabrica_plumbers_block_act.py`](../roboassemblybench/scripts/evaluate_fabrica_plumbers_block_act.py) 与 [启动脚本](../roboassemblybench/scripts/evaluate_fabrica_plumbers_block_act.sh) | 现有基线默认是 **`fabrica_plumbers_block_ur5e_right_base_prepare`**，不是七任务通用评测器，也不是默认的 canonical staged plumbers recipe。 |
| 评测结果汇总 | [`aggregate_fabrica_plumbers_block_act_eval.py`](../roboassemblybench/scripts/aggregate_fabrica_plumbers_block_act_eval.py) | 已实现按 seed 去重和成功数汇总；尚未完整核对每次运行的 recipe、指纹、checkpoint 与逐 seed 布局映射。 |
| Oracle/环境验证 | [`validate_fabrica_canonical_ur5e.py`](../roboassemblybench/scripts/validate_fabrica_canonical_ur5e.py) | 跑的是生成策略是否完成任务，**不是学习模型的成功率**。默认任务列表未包含 plumbers_block，显式传 `--tasks` 才会运行它。 |

现有单任务评测及聚合的基础测试见 [`test_roboassemblybench_act_evaluation.py`](../tests/toolkits/test_roboassemblybench_act_evaluation.py)；连续成功步数测试见 [`test_roboassemblybench_position_randomization.py`](../tests/toolkits/test_roboassemblybench_position_randomization.py)。

## 现有成功检测器究竟检查什么

`_check_success()` 遍历每个 `success_criterion`：

- 零件世界位姿是否落入目标容差；朝向通常也要满足容差。
- `require_released`：对象不在内部 `_attachments` 字典中。
- `require_static`：线/角速度低于阈值，或短窗口的位姿漂移满足静止替代条件。
- 若某条条件显式配置接触要求，才检查机器人和物体接触；当前七任务最终条件主要是位姿、释放、静止。

七任务编译后的最终条件都覆盖全部零件、要求 `require_released=True`、`require_static=True`，朝向容差为 `0.15 rad`。位置容差按零件编译，约 `15–62 mm`，不能把某一个统一阈值硬写进评测器。当前静止阈值是 `0.05 m/s` 和 `2 rad/s`。模型评测模式要求 `_check_success()` 连续通过 `24` 次 `is_done()` 调用才宣布成功；实际对应多少仿真秒须按评测环境的步进方式测量，不能直接由渲染帧率推算。

生成模式的 `calculate_metrics()` 可把“当前逐件条件满足”并入 `success`；策略评测模式只报告真正通过连续成功步数门槛的 `self.success`。这是现有两个流程的重要差别。

### 不能直接沿用的原因

1. **辅助机制会抬高物理成绩。** 当前六个非 Beam canonical 任务的抓取规格是 `fixed_joint`，并可在生成阶段锁定夹具零件及释放后的零件。零件位置正确、已从机械臂解除连接、速度低，不等于它能靠指垫和装配接触自行保持。
2. **当前在线评测的自动抓取也会改变环境。** `_policy_evaluation_interactions()` 默认启用：满足开度和接触条件后，主动按阶段抓取规格附着零件。因此即便阶段推进被关闭，策略仍可能得到固定连接的帮助。物理主指标应关闭这项自动附着，让策略的夹爪命令和 PhysX 接触直接作用于零件；检测器只读状态。
3. **Beam 有一个现成的评测阻断点。** Beam 的 `_check_success()` 额外要求处于脚本生成流程的最终 `beam_final_unloaded_verification` 阶段；而 `policy_evaluation_mode` 绕过阶段推进。当前 Beam 编译结果首阶段带 `beam_coordinated=True`，所以将现有单任务 evaluator 直接换成 Beam recipe，最终条件即使满足也会被该阶段门槛挡住。需要把“生成阶段完成”与“策略评测目标完成”拆成两个谓词。
4. **世界坐标不能单独代表装配。** 生成流程放下底座时可以把后续目标重定向到实测底座位置；策略评测不执行该阶段动作。评测应分别检查底座在板面的位置、其他零件相对实际底座/父件的位姿，并核对装配图，而不是只检查固定世界坐标。
5. **24 步只证明短暂满足。** 最终释放后应继续推进物理仿真并复查稳定性，防止暂时停住、随后散架或滑落。

## 建议的自动评测判定

每个 episode 固定 `task_id、recipe_fingerprint、scene_profile、资产版本、物理步长、策略 checkpoint、episode_seed、layout_seed、随机化 profile、最大步数、检测器版本`。策略只接收规定的相机/状态观测；生成器的阶段标签、真值目标和检测结果不能作为策略输入。

评测环境应提供两种明确模式，不能在报告里混写：

- **legacy 模式**：保留当前自动附着等机制，复用现有判据，只用于和既有结果对比。
- **strict 模式**：关闭自动固定关节、运行时零件锁和运动学搬运；初始化布置结束后禁止为帮助策略而写入自由零件位姿/速度。若任务因此无法稳定启动，应修场景/抓取物理，不得悄悄恢复脚本辅助。记录所有被允许的初始化写入与运行期对象状态写入。

strict 终态建议按以下顺序检查，任一失败即失败，并保存逐项诊断：

1. **完整性**：需要的全部零件都存在、没有越界或非有限状态；没有未授权固定连接/位置锁/碰撞关闭；没有运行期脚本扶持自由件。
2. **装配几何**：底座位于板面指定区域；每条官方装配边的子件相对父件位姿满足该任务独立配置的容差。以当前 recipe 的逐件容差作为初始参考，再用实际间隙和标注样本校准。
3. **真正卸载**：所有零件均脱离机械臂，两个机械臂离开装配体；最终状态没有抓取关节或对象位置锁。仅检查 `_attachments` 为空不够。
4. **支撑与稳定**：在最终张爪、机械臂退出后继续仿真，建议先用 `1 s` 稳定窗口作为实验配置；窗口内速度、位姿漂移和装配边相对位姿持续达标。底座应由板面承托；其他边根据真实连接几何检查接触/承托或等效的装配保持证据。接触形式和数值阈值需逐任务标定，不能仅用“位置接近”替代。

检测器应作为**只读观察者**实现：读取对象世界位姿/速度、机械臂状态、抓取关节/锁状态、碰撞状态和接触/支撑记录；不得在检测时修正对象状态。可复用 `get_tracked_object_states()` 的输出字段（位置、速度、`attachment`、`locked_target`、碰撞状态），但严格物理审计还需要在对象位姿/速度写入点、关节创建/删除点记录事件，不能只从最后一帧反推历史。

建议先完成各任务的**人工标注校准集**：明显成功、未完全插入、虚假位置锁、固定关节辅助、释放后滑落、零件穿透/缺失。对照这些样本确定各边的几何与支撑阈值；在阈值冻结前不要发布物理成功率。

## 运行协议与成功率口径

- 先做每任务少量 smoke 检查策略 RPC、观测、动作、重置和检测器；smoke 不计入正式成功率。
- 正式比较各模型时，对七任务使用相同且预先冻结的 episode seed 列表。建议每任务至少 `40` 次，四个布局各 `10` 次，共 `280` 次；预算允许时增加到每任务 `100` 次。失败和超时都计入分母。
- 将“现有四布局上的未见 episode seed”与“完全未见 layout seed”分成 ID/OOD 两张结果表，不混为一个成功率。后一组布局须在评测前选定，并确认未出现在训练集。
- 每个 episode 独立 Isaac 进程或经过验证的完全隔离重置；策略服务器在每条 episode 前重置内部状态。固定控制频率、动作限幅、仿真版本和资源配置。
- 任务成功率 `S_t/N_t`，七任务主汇总为七个任务成功率的算术平均（macro average）。可另报总成功数/总尝试数（micro rate），但不能用它代替 macro；同时报告各任务的失败原因与置信区间。
- 仿真/基础设施崩溃先原 seed 重试并记录；若不能恢复，结果标记 `invalid`，不得静默剔除后仍报告 `complete=true`。发布指标时同时给出计划尝试数、有效尝试数和 invalid 数。

每条结果至少输出：`task、mode、episode_seed、layout_seed、recipe_fingerprint、checkpoint_hash、detector_version、success、terminal_reason、steps、goal_diagnostics、physics_audit、settle_window、invalid_reason`。汇总器须拒绝混入不同 recipe/模型/模式/检测器版本的结果，并核对每个 seed 对应的预期 layout seed；当前聚合器只校验 layout seed 是否属于允许集合，不足以保证这一点。保存少量成功/失败视频供抽查，视频不替代自动判定。

## 交付顺序与必测反例

1. 先把 `_check_success()` 的**任务几何条件**从**生成阶段终止条件**拆开；修复 Beam 在策略模式下无法到达最终阶段门槛的问题。保持生成器原有指标兼容。
2. 将现有 plumbers ACT runner 改为按任务/recipe 参数化；strict 模式禁止自动固定抓取，新增只读物理审计与终态稳定窗口。不要把单任务脚本改名后就宣称七任务完成。
3. 加七任务配置和装配图检查，冻结种子/阈值/指纹，最后扩展聚合器生成逐任务和 macro 报告。
4. 至少覆盖以下反例：目标位姿正确但零件被锁；固定关节带着零件到位；前 `24` 步静止、之后滑落；漏一个零件；底座移位但子件相对装配正确；子件贴近目标却接到错误父件；Beam 在策略模式满足最终几何条件；评测目录混入另一 checkpoint 或 recipe。上述假阳性必须判失败或明确标记无效。

完成定义：七个 recipe 都能运行独立策略闭环；每条结果可复现并可定位失败零件/失败条件；legacy 与 strict 成功率分别报告；人工标注反例与至少一批原生 Isaac Sim 回归通过。**目前仓库只提供单任务 ACT 评测基线，不应把本文当作已实现结果。**
