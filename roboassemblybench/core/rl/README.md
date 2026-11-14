# 接触插入 RL 核心库

该库将高精度插入表示为 `InsertionTask`，供规划流程在局部窗口交给 RL。控制器、PPO、入口快照和评估不依赖零件编号、颜色、左右臂名称或固定的孔座。默认使用 55 维状态和 6 维位姿残差，几何引导和外环导纳共同执行动作。

现有原生后端复用 Isaac Sim / UR5e 的运动学和夹力驱动。其他机器人或仿真器实现 `InsertionBridge`，不用改 PPO 和几何控制。库尚未移植成 Isaac Lab 并行训练环境。

## 文件分工

| 文件 | 职责 |
| --- | --- |
| `task.py` | 插入对象、孔座、执行臂、可选支撑臂、目标、接触对象和阶段协议 |
| `control.py`、`geometry.py` | 坐标变换、导纳、动作仲裁、奖励和保护；仅依赖 NumPy |
| `runtime.py` | 共用插入控制循环，通过 bridge 读状态和发命令 |
| `bridges.py` | 机器人/仿真器接口 |
| `backends/isaac_ur5e.py` | Isaac 接触测量、UR5e 局部 IK、夹力保持及关节命令 |
| `backends/fabrica.py` | 从 Fabrica 装配元数据推导任务绑定，供 recipe 编译使用 |
| `env.py` | 原生单环境 Gymnasium 接口；仅在 reset 写状态，保留整个场景和已装配物体 |
| `snapshot.py`、`capture.py` | 执行前置流程、捕获并验证任意插入入口 |
| `ppo.py` | PPO、模型保存、继续训练和任务协议校验 |
| `validate.py` | 真实夹持、柔顺响应、卸力验证与局部评估 |
| `recipe.py`、`integration.py` | 多阶段编译及策略派发 |
| `cli.py` | 资源保护、训练流水线、局部续跑和完整流程验证 |
| `legacy.py` | 第一版实验权重和快照的读取兼容；仅此文件记录旧物体身份 |

`toolkits/factory_dual_franka_assembly/part0_rl` 已移除，不保留转发包。工具层的局部技能执行器调用本库，底层 UR5e 驱动暂由原有原子技能实现提供。

## 任务描述

以下是一个与 plumbers 无关的单臂任务示例。真实场景需已有这些物体、机器人和目标：

```python
from roboassemblybench.core.rl import InsertionTask, InsertionConfig
from roboassemblybench.core.rl.runtime import InsertionController

binding = InsertionTask(
    task_family="fixture_bench",
    stage_id="pin_slot",
    moving_object="blue_pin",
    socket_object="red_socket",
    insertion_robot="arm_beta",
    entry_target="pin_entry",
    goal_target="pin_seated",
    socket_is_static=True,
    contact_objects=("red_socket", "guide"),
    entry_phase="contact_approach",
    replace_phases=("contact_approach", "fine_insert"),
    rl_phase="learned_insert",
    settle_phase="released_settle",
)
# 控制器可接收真实原生 task，或桥接其他环境。
# controller = InsertionController(binding=binding, bridge=my_bridge,
#                                 config=InsertionConfig())
```

双臂任务再配置 `support_robot` 与 `support_object`，并令动态孔座 `socket_is_static=False`。孔座与支撑物可以不同，例如孔位于已装好的零件上，另一只手夹住承载底座。控制器分别检查孔座漂移、支撑物漂移和两手抓取滑移。

`entry_target`、`goal_target` 是**被插物体的目标位姿**，不是 TCP 位姿。默认从入口到目标推导插入轴，也可通过 `insertion_axis` 指定孔座坐标系下的轴。入口和目标须有非零局部平移距离；只旋转的任务需另设计进度定义。

`contact_objects` 可包含多个真实接触体，后端汇总法向、摩擦和相对于 TCP 的力矩。重叠过滤路径被拒绝，避免力重复计算。夹指夹力单独读取，不当作插入力。

无支撑臂时，网络的第二组夹持槽填零，实际抓取记录仍只有一个。网络两组槽的顺序始终是执行臂、支撑臂，与机器人是否叫 left/right 无关。

## 在 recipe 中使用

Fabrica canonical recipe 的 `fabrica_canonical.precision_insertion` 按装配零件 id 配置多个插入段。编译器从 assembly order 推导物体、孔座、执行臂、入口和阶段范围；`binding` 可覆盖任意字段。每个段独立加载自己的策略。

```yaml
fabrica_canonical:
  continuous_physics: true
  precision_insertion:
    '0':
      controller: residual_rl
      backend: native_contact_insertion
      checkpoint: checkpoints/example/first_insert/policy.pt
      entry_waypoint: 1
      timeout_steps: 6000
    '3':
      controller: residual_rl
      backend: native_contact_insertion
      checkpoint: checkpoints/example/second_insert/policy.pt
      binding:
        contact_objects: [fabrica_example_0, fabrica_example_2]
```

这是配置片段，需与实际 assembly/物体名及训练权重对应。当前 plumbers `@rl` recipe 为 `'0'`、`'1'`、`'4'` 配置各自的试训权重；完整连续装配尚未通过。其他任务如果仍使用 fixed joint、物体锁定或运动学搬运，必须先建立真实物理抓取环境，本库不会把它们自动转换成已验证的物理任务。

完整流程的 RL 技能可通过 `completion_stable_steps` 指定执行时的连续就位确认窗口；`InsertionController` 接受同名参数，未指定时使用权重配置的 `seated_stable_steps`。该覆盖只改变何时确认局部就位，不修改模型输入、动作、几何/受力门槛或 `controller_config`，原有权重配置一致性和摘要校验保留。当前 plumbers `@rl` 的 `physical_wait_step_limit: 30` 会在最终阶段图中生成这个执行字段，训练默认值仍为 96 步。

非 Fabrica 的 recipe 在根级声明 `continuous_physics: true`、`precision_insertion`，每个条目的 `binding` 给出完整的 `InsertionTask` 字段。阶段 key 可以是 `pin_slot` 等任意名称。编译器检查引用存在、替换范围连续、不重叠，并保留释放和卸载稳定验证。多个阶段收紧成功门槛时，不会放宽先前阶段的精度。

每段可用 `controller_config` 覆盖 `InsertionConfig` 参数。频率与控制器参数写入权重；同一物理场景中的策略须使用同一物理频率。`retreat_control` 是可选的任务后处理配置，具体退出速度不硬编码在 RL 核心中。

`torque_soft_limit` 是与 `controller_config` 同级的可选执行保护，单位 N·m，必须低于 `insertion_torque_hard`。它使用与硬限相同的物理手腕参考点，不使用虚拟销尖处的力矩。达到软限时禁止正向送入，并随接近硬限逐渐减小策略的横向和转角动作；原有导纳继续工作。硬限仍读取原始数据并立即终止。plumbers 两销的软限配置为 1.2 N·m，第一销硬限为 2 N·m，第二销通过执行配置设为 2.2 N·m；`part_0` 未启用软限。此字段不改 55 维观测、6 维动作及旧权重的控制参数，启用新保护后的成功率需重新验证。新资格报告及训练权重记录该字段，训练拒绝复用不匹配的资格报告；部署不允许移除或放宽权重中已记录的软限。

可靠夹持的 `force_relief` 允许连续的受限导纳卸载，不需要新的阶段或重试状态机。策略的横向/旋转探索被抑制；导纳修正只能沿反力卸载，禁止正向送入，每个物理步受已有柔顺线速度/角速度上限约束。支撑臂目标保持不变。桥接器的 `commanded_control_pose(robot)` 必须返回上一条关节目标对应的控制坐标系 FK，以保留静置保持偏置；不能用当前实测位姿冒充该目标。接触无效、弱侧夹力不足、持续滑移、探针异常和硬限均继续阻止运动。

当前 plumbers 的 `fabrica_canonical.hold_force_control.max_closing_travel` 设置为 `null`，关闭跨往复调节的累计正向补夹预算；`closing_travel` 仍作为诊断值累加，不再据此停止。`max_seek` 限制相对抓取成立位置的闭合端点，`max_command_lead` 限制相对实测关节的指令领先量，两者以及夹力、滑移、探针和恢复超时保护均保留。其他配置未显式设置该项时仍沿用原有累计预算。`loaded_force_travel_limit` 的具体来源由 `travel_limit_reason` 区分为累计预算、抓取位置相对闭合范围或机械闭合端点。旧入口快照保留旧抓取配置；对照复验需明确更新其抓取配置，不能只改 recipe 后假定旧快照内的限制也改变。

`released_rotation_tolerance`（rad，范围 `(0, pi]`）可单独设置松爪后的角度公差，默认沿用 `controller_config.seated_rotation_tolerance`。它用于收紧释放后的检查及最终成功判定，已有更严格的任务公差仍保留。

需要明确覆盖释放后的验收公差时，使用 `released_pose_tolerances: {position_m: 0.015, orientation_rad: 0.8726646259971648}`。这是 plumbers 白件当前的 15 mm / 50° 设置，作用于该物件首次释放后的稳定检查、后续各装配阶段的稳定检查及最终成功判定；覆盖原来的位置/转角门槛，固定 CAD 参考点、释放要求、速度和稳定窗口保留。夹持就位仍由 `controller_config` 决定，白件仍为轴向 1 mm、横向 0.5 mm、转角 1°。其他物件和 3 mm / 5° 抓取滑移保护不受影响。配置 `released_axial_rotation` 的物件仍分别检查轴线倾斜和自身轴自转，`orientation_rad` 对应轴线倾斜。

可选 `released_friction: {static_friction: 1.0, dynamic_friction: 0.8}` 在插入 binding 的 `settle_phase` 入口生效，即完成释放、撤离后改变被插物的原生碰撞材料。当前白件初始材料仍为 0.8/0.7，抓取、运输和插入期间保持基线；释放后设为 1.0/0.8，与未改变的绿件 0.8/0.7 通过 average 合并得到有效静/动摩擦 0.9/0.75。原生属性设置后核对读回值，保留恢复系数，不在运行时添加 USD API；实际读回记录在 `friction_policy_history`。`build_flow_env()` 恢复后续快照时，会在物理初始化前恢复当前 recipe 累积的摩擦配置。

任务可在相邻插入之间保留支撑抓取。plumbers 的 `physical_base_support.hold_between_insertions` 将中间检查编译为 `*_supported_settle`，只检查已释放的装配件，并逐步检查支撑抓取；最终卸载检查仍要求所有零件释放。插入 binding 的 `settle_phase` 自动匹配实际检查阶段。`--mode resume --recipe ... --through-stage <stage>` 可从入口快照运行到后续阶段的稳定检查；结果记录 `support_retained` 和局部范围，不能作为完整装配成功率。

## 捕获、验证、训练和回接

通过已配置 Isaac Sim 的 Conda 环境运行。新入口为 `roboassemblybench/scripts/run_insertion_rl.py`：

```bash
# 例：选择真实 recipe 和任意装配段，不需要新增 Python 控制器。
conda run --no-capture-output -n internutopia311 \
  python roboassemblybench/scripts/run_insertion_rl.py \
  --mode capture --recipe fabrica_plumbers_block_ur5e_staged@rl --stage 0

# 将下面的 snapshot 路径替换为本次 capture 的输出。
conda run --no-capture-output -n internutopia311 \
  python roboassemblybench/scripts/run_insertion_rl.py \
  --mode pipeline --snapshot /path/to/entry_snapshot.json --steps 2048 --episodes 3
```

模式包括 `capture`、`validate`、`baseline`、`train`、`play`、`resume`、`full`、`pipeline`。`--binding task.json` 可为自定义任务提供完整描述。`capture` 执行正常前置流程，在选定入口停下；不将入口采集标为装配成功。

`validate` 先验证复位后的真实夹持、受力退让和后退卸力。`QualificationConfig` 可调整探测决策数、动作和接触力，硬保护由控制器继续执行。PPO 训练要求资格报告的快照 SHA、控制参数和任务协议都匹配。

`baseline` 使用同一几何引导/导纳，关闭学习残差。`play` 回放确定性策略，需指定 `--checkpoint`。`resume` 从快照验证选定插入的释放与卸载稳定；`full --recipe ... --checkpoint ... --snapshot ...` 从原始场景运行完整流程，快照在该模式只提供任务绑定和控制参数，不用于跳过前置动作。

`pipeline` 在资格验证通过后训练，全部局部评估通过才另启进程运行完整流程。局部 `local_seated` 表示夹持状态下到位；张爪、撤出和卸载稳定仍须另外通过。任务停止原因和结果文件决定是否成功，进程码 0 本身不代表成功。

输出目录以北京时间日期和时间结尾，非空目录拒绝覆盖；系统可用内存低于 1.5 GiB 时由监督进程终止仿真。

## 权重兼容和现有限制

新格式 `roboassemblybench-contact-insertion-ppo-v2` 保存物体/角色/目标/接触对象协议。跨物体、跨角色或跨目标使用不匹配权重会报错；新任务需要对应训练，代码通用不代表一份权重已经学会所有任务。

旧试训权重和入口快照只由 `legacy.py` 按原任务协议读取，可在新框架内回放。旧资格报告缺少任务协议，继续训练前需重新验证。旧的 plumbers 脚本只是示例兼容入口，不包含 PPO 或控制实现。

当前原生后端是 UR5e + Robotiq，使用仿真真实状态训练，尚无跨入口扰动的泛化结论，也未完成并行化。通用接口单测与实际原生插入回放用于验证迁移；现有示例的释放后摆动仍是独立待修复项。

本次迁移验证：相关回归 331 项通过；后续核心/recipe 专项 68 项通过。旧试训权重经新库原生回放，第 658 次策略决策通过局部夹持就位；常规 recipe 执行器的 480 个物理步派发检查也通过。这些结果不等于其他物体已有可用权重，也不等于完整装配通过。


## 执行层力矩硬限

`precision_insertion.<stage>.torque_hard_limit` 可显式设置执行时的物理手腕接触力矩停止值，单位 N·m。不填写时沿用权重的记录值或 `controller_config.insertion_torque_hard`。该字段只改变原始接触力矩的硬停止检查，不修改权重、动作尺度或软限的衰减曲线；软限仍由 `torque_soft_limit` 配置，衰减曲线仍沿用权重的控制参数。硬限须为正且高于启用的软限。

接触明细的 `contact_object` 标记实际碰撞的对方零件，法向力和摩擦力的合成方法不变。更改执行硬限后，训练前需重新取得参数匹配的原生资格报告；旧权重可用于明确记录阈值的执行对照，不代表较大阈值已经得到训练验证。


## 多截面孔轴约束与滑移补夹

`precision_insertion.<stage>.axis_alignment` 是执行层选项，不改变 55 维观察、权重角色协议或既有动作尺度。配置销的局部轴线 `shaft_origin_m / shaft_axis`，并在 `sections` 中配置至少两个不同轴向高度的孔截面，每项为物体名、局部孔心 `center_m` 和局部孔轴 `axis`。运行时读取这些零件的当前位姿，计算销轴与各截面交点的偏心，联合求解横向与倾斜微调。`radial_clearance_m - margin_m` 是允许继续送入的轴偏心范围；剩余夹持就位条件仍须满足。

几何间隙、拟合和报警独立配置：`radial_clearance_m` 使用实际网格间隙；`fit_strategy: minimax` 优先压低最大截面偏心，且不由报警阈值选择算法；`alarm_tolerance_m` 只控制共同孔轴不可行的连续报警窗口，缺省沿用几何允许值。plumbers 两销分别为 0.25 mm 几何间隙、0.02 mm 裕度和 0.5 mm 报警阈值。拟合仍不能进入 0.23 mm 范围时禁止送入，不会因为尚未达到报警阈值而继续顶压。

`hold_force_control.unload_contact_force` 单独定义有限卸载所需的真实双指接触力，缺省沿用 `low`。plumbers RL 设置为 1 N；正常送入仍须达到原夹力恢复条件。缺少接触、硬滑移、强侧过载或额外支撑物件失去接触时仍禁止卸载。该设置不改变每步退让速度、总退让范围与手腕硬限。

`physical_junction_support.precision_hold` 对绿件在紫件坐标系中的实际孔位参考点进行检查，同时保留首次复合抓取的固定相对位姿参考。绿件可独立请求有限补夹，不再只靠紫绿合力判断。当前在累计漂移 0.075 mm 时、且绿件夹力不足目标时请求补夹，累计漂移超过 0.25 mm 或孔位参考点误差超过 0.35 mm 时暂停送入；绿件每指最低保持 4 N，补夹目标 7 N，仍服从紫绿合计每指 16 N 的软限。连续稳定 24 步恢复，480 步内无法恢复会报告 `support_precision_recovery_timeout`。这不是姿态锁定，也不会重新捕获漂移后的参考点或自动执行尚未验证的双臂重新摆正动作。

7 N 是预防性补夹目标，4 N 是该支撑物件的最低夹力门槛。最低夹力、真实接触、孔位精度及普通抓取保护均合格时，预防性补夹使用 `support_preload`，可继续累计运动就绪稳定步数；没有足够夹力或精度不合格时仍暂停并执行恢复计时。合计指爪负载触及软限后暂停预防补夹，回落到 `max(target, soft_limit - 0.25 * (unload - soft_limit))` 才恢复；当前 16/20 N 设置对应 15 N，减少贴着软限反复补夹、卸力。实际夹力不足的 `preload_required` 可以请求必需补夹，但继续受软限、硬限及行程限制。预防补夹达到行程边界时停止该次补夹，不将可用支撑误报为夹持恢复失败。

孔轴与支撑精度初次修复的相关回归 341 项通过。原生静置对照使用相同初始零件位姿、相同固定机械臂指令和原始抓取参考：10 秒后累计漂移由 0.430 mm 降至 0.200 mm，新控制峰值为 0.207 mm。已有明显错位的复位场景在 480 步后报告支撑精度恢复超时，没有将错位姿态重新设为基准。对齐初态下，第一销局部回放在 3499 步后通过夹持就位，进入释放阶段，末态最大截面偏心 0.047 mm；该回放未触发手腕力矩软限。低于送入力门槛时的有限卸载由记录工况的数值回归覆盖，尚未验证新的高力矩原生工况、释放稳定性与完整装配。

预防补夹与强制恢复分离后的相关回归 424 项通过。同一完整回放停机快照、相同初始零件位姿下，旧控制约 2 秒后重复夹力恢复超时，新控制完成 10 秒支撑并保持就绪，绿件弱侧约 6.03 N，累计漂移约 0.210 mm。按原流程续跑已通过原停机阶段，到第一销闭爪时累计漂移约 0.268 mm，超过保留的 0.25 mm 门槛，触发独立的支撑精度恢复超时；此时夹力恢复暂停计数为 0。原先的补夹计时冲突已消除，完整装配仍未通过。

机器人链接的位姿查询使用只读路径：`FactoryDualFrankaAssemblyTask._robot_rigid_body_by_suffix()` 优先复用场景已登记的刚体，未登记的手指、手掌等链接通过 `toolkits/factory_dual_franka_assembly/robot_link_pose.py` 绑定现有 Physics SimulationView 的精确路径读取视图。运行时不再构造 `SingleRigidPrim`，因此不会因读位姿而重写 `xformOpOrder`、刚体 API 或休眠阈值。物理初始化前只读 USD 位姿，初始化后读取实时物理状态；保留环境偏移、wxyz 四元数格式和物理帧缓存，物理视图失效或重置时重新绑定读取句柄。读取器只保存在任务缓存，清理任务时释放，不加入机器人的可写刚体列表。

只读链接修复的相关回归 375 项通过。匹配历史右手接触初始化条件的局部续跑中，正式代码完成第一销抓取并进入抬升，累计支撑漂移保持约 0.210 mm；首次左手手指查询及两手手掌查询没有产生 USD 属性改写，物理读取与 USD 位姿一致。直接冷复位旧故障快照时，重建接触后的初始累计误差约 0.262 mm，已经超过保留的 0.25 mm 门槛；不能将该初态与匹配初始化的对照混为完整装配验证。

后续从正常初始化执行的完整运行 `outputs/plumbers_readonly_full_20261114_211956/` 已通过 105 阶段、64,279 步，最终 `success: true`。白件、紫件、两销装配和释放、双臂 home 及最终无夹持连续 30 步稳定检查均通过；没有使用快照或初始化兼容补丁。绿件最大累计漂移约 0.179 mm，最大孔位参考点误差约 0.234 mm。三路视频各 2,678 帧、10 FPS、无丢帧并完整解码，实际耗时约 66.18 分钟。该布局登记为 `layout_001`，历史成功布局登记为 `layout_000`，见[布局图、坐标与复现说明](../../../docs/plumbers_block_layouts.md)。这次仅验证单布局的一次完整成功；RL 段手腕力矩记录均为 0，测力有效性仍需独立核查。

约束作用于几何引导、残差叠加和柔顺偏移后的最终目标。微调以最后实际发布的关节目标对应的控制位姿为基准，保留静态保持偏差；被限制的虚拟目标做抗累积处理。既有指令预瞄、抓取滑移、接触有效性、力与力矩保护继续执行。已有权重可以做明确记录参数的执行实验；训练前须重新取得 `axis_alignment` 参数相同的原生资格报告。

`hold_force_control.slip_preload: true` 允许在早期滑移增长时，弱侧夹力尚处于 `low` 与 `target` 之间便恢复原 `target`。强侧上限、开度行程、每步速度和机械臂暂停条件保持生效。默认关闭，在 plumbers 的 `@rl` 中启用。入口复位只将当前 recipe 的夹持力配置更新到恢复的抓取上，固定抓取参考点不变。

## 释放后的轴线倾斜与轴向自转

`precision_insertion.<stage>.released_axial_rotation` 可配置物件局部轴 `axis` 和自转公差 `twist_tolerance_rad`。启用时，对目标与当前姿态的相对四元数做 swing/twist 分解：`released_rotation_tolerance` 约束轴线倾斜，自转由 `twist_tolerance_rad` 单独约束。未配置时仍检查原有总转角。

当释放后的目标姿态只需约束轴倾斜时，将 `twist_tolerance_rad` 设为 `3.141592653589793`（π）。自转误差取最短等效角 `[0, π]`，因此该设置接受任意轴向自转；仍保留 `axis` 和倾斜公差。当前 plumbers `@rl` 两根销使用此设置，释放稳定窗口与抓取过程的角度保护独立保留。

该选项只作用于该次插入之后的释放稳定检查和最终成功判定，不改变夹持就位控制、抓取滑移保护、权重或固定位置参考点。诊断保留总转角 `orientation_error`，并增加 `orientation_tolerance_mode: swing_twist`、`swing_error/swing_tolerance` 和 `twist_error/twist_tolerance`，单位均为弧度。
