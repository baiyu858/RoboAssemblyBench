cd /home/baiyu24/model/InternUtopia

PLUMBERS_RUN_DIR="$PWD/outputs/plumbers_rl_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir -- "$PLUMBERS_RUN_DIR"
echo "$PLUMBERS_RUN_DIR"

/home/baiyu24/APP/miniconda3/bin/conda run \
  --no-capture-output -n internutopia311 \
  python roboassemblybench/scripts/validate_fabrica_canonical_ur5e.py \
  --tasks fabrica_plumbers_block_ur5e_staged@rl \
  --seed 4906 \
  --layout-seed 4906 \
  --output-dir "$PLUMBERS_RUN_DIR" \
  --record-live-video \
  --live-video-fps 10 \
  --live-video-frame-stride 24 \
  --live-video-encoder stream

# 上述命令从 recipe 加载 part_0、part_1、part_4 的局部残差 RL；part_3 使用物理伺服。
# 策略与训练组件位于 roboassemblybench/core/rl，权重仍为试训版本。
# 相机只在新渲染帧读取，非渲染步复用上一帧；机器人状态/FK 在同一物理步内复用。
# 复位和快照恢复会清理测量缓存，真实接触、滑移和夹力控制仍逐物理步执行。
# 两销运输使用等效轴向朝向；孔轴、轴心和插入深度保留，白件和紫件朝向不变。
# 两销从取料净空点直接斜升到更高的插入悬停点，后续精细收敛/稳定检查保留。
# 两销持有时相对夹爪的转角上限为 10°，任务检查与 RL 执行检查共用该值。
# 两销插入力矩硬限均为 4 N m，软限仍为 1.2 N m。
# 策略权重、观测归一化、3 mm 平移滑移保护和两指真实接触要求保留。
# 读取方式对照：4800 步循环 189.68 s -> 184.29 s，两组阶段及最终物体/机器人状态一致。
# 上述 4800 步结果仅为读取优化短程对照；完整验证结果见下方最新记录。
# @rl 让右臂持续支撑，part_3 完成后换到紫色件/绿色底座连接处，最后才卸载验证。
# 已取消左臂临时支撑和接缝对齐：左臂开爪等待，右臂直接换抓；换抓间隙靠桌面承托。
# 当前 @rl 的等待/稳定确认窗口上限已改为 30 个物理步（0.125 s），原有更短窗口保留。
# 阶段超时、运动行程、真实接触、夹力/力矩及滑移阈值保留。
# 30 步配置相关 440 项回归通过，三个现有 RL 权重加载及配置/摘要校验通过。
# 当前已取消两销释放后的目标自转限制；轴倾斜 5°、位置、30 步稳定和抓取滑移保护保留。
# 取消自转限制后的 319 项相关检查为历史记录；当前完整仿真也已通过。
# 历史从头运行：outputs/plumbers_wait30_full_20261108_163520/verification_results.json。
# 两销 RL、右臂释放及双臂 home 均完成；最终卸载检查因第一销自转 114.85° > 45° 失败。
# 第一销中间稳定验证已经通过：要求连续 30 步，该段总计 111 步。
# 历史运行分别使用过 15°、45° 自转目标公差；这些旧报告仍保留当时的真实失败结果。
# 当前 @rl 的高位运输转向为 0.35 rad/s、1.80 rad/s²，右臂空载横移为 0.12 m/s。
# 抬升、接近等段仍用原转向设置；取料夹具保持 45° 朝向。
# 取消前的最新完整尝试通过白件、紫件插入及释放检查，第 50198 步在临时接缝对齐超时。
# 右手换抓、两销和回 home 尚未执行，不能认定整套装配已成功。
# 之前的 104 阶段及运输提速配置通过 311 项相关检查；从头仿真通过直接换抓及第一销 RL 插入和松爪。
# 该轮尚用 120 步稳定窗口，第 56334 步检查超时：末态 65/120 步；第二销及回 home 未执行。
# 结果：outputs/plumbers_fast_no_left_full_20261108_143754/verification_results.json。
# 底座、白件、紫件高位运输仿真用时较上次缩短约 36%–37%；这不代表整套装配已成功。
# 普通入口已按阶段图分配总步数预算，不需要复用 outputs 下的诊断脚本。
# 最新从头完整验证：outputs/plumbers_limited_release_full_20261110_142548/verification_results.json。
# success=true，64130 步，实际 60.32 分钟，视频 267.2 秒；两销、右臂释放、双臂 home 及卸载检查通过。
# 第二销先张爪至 0.34，实测至少 0.33；双侧销接触力连续 6 步低于 0.1 N 后撤离 60 mm，再完全张爪。
# 两销按紫件和绿件的四个孔截面对齐整根销轴，保持夹持方向；软力矩卸载降至 0.9 N m 后恢复。
# 本次为固定布局单次成功，随机布局成功率尚未评测；完整运行仍未达到 30 分钟目标。
# 当前修复与验证范围见 docs/plumbers_block_readback_shaft_release_fixes.md；历史见 docs/plumbers_block_physical_rl_fixes.md。
# 继续调试到下一件 part_3：
# python roboassemblybench/scripts/run_insertion_rl.py --mode resume \
#   --recipe fabrica_plumbers_block_ur5e_staged@rl --through-stage 3 \
#   --snapshot outputs/plumbers_part0_rl_20261028_144422_lean_physics/entry_snapshot.json \
#   --checkpoint checkpoints/plumbers_block/part0_physical_rl/policy.pt --video
# 原规划对照：--tasks fabrica_plumbers_block_ur5e_staged@reverse_part0。
# 正向规划对照：--tasks plumbers_block。

# 单因素提速对照：将上述 --tasks 的值替换为下面之一。
# 仅空载高位长距离横移：0.06 -> 0.10 m/s；其余空载阶段和负载旋转不变。
# fabrica_plumbers_block_ur5e_staged@fast_empty
# 负载旋转：0.10 -> 0.15 rad/s，0.30 -> 0.45 rad/s²；空载速度不变。
# fabrica_plumbers_block_ur5e_staged@fast_rotation
# 都是待原生验证的实验配置，不代表完整装配已成功。
# 需要全量诊断时显式加 --full-debug-traces，并自行配置 BEAM_*_TRACE_PATH。
# 默认只写 rollout_performance.json 汇总，接触/夹持/滑移保护仍逐步执行。
# part_0 反向插入：fabrica_plumbers_block_ur5e_staged@reverse_part0。
# 当前原生对照需保留扩展接触诊断依赖：指定 BEAM_CONTACT_TRACE_PATH、BEAM_SUPPORT_CONTACT_TRACE=1，
# 并加 --full-debug-traces；其他全量调试路径保持关闭。
# 本轮轻量启动器：outputs/plumbers_reverse_part0_20261026_124303/run_lean.py。
# 它在 part_0 松爪、退出并完成无外力沉降核验后保存 part0_checkpoint.json，再结束仿真；不评判整个装配。
# 启动器固定本轮输出目录，作为运行记录保留；不要重新执行并覆盖结果。
# 常规命令替换为上述反向 recipe 会继续运行后续零件，不带这个单零件停止检查点。
