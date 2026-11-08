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
# @rl 让右臂持续支撑，part_3 完成后换到紫色件/绿色底座连接处，最后才卸载验证。
# 已取消左臂临时支撑和接缝对齐：左臂开爪等待，右臂直接换抓；换抓间隙靠桌面承托。
# 当前 @rl 的等待/稳定确认窗口上限已改为 30 个物理步（0.125 s），原有更短窗口保留。
# 阶段超时、运动行程、真实接触、夹力/力矩及滑移阈值保留。
# 30 步配置相关 440 项回归通过，三个现有 RL 权重加载及配置/摘要校验通过。
# 当前已取消两销释放后的目标自转限制；轴倾斜 5°、位置、30 步稳定和抓取滑移保护保留。
# 取消自转限制后 319 项相关检查通过，尚未重新进行原生仿真。
# 最新从头运行：outputs/plumbers_wait30_full_20261108_163520/verification_results.json。
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
# 当前修复与验证范围见 docs/plumbers_block_physical_rl_fixes.md。
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
