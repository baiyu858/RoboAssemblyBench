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
  --live-video-fps 15 \
  --live-video-frame-stride 16 \
  --live-video-encoder stream

# 上述命令从 recipe 加载 part_0、part_1、part_4 的局部残差 RL；part_3 使用物理伺服。
# 策略与训练组件位于 roboassemblybench/core/rl，权重仍为试训版本。
# @rl 让右臂持续支撑，part_3 完成后换到紫色件/绿色底座连接处，最后才卸载验证。
# 最新完整流程仍因白色件释放后漂移而停止；局部插入通过不等于完整装配成功。
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
