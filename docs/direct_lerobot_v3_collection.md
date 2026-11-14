# 仿真采集与后台直接写入 LeRobot v3

实时 RGB-D 采集现在可以直接生成官方 LeRobot v3 数据，不再先编码 raw MP4、再解码转存。物理控制和成功检测保持原来的流程。后台写入进程使用独立 Python 环境，不向 Isaac 环境安装另一套 PyTorch。

## 数据字段

| 数据 | 字段或文件 | 说明 |
| --- | --- | --- |
| 三路 RGB | `observation.images.front/left_wrist/right_wrist` | H.264/H.265 流式编码；默认前视 960×540，腕部 640×480，保留宽高比 |
| 三路深度 | `sensors/depth/*.u16.bshuf.zst`、`index.json` | uint16 毫米深度，0 为无效值，bitshuffle + zstd 无损压缩；与对应 RGB 的尺寸、帧号一致 |
| 双臂 EE 与夹爪 | `observation.state`、`observation.eef_pose`、`observation.gripper_state` | 16D 状态、14D EE 位姿、2D 夹爪开度；四元数 wxyz，位置单位米 |
| 专家动作 | `action` | 与原采集契约相同：下一采样帧的绝对 EE 位姿及夹爪目标；末帧保持当前位姿 |
| 关节 | `observation.joint_state/joint_velocity/joint_effort` | 沿用每臂前 7 个 DOF 的 14D 字段；UR5e 为 6 个臂关节及资产中的第 7 个 DOF，不能解释成 7 轴 UR5e |
| 实测关节力矩 | `observation.joint_measured_effort` | PhysX 沿关节运动轴投影的实测力矩；`joint_effort` 则来自已有 effort 接口或 SDK applied effort，二者不能混为一谈 |
| 完整关节位置 | `observation.full_joint_state` | 两个 articulation 的全部 DOF；实际 DOF 名称写入 `meta/info.json` 的 `full_joint_names` |
| 手腕六维力 | `observation.wrist_wrench` | 仅读取显式六维 wrench 接口；没有接口时 availability=0，不将多个刚体的三维接触力拼成六维力 |
| 手腕关节反力 | `observation.wrist_joint_reaction` | PhysX 在 wrist_3_link 处的关节反力/力矩，链接局部坐标系，包含重力和惯性反力；与 RL 的零件接触力矩区分 |
| 双臂四指接触力 | `observation.finger_contact_force` | 顺序为左臂左/右指、右臂左/右指；读取当前步已经存在的测力缓存，单位 N，不创建额外探针 |
| 可用性 | 对应 `*_available` | 关节、wrench、反力和指爪测力都有可用性标记；0 表示没有有效测量，不能当成真实零受力 |
| 阶段与时序 | `observation.simulation_step/phase_index/phase_step/subtask_index/substage_index/waiting_state/handoff_state` | 与图像同一采样步采集，保留 240 Hz 物理步编号 |
| 零件位姿、回放 | `observation.tracked_object_position/orientation`、`replay/episode_000000.npz` | 零件名称、完整机器人关节位置和低维回放轨迹 |
| 碰撞与过程标注 | `observation.collision_signal`、`annotations/episode_000000.json` | 沿用原始检测信号和装配阶段标注 |

深度通过 `meta/info.json` 中的 `roboassemblybench_sidecars.metric_depth` 索引，不是 RGB 视频，不会被普通视频压缩破坏。官方 `LeRobotDataset` 直接读取 RGB 和 Parquet；深度使用 `roboassemblybench.datasets.media_codecs.read_depth_chunk()` 按同一 `frame_index` 读取。

## 命令与默认行为

`collect_fabrica_plumbers_block_2k.py` 和并行管理器默认 `--dataset-format lerobot-v3`。如需旧 raw 模式，显式传入 `--dataset-format raw`。`--trajectory-only` 仍走原始低维回放采集。

92 上已配置：

- 仿真：`~/miniconda3/envs/robo-assembly/bin/python`
- 后台写入：`~/miniconda3/envs/robo-assembly-data/bin/python`
- 激活仿真环境后，`ROBOASSEMBLYBENCH_LEROBOT_PYTHON` 自动指向后台环境；也可通过 `--lerobot-python` 显式指定。

本地未指定写入器 Python 时，优先寻找同一 Conda envs 目录下的 `robo-assembly-data`，其次是现有 `roboassemblybench-act`。读取某个已完成分片及其深度：

```python
import json
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from roboassemblybench.datasets.media_codecs import read_depth_chunk

root = Path('/path/to/episode_000000_lerobot_v3')
dataset = LeRobotDataset(repo_id='local/roboassemblybench-direct-v3', root=root, video_backend='pyav')
sample = dataset[0]
streams = json.loads((root / 'sensors/depth/index.json').read_text())
stream = streams['observation.images.front']
depth_mm = read_depth_chunk(root / stream['path'], stream, frame_index=0)
```

单条完整采集：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robo-assembly
cd ~/work/RoboAssemblyBench
python roboassemblybench/scripts/collect_fabrica_plumbers_block_2k.py \
  --conda-env robo-assembly \
  --recipe fabrica_plumbers_block_ur5e_staged@rl \
  --dataset-format lerobot-v3 \
  --dataset-fps 10 --dataset-frame-stride 24 --rendering-fps 240 \
  --num-episodes 1 --max-attempts 4 --layout-seeds 4906 \
  --worker-timeout-seconds 7200 \
  --output-dir /mnt/nfs01/baiyongjie/RoboAssemblyBench/data/lerobot/ur5e/plumbers_block/<本次唯一目录>
```

原生 Isaac worker 可直接传 `--record-lerobot-v3`、`--lerobot-writer-python /path/to/python`。同时录制 `--record-live-video` 会另外编码一份观看用视频；正式采集只用数据集视频时无需此参数。

## 落盘与多卡

每个 episode 生成独立、可以直接由官方加载器读取的 v3 数据集分片：

```text
<batch>/episode_000000_lerobot_v3/
  meta/info.json, stats.json, tasks.parquet, episodes/...
  data/chunk-000/file-000.parquet
  videos/<camera>/chunk-000/file-000.mp4
  sensors/depth/index.json, *.u16.bshuf.zst
  annotations/episode_000000.json
  replay/episode_000000.npz
  roboassemblybench_episode.json
  .roboassemblybench_v3_complete
```

旁边的 `episode_000000_cartesian_raw/` 只保留小体积的轨迹、标注和采集恢复凭据，不重复写 RGB 或深度。凭据中的 `videos`、`depth` 指向 v3 中的同一份文件。采集清单中每个成功 episode 的 `lerobot_v3.root` 给出对应的标准 v3 根目录。

整个批次/多卡输出目录是分片集合，不是一个可以直接传给 `LeRobotDataset(root=...)` 的单一根目录。此实现先保证每回合完整、可恢复、互不写冲突；没有在采集结束时自动合并这些分片。后续合并无需重新运行仿真，应同时重建深度和回放索引。

先在 `.<episode>.pending-...` 中编码；成功后关闭 Parquet、视频和深度流，再原子重命名并发布完成标记。失败回合默认丢弃媒体。`--lerobot-keep-failed` 只供短测，产生的分片会明确标记 `diagnostic_only` 和 `success=false`，采集管理器不会把它计入成功数据。

队列默认 2 帧，满时反压等待，不丢帧。相机数组入队前复制，防止下一次渲染覆盖数据；图像和深度只在采样步读取。写入器错误或编码帧数不一致会使当前回合失败，不能仅凭任务成功标志计入数据。进程异常退出留下的 pending 目录不算完整分片，重试使用新的批次目录。

本地和远端改动不包含 `/tests/` 的同步或 Git 提交。验证记录放在 `outputs/direct_v3_deploy_20261112/`，远端短测输出在 `~/tmp/roboassemblybench/runs/`。
