# vla_isaac_demo

基于 **NVIDIA Isaac Sim 5.1** + **OpenVLA-7B** 的机器人视觉-语言-动作（VLA）仿真项目。支持单臂/双臂控制、实时碰撞检测、动作序列合法性验证。

## 目录结构

```
vla_isaac_demo/
├── README.md
├── requirements.txt              # Python 依赖（transformers, torch, PIL 等）
├── run.sh                        # VLA 主流程启动脚本
├── src/                          # 核心模块
│   ├── __init__.py
│   ├── scene.py                  # 场景构建（桌子、方块、相机、Franka 机械臂）
│   ├── controller.py             # 夹爪控制器（支持 LIBERO/Bridge 两种 gripper 约定）
│   ├── vla_client.py             # OpenVLA-7B 模型加载与推理客户端
│   └── checker.py                # 动作序列合法性检查器
├── demos/                        # 可运行的演示脚本
│   ├── main.py                   # VLA 闭环抓取主流程（需 Isaac Sim）
│   ├── pick_cube.py              # 单臂 RMPFlow 抓取 Demo（需 Isaac Sim）
│   ├── two_arm_collide.py        # 双臂碰撞检测 Demo（需 Isaac Sim）
│   └── test_checker.py           # Checker 单元测试（14 项，纯 Python）
└── image/                        # 保存的仿真相机截图
```

## 环境要求

- **Isaac Sim 5.1**: `/mnt/SSD_7T/panxubei/isaac-sim5.1/`
- **Python**: 3.10+（Isaac Sim 内置，`python.sh` 启动）
- **GPU**: NVIDIA RTX 系列，CUDA 12+

### 安装 Python 依赖

```bash
# 在 Isaac Sim 的 Python 环境中安装
/mnt/SSD_7T/panxubei/isaac-sim5.1/python.sh -m pip install -r requirements.txt
```

### HuggingFace 模型缓存

项目使用 OpenVLA-7B 模型，首次运行需下载。国内网络建议：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

模型缓存路径默认为 `~/.cache/huggingface/`，可在 `run.sh` 中通过 `HF_HOME` 自定义。

## 快速开始

### 1. VLA 自主抓取

```bash
# 方式一：使用启动脚本（推荐）
cd /home/panxubei/vla_isaac_demo
./run.sh

# 方式二：直接调用
/mnt/SSD_7T/panxubei/isaac-sim5.1/python.sh demos/main.py
```

流程：场景初始化 → 相机拍照 → 发送给 OpenVLA → 执行 7-DoF 动作 → 循环直至抓取成功。

### 2. 单臂 RMPFlow 抓取（无 VLA）

```bash
/mnt/SSD_7T/panxubei/isaac-sim5.1/python.sh demos/pick_cube.py
```

使用 Isaac Sim 内置 RMPFlow 运动规划直接抓取桌面上的红色方块。

### 3. 双臂碰撞检测

```bash
/mnt/SSD_7T/panxubei/isaac-sim5.1/python.sh demos/two_arm_collide.py
```

两个 Franka 机械臂分别从桌子两侧向同一个绿色方块运动。每帧读取所有 link 的世界坐标（`SingleXFormPrim`），计算两臂之间任意 link 对的最小欧氏距离，低于 12cm 阈值即打印碰撞对。

**碰撞检测原理**：
- 每臂 11 个 link（link0~link7 + hand + leftfinger + rightfinger）
- 每帧计算 11×11 = 121 对 link 中心距
- 取最小值，低于 0.12m 阈值判定碰撞
- 打印具体碰撞 link 对及距离

### 4. 动作序列检查器

```bash
cd /home/panxubei/vla_isaac_demo
python demos/test_checker.py
```

`Checker` 类验证 9 种操作（move/reach/grasp/place/open/close/handover/interact/push）在当前状态下是否合法，并支持多 agent 约束兼容性检查。

## 技术要点

### 夹爪约定

| 数据集 | 张开 | 闭合 |
|--------|------|------|
| LIBERO | `gripper ≈ 0` | `gripper ≈ 1` |
| Bridge | `gripper ≈ 1` | `gripper ≈ 0` |

项目使用 **LIBERO 微调**的 OpenVLA-7B，gripper 逻辑已适配（`controller.py`）。

### 碰撞检测方案选型

| 方案 | 状态 |
|------|------|
| `_dynamic_control.get_articulation_contacts()` | Isaac Sim 5.1 不存在此 API |
| `_dynamic_control.get_rigid_body_contacts()` | 对 articulation link 不可用 |
| `PhysxContactReportAPI`（USD stage 操作） | 同步脚本中触发场景损坏 |
| `isaacsim.sensors.physics.ContactSensor` | 同步脚本中不稳定 |
| **`SingleXFormPrim` + 中心距** | ✅ 当前方案，已验证可靠 |

## 许可证

内部研究项目。
