#!/bin/bash
# ============================================================
#  VLA + Isaac Sim 5.1 启动脚本
#  用法: ./run.sh
# ============================================================

# --- 路径配置 ---
ISAAC_SIM_DIR="/mnt/SSD_7T/panxubei/isaac-sim5.1"
DEMO_DIR="/home/panxubei/vla_isaac_demo"
MAIN_SCRIPT="${DEMO_DIR}/demos/main.py"

# --- HuggingFace 离线 + 本地缓存 ---
export HF_HOME="/mnt/SSD_7T/panxubei/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- 启动前自检 ---
if [ ! -f "${ISAAC_SIM_DIR}/python.sh" ]; then
    echo "[ERROR] 找不到 Isaac Sim python.sh: ${ISAAC_SIM_DIR}/python.sh"
    exit 1
fi

if [ ! -f "${MAIN_SCRIPT}" ]; then
    echo "[ERROR] 找不到主脚本: ${MAIN_SCRIPT}"
    exit 1
fi

if [ ! -d "${HF_HOME}" ]; then
    echo "[WARN] HF_HOME 不存在: ${HF_HOME}"
fi

# --- 打印当前环境（方便排错） ---
echo "=========================================="
echo " Isaac Sim   : ${ISAAC_SIM_DIR}"
echo " Demo Script : ${MAIN_SCRIPT}"
echo " HF_HOME     : ${HF_HOME}"
echo " OFFLINE     : HF=${HF_HUB_OFFLINE}  TRANSFORMERS=${TRANSFORMERS_OFFLINE}"
echo "=========================================="

# --- 启动 ---
"${ISAAC_SIM_DIR}/python.sh" "${MAIN_SCRIPT}" "$@"
