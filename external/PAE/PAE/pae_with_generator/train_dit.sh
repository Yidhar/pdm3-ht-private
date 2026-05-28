#!/bin/bash
# ============================================================
# LightningDiT Training Script (Multi-Node)
# ============================================================
# Usage:
#   # Single node, auto-detect GPUs
#   bash train_dit.sh
#
#   # Single node, custom config
#   bash train_dit.sh configs/custom.yaml
#
#   # Multi-node training
#   NNODES=2 RANK=0 MASTER_ADDR=192.168.1.1 bash train_dit.sh
# ============================================================

CONFIG=${1:-"configs/DiT80ep_PAE_DINOv2L_d32.yaml"}

# ---- Auto-detect GPUs if not set ----
if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

# ---- Avoid port conflicts ----
if [ "$MASTER_PORT" == "" ] || [ "$MASTER_PORT" == "1236" ]; then
    MASTER_PORT=$(shuf -i 10000-20000 -n 1)
fi

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

export TORCH_DIST_TIMEOUT=36000

echo "==========================================="
echo "  LightningDiT Training Pipeline"
echo "==========================================="
echo "CONFIG:       $CONFIG"
echo "NNODES:       $NNODES"
echo "NODE_RANK:    $NODE_RANK"
echo "GPUS/NODE:    $GPUS_PER_NODE"
echo "MASTER_ADDR:  $MASTER_ADDR:$MASTER_PORT"
echo "==========================================="

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision bf16 \
    train_dit.py \
    --config "$CONFIG"
