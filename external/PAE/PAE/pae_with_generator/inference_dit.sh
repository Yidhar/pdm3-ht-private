#!/bin/bash
CONFIG=${1:-"configs/DiT80ep_PAE_DINOv2L_d32.yaml"}
USE_EMA=${3:-"True"}

if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

if [ "$MASTER_PORT" == "" ] || [ "$MASTER_PORT" == "1236" ]; then
    MASTER_PORT=$(shuf -i 10000-20000 -n 1)
fi

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

export TORCH_DIST_TIMEOUT=36000

accelerate  launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision bf16 \
    inference_dit.py \
    --config "$CONFIG" \
    --use_ema "$USE_EMA" \
    # --batch_demo
    # --demo
    # --search_best_cfg
