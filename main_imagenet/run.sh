#!/bin/bash

NUM_GPUS=8                     
MASTER_PORT=29500               
CONFIG_NAME="base"              
RUN_TYPE="train"               
TARGET_FLOPS=4.0               

# Find available port
while true; do
    if ! nc -z 127.0.0.1 $MASTER_PORT; then
        break
    fi
    ((MASTER_PORT++))
done

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup \
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    main_imagenet.py \
    +experiment=$CONFIG_NAME \
    run=$RUN_TYPE \
    target_flops=$TARGET_FLOPS \
    > 1.txt &
