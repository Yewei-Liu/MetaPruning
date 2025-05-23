#!/bin/bash

INDEX=2
METANETWORK_INDEX=14
RUN_TYPE="prune_after_metanetwork"                
NAME=Maybe  # "Ame"
SPEED_UP=2.5001
RESUME_EPOCH=-1

NUM_GPUS=8                     
MASTER_PORT=18900             
CONFIG_NAME="base"              
        

# Find available port
while true; do
    if ! nc -z 127.0.0.1 $MASTER_PORT; then
        break
    fi
    ((MASTER_PORT++))
done

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4

mkdir -p "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/${SPEED_UP}/"

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
    index=$INDEX \
    name=$NAME \
    speed_up=$SPEED_UP \
    resume_epoch=$RESUME_EPOCH \
    +metanetwork_index=$METANETWORK_INDEX \
    > "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/${SPEED_UP}/prune.log" &
