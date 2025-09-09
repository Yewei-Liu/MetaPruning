#!/bin/bash

MODEL="resnet50"            
INDEX=0
RUN_TYPE="finetune"                 
NAME=Final
EPOCHS=60
LR=0.01
LR_DECAY_MILESTOMES=\'30\'
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

mkdir -p "save/${NAME}/${INDEX}/${RUN_TYPE}/${SPEED_UP}"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup \
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    main_imagenet.py \
    +experiment=$CONFIG_NAME \
    model=$MODEL \
    run=$RUN_TYPE \
    index=$INDEX \
    name=$NAME \
    epochs=$EPOCHS \
    lr=$LR \
    speed_up=$SPEED_UP \
    resume_epoch=$RESUME_EPOCH \
    lr_decay_milestones=$LR_DECAY_MILESTOMES \
    > "save/${NAME}/${INDEX}/${RUN_TYPE}/${SPEED_UP}/finetune.log" &
