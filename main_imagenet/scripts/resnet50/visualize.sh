#!/bin/bash

MODEL="resnet50"  
INDEX=2
METANETWORK_INDEX=13
RUN_TYPE="visualize"                
NAME=Final 
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

mkdir -p "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/"

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
    resume_epoch=$RESUME_EPOCH \
    +metanetwork_index=$METANETWORK_INDEX \
    > "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/${INDEX}.log" &
