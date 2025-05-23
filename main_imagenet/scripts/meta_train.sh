#!/bin/bash

DATA_MODEL_NUM=2
RUN_TYPE="meta_train"     
NAME=Maybe  # "Ame"

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

mkdir -p "save/${NAME}/${RUN_TYPE}"
mkdir -p "save/${NAME}/${RUN_TYPE}/metanetwork"
# save/Maybe/0/meta_train/metanetwork 

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
    data_model_num=$DATA_MODEL_NUM \
    name=$NAME \
    > "save/${NAME}/${RUN_TYPE}/train.log" &
