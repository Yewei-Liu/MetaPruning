#!/bin/bash

DATA_PATH=../../../../data/imagenet/
MODEL="vit_b_16"  
INDEX=1 
METANETWORK_INDEX=8
RUN_TYPE="visualize"                
NAME=ViT_visualize_4
RESUME_EPOCH=-1
LR=0.0001
WEIGHT_DECAY=0.01
EPOCHS=100
BATCH_SIZE=128
OPT="adamw"     
LR_SCHEDULER="cosineannealinglr"  
LR_WARMUP_METHOD="linear"
LR_WARMUP_EPOCHS=10
LR_WARMUP_DECAY=0.1
LABEL_SMOOTHING=0.1
MIXUP_ALPHA=0.2
CUTMIX_ALPHA=0.1

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
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=INFO

mkdir -p "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/"

nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    main_imagenet.py \
    data_path=$DATA_PATH \
    +experiment=$CONFIG_NAME \
    batch_size=$BATCH_SIZE \
    model=$MODEL \
    run=$RUN_TYPE \
    index=$INDEX \
    name=$NAME \
    resume_epoch=$RESUME_EPOCH \
    lr=$LR \
    weight_decay=$WEIGHT_DECAY \
    lr_scheduler=$LR_SCHEDULER \
    opt=$OPT \
    lr_warmup_method=$LR_WARMUP_METHOD \
    lr_warmup_epochs=$LR_WARMUP_EPOCHS \
    lr_warmup_decay=$LR_WARMUP_DECAY \
    label_smoothing=$LABEL_SMOOTHING \
    mixup_alpha=$MIXUP_ALPHA \
    cutmix_alpha=$CUTMIX_ALPHA \
    epochs=$EPOCHS \
    +metanetwork_index=$METANETWORK_INDEX \
    > "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/${INDEX}.log" 2>&1 &
