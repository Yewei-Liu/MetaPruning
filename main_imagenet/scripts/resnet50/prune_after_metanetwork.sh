#!/bin/bash

#SBATCH -J metapruning
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o tmp.out
#SBATCH -e tmp.err

MODEL="resnet50"  
INDEX=2
METANETWORK_INDEX=15
RUN_TYPE="prune_after_metanetwork"                
NAME=Large 
SPEED_UP=2.3095
RESUME_EPOCH=-1
LR=0.01
EPOCHS=200
LR_DECAY_MILESTONES=\'120,160,185\'

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

mkdir -p "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/${SPEED_UP}/"


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
    speed_up=$SPEED_UP \
    resume_epoch=$RESUME_EPOCH \
    lr=$LR \
    lr_decay_milestones=$LR_DECAY_MILESTONES \
    epochs=$EPOCHS \
    +metanetwork_index=$METANETWORK_INDEX \
    > "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/${SPEED_UP}/prune.log" 2>&1
