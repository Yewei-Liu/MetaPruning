#!/bin/bash

#SBATCH -J metapruning
#SBATCH -p IAI_SLURM_3090
#SBATCH --qos=8gpu
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o visualize.out
#SBATCH -e visualize.err


~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.__version__)"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.is_available())"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.device_count())"

MODEL="vit_b_16"  
INDEX=1 
METANETWORK_INDEX=2
RUN_TYPE="visualize"                
NAME="Final_ViT_1"
RESUME_EPOCH=0
LR=0.01
WEIGHT_DECAY=0.05
EPOCHS=200
BATCH_SIZE=128 
OPT="sgd"     
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

mkdir -p "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    main_imagenet.py \
    +experiment=$CONFIG_NAME \
    batch_size=$BATCH_SIZE \
    model=$MODEL \
    run=$RUN_TYPE \
    index=$INDEX \
    name=$NAME \
    resume_epoch=$RESUME_EPOCH \
    lr=$LR \
    weight_decay=$WEIGHT_DECAY \
    lr_decay_milestones=$LR_DECAY_MILESTONES \
    opt=$OPT \
    epochs=$EPOCHS \
    +metanetwork_index=$METANETWORK_INDEX \
    > "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/${INDEX}.log" 2>&1
