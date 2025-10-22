#!/bin/bash

#SBATCH -J metapruning
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o visualize.out
#SBATCH -e visualize.err


~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.__version__)"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.is_available())"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.device_count())"

MODEL="vit_b_16"  
INDEX=1 
METANETWORK_INDEX=22
RUN_TYPE="visualize"                
NAME=ViT
RESUME_EPOCH=-1
LR=0.0001
LR_MIN=0.000001
CLIP_GRAD_NORM=1.0
WEIGHT_DECAY=0.01
EPOCHS=300
BATCH_SIZE=256
OPT="adamw"     
LR_SCHEDULER="cosineannealinglr"  
LR_WARMUP_METHOD="linear"
LR_WARMUP_EPOCHS=30
LR_WARMUP_DECAY=0.01
LABEL_SMOOTHING=0.1
MIXUP_ALPHA=0.2
CUTMIX_ALPHA=0.1
FORCE_START_EPOCH=-1

NUM_GPUS=4
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
    lr_min=$LR_MIN \
    clip_grad_norm=$CLIP_GRAD_NORM \
    weight_decay=$WEIGHT_DECAY \
    lr_scheduler=$LR_SCHEDULER \
    opt=$OPT \
    lr_warmup_method=$LR_WARMUP_METHOD \
    lr_warmup_epochs=$LR_WARMUP_EPOCHS \
    lr_warmup_decay=$LR_WARMUP_DECAY \
    label_smoothing=$LABEL_SMOOTHING \
    mixup_alpha=$MIXUP_ALPHA \
    cutmix_alpha=$CUTMIX_ALPHA \
    force_start_epoch=$FORCE_START_EPOCH \
    epochs=$EPOCHS \
    +metanetwork_index=$METANETWORK_INDEX \
    > "save/${NAME}/${RUN_TYPE}/${INDEX}/metanetwork_${METANETWORK_INDEX}/${INDEX}.log" 2>&1
