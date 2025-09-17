#!/bin/bash
#SBATCH -J metapruning
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o tmp.out
#SBATCH -e tmp.err
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.__version__)"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.is_available())"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.device_count())"
nvidia-smi
MODEL="vit_b_16"  
INDEX=0
RUN_TYPE="train_from_scratch"                 
NAME="Final_ViT"
PRETRAINED=True
EPOCHS=100
LR=0.001
LR_DECAY_MILESTOMES=\'10000\'
NUM_GPUS=4                   
MASTER_PORT=18900             
CONFIG_NAME="base"   
OPT="adamw"           
BATCH_SIZE=256
BIG_BATCH_SIZE=64
WEIGHT_DECAY=0.3      
        
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
mkdir -p "save/${NAME}/${INDEX}/${RUN_TYPE}"
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    main_imagenet.py \
    +experiment=$CONFIG_NAME \
    opt=$OPT \
    model=$MODEL \
    run=$RUN_TYPE \
    index=$INDEX \
    name=$NAME \
    epochs=$EPOCHS \
    lr=$LR \
    lr_decay_milestones=$LR_DECAY_MILESTOMES \
    batch_size=$BATCH_SIZE \
    big_batch_size=$BIG_BATCH_SIZE \
    pretrained=$PRETRAINED \
    weight_decay=$WEIGHT_DECAY \
    > "save/${NAME}/${INDEX}/${RUN_TYPE}/${INDEX}.log" 2>&1
