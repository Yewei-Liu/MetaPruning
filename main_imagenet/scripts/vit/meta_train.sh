#!/bin/bash

#SBATCH -J metapruning
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o meta_train.out
#SBATCH -e meta_train.err


~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.__version__)"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.is_available())"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.device_count())"

MODEL="vit_b_16"  
DATA_MODEL_NUM=2
RUN_TYPE="meta_train"     
NAME="Final_ViT"
RESUME_EPOCH=1
# metanetwork
TARGET
NUM_LAYER=2
HIDDIM=4
IN_NODE_DIM=6
NODE_RES_RATIO=0.001
EDGE_RES_RATIO=0.001


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

mkdir -p "save/${NAME}/${RUN_TYPE}"
mkdir -p "save/${NAME}/${RUN_TYPE}/metanetwork"
# save/Maybe/0/meta_train/metanetwork 

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    main_imagenet.py \
    +experiment=$CONFIG_NAME \
    metanetwork.num_layer=$NUM_LAYER \
    metanetwork.hiddim=$HIDDIM \
    metanetwork.in_node_dim=$IN_NODE_DIM \
    metanetwork.node_res_ratio=$NODE_RES_RATIO \
    metanetwork.edge_res_ratio=$EDGE_RES_RATIO \
    model=$MODEL \
    run=$RUN_TYPE \
    data_model_num=$DATA_MODEL_NUM \
    resume_epoch=$RESUME_EPOCH \
    name=$NAME \
    > "save/${NAME}/${RUN_TYPE}/train.log" 2>&1
