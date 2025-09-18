#!/bin/bash

#SBATCH -J metapruning
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH -o tmp.out
#SBATCH -e tmp.err


~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.__version__)"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.is_available())"
~/.conda/envs/MetaPruning/bin/python -c "import torch; print(torch.cuda.device_count())"



MODEL="vit_b_16"
INDEX=2
RUN_TYPE="prune"               
SPEED_UP=1.03
NAME="Final_ViT"
PRETRAINED=True
CONFIG_NAME="base"              
        
export HYDRA_FULL_ERROR=1

mkdir -p "save/${NAME}/${INDEX}/${RUN_TYPE}"

python main_imagenet.py \
    +experiment=$CONFIG_NAME \
    model=$MODEL \
    no_distribution=True \
    run=$RUN_TYPE \
    speed_up=$SPEED_UP \
    index=$INDEX \
    name=$NAME \
    pretrained=$PRETRAINED \
    > "save/${NAME}/${INDEX}/${RUN_TYPE}/${SPEED_UP}.log" 2>&1
