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
INDEX=0
RUN_TYPE="prune"               
SPEED_UP=1.2920
NAME=Final 
PRETRAINED=True
DEVICE=7                   
CONFIG_NAME="base"              
        
export HYDRA_FULL_ERROR=1

mkdir -p "save/${NAME}/${INDEX}/${RUN_TYPE}"

CUDA_VISIBLE_DEVICES=${DEVICE} nohup \
python main_imagenet.py \
    +experiment=$CONFIG_NAME \
    model=$MODEL \
    no_distribution=True \
    run=$RUN_TYPE \
    speed_up=$SPEED_UP \
    index=$INDEX \
    name=$NAME \
    pretrained=$PRETRAINED \
    > "save/${NAME}/${INDEX}/${RUN_TYPE}/${SPEED_UP}.log" &
