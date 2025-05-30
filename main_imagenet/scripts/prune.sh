#!/bin/bash

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
    no_distribution=True \
    run=$RUN_TYPE \
    speed_up=$SPEED_UP \
    index=$INDEX \
    name=$NAME \
    pretrained=$PRETRAINED \
    > "save/${NAME}/${INDEX}/${RUN_TYPE}/${SPEED_UP}.log" &
