<<<<<<< HEAD:main_imagenet/run.sh
python -m torch.distributed.launch --nproc_per_node=4 --master_port 18119 --use_env main_imagenet.py --model resnet50 \
 --epochs 90 --batch-size 64 --lr-step-size 30 --lr 0.01 --prune --method group_sl --pretrained --output-dir run/imagenet/resnet50_sl \
 --target-flops 2.00 --cache-dataset --print-freq 100 --workers 16 --data-path PATH_TO_IMAGENET --output-dir PATH_TO_OUTPUT_DIR # &> output.log
=======
#!/bin/bash

INDEX=5  
RUN_TYPE="train_from_scratch"                 
NAME=Maybe  # "Ame"
PRETRAINED=True
EPOCHS=30
LR=0.01
LR_DECAY_MILESTOMES=\'10\'

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

mkdir -p "save/${NAME}/${INDEX}/${RUN_TYPE}"

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
    index=$INDEX \
    name=$NAME \
    epochs=$EPOCHS \
    lr=$LR \
    lr_decay_milestones=$LR_DECAY_MILESTOMES \
    pretrained=$PRETRAINED \
    > "save/${NAME}/${INDEX}/${RUN_TYPE}/${INDEX}.log" &
>>>>>>> e4c40fd (final):main_imagenet/scripts/train_from_scratch.sh
