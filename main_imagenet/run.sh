python -m torch.distributed.launch --nproc_per_node=4 --master_port 18119 --use_env main_imagenet.py --model resnet50 \
 --epochs 90 --batch-size 64 --lr-step-size 30 --lr 0.01 --prune --method group_sl --pretrained --output-dir run/imagenet/resnet50_sl \
 --target-flops 2.00 --cache-dataset --print-freq 100 --workers 16 --data-path PATH_TO_IMAGENET --output-dir PATH_TO_OUTPUT_DIR # &> output.log