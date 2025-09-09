CUDA_VISIBLE_DEVICES=1 nohup python generate.py data_generator=resnet110_on_CIFAR10 index=0 num=3 > 0.txt &
CUDA_VISIBLE_DEVICES=3 nohup python generate.py data_generator=resnet110_on_CIFAR10 index=1 num=3 > 1.txt &
CUDA_VISIBLE_DEVICES=3 nohup python generate.py data_generator=resnet110_on_CIFAR10 index=2 num=3 > 2.txt &
CUDA_VISIBLE_DEVICES=4 nohup python generate.py data_generator=resnet110_on_CIFAR10 index=3 num=3 > 3.txt &
CUDA_VISIBLE_DEVICES=4 nohup python generate.py data_generator=resnet110_on_CIFAR10 index=4 num=3 > 4.txt &
