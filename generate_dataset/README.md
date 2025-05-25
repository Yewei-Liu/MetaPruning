## 🔍 Overview

This directory includes codes for generating data models.

---

## 🚀 Generate

For efficiency, we generate datasets on different gpus and merge them together as a big dataset.

### Config

In ['configs/base.yaml'](configs/base.yaml) we need focus on these few arguments:
- level: Simply set it to 0 is OK for all our experiments. This is a legacy argument. We tried to prune with multi-level metanetworks but turned out finding that only one metanetwork is enough and performs great. You can explore it by yourself if you want.
- Index: index for your gpus (or processes if you put more than one processes on a gpu). This is used for merge datasets.
- Num: How many data models to generate.

### Generate on multiple gpus

We take generating 10 data models for resnet56 on CIFAR10 with 3 gpus as example. To do this, we run :
```bash
CUDA_VISIBLE_DEVICES=0 nohup python generate.py data_generator=resnet56_on_CIFAR10 index=0 num=4 > 0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python generate.py data_generator=resnet56_on_CIFAR10 index=1 num=3 > 1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python generate.py data_generator=resnet56_on_CIFAR10 index=2 num=3 > 2.txt &
```
Then wait until it finish.

### Merge multiple small datasets into one dataset

Run :
```bash
python merge.py data_generator=resnet56_on_CIFAR10
```
Then type in the number of your small datasets (3 here)

---

## Clean

Each time before you generate new data models, use `sh clean.sh` to clean the cache and useless files. **must do this or it will directly use .cache instead of generating new datasets**