## 🔍 Overview

This directory includes codes for generating data models.

---

## 🚀 Generate

For efficiency, we generate datasets on different gpus and merge them together as a big dataset.

### Config

In ['configs/base.yaml'](configs/base.yaml) we need focus on these few arguments:
- Index: index for your gpus (or processes if you put more than one processes on a gpu). This is used for merge datasets.
- Num: How many data models to generate.
- Method: Which pruning criterion to use. You can choose from `group_l2_norm_no_normalizer` `group_l2_norm_mean_normalizer` `group_l2_norm_max_normalizer` or add your own pruning criterion at ['utils/pruner.py'](../utils/pruner.py)

### Generate on multiple gpus

We take generating 10 data models for resnet56 on CIFAR10 with 3 gpus as example. To do this, we run :
```bash
CUDA_VISIBLE_DEVICES=0 nohup python generate.py data_generator=resnet56_on_CIFAR10 method=group_l2_norm_max_normalizer index=0 num=4 > 0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python generate.py data_generator=resnet56_on_CIFAR10 method=group_l2_norm_max_normalizer index=1 num=3 > 1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python generate.py data_generator=resnet56_on_CIFAR10 method=group_l2_norm_max_normalizer index=2 num=3 > 2.txt &
```
Then wait until it finish.

### Merge multiple small datasets into one dataset

Run :
```bash
python merge_dataset.py data_generator=resnet56_on_CIFAR10 method=group_l2_norm_max_normalizer
```
Then type in the number of your small datasets (3 here)

---

## Clean

Use `sh clean.sh` to clean the cache and useless files. 