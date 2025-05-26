# 📄 Experiments on Large Datasets (with parallel)

---

## 🔍 Overview

This directory includes codes for our method on large datasets with parallel (pruning ResNet50 on ImageNet). You can also easily add other experiments based on our framework.

---

## 🚀 Quick Reproduce 

You can download our pretrained metanetworks and models to quickly reproduce the experiments in our paper. (This is the abbreviated version of our experiments)

### Download pretrained models

download the `download.zip` from [Latest Release](https://github.com/Yewei-Liu/MetaPruning/releases/latest) and unzip it. 

Then create a `save/Final` directory and move `resnet50_on_ImageNet/meta_train/` from your download to it in a way like this.

```
main_imagenet/
├── save/
│    ├── Final/
│    │    └── meta_train/
│    └── ...
└── README.md                  # You are here!
```

### Get dataset

Download ImageNet dataset and set `data_path` in ['base.yaml'](configs/base.yaml) as the path to your ImageNet.

### Pruning

We have 3 data models. Among them, 0 and 1 are used for meta-training, and 2 is used for pruning.

We first feedforward data model 2 through our metanetwork, finetune and visualize the "Test Acc VS. Speed Up" curve by running
```bash
sh scripts/visualize.sh
```

Hyperparameters of `visualize.sh` is as follows
- `INDEX=2` : Use data model with index 2.
- `METANETWORK_INDEX=13` : Use metanetwork from epoch 13 in meta-training.
- `RUN_TYPE="visualize"` : Set running type to visualize.               
- `NAME=Final ` : A name for read data models and output.
- `RESUME_EPOCH=-1` : Resume from a previously saved checkpoint. For example, if finetuning completed epoch 5 before the process was interrupted, you can set this value to 5 to continue training at epoch 6. By default, this is set to -1, which means training will start from the beginning.

Then we continue to prune and finetune this network by running:
```bash
sh scripts/prune_after_metanetwork.sh
```

Hyperparameters of `prune_after_metanetwork.sh` is as follows
- `INDEX=2` : Use data model with index 2.
- `METANETWORK_INDEX=13` : Use metanetwork from epoch 13 in meta-training.
- `RUN_TYPE="prune_after_metanetwork"` : Set running type to prune after metanetwork.               
- `NAME=Final ` : A name for read data models and output.
- `SPEED_UP=2.3095` : The final target speed up, pruning to a speed up a little bit larger than this.
- `RESUME_EPOCH=-1` : Resume from a previously saved checkpoint. For example, if finetuning completed epoch 5 before the process was interrupted, you can set this value to 5 to continue training at epoch 6. By default, this is set to -1, which means training will start from the beginning.

---


## ✈️ Full reproduce

You can also do our experiments from scratch, generate data models, meta-train metanetworks and select the proper metanetwork for pruning. (This is the complete version of our experiments)

### Generate data models

We generate data models by finetuning the pytorch pretrained model by running :
```bash
sh scripts/train_from_scratch.sh
```
You can modify `NAME` to choose a unique name you like (this name must be the same in one experiment). `INDEX` means the index for your data models. For example, if you want to generate 3 data models, you should run `sh scripts/train_from_scratch.sh` 3 times with `INDEX` 0, 1, 2 respectively while keeping all other hyperparameters the same.

Then to do initial pruning and finetuning, you can change the `NAME` and `INDEX` in `prune.sh` and `finetune.sh` and run:
```bash
sh scripts/prune.sh
```
After finished, run :
```bash
sh scripts/finetune.sh 
```

### Meta-Training

After all data models generated, we can do meta-training by running:
```bash
sh scripts/meta_train.sh
```
Before running, we should modefied hyperparameters in `meta_train.sh`. We should change `NAME` to the same as `finetune.sh`, and set `DATA_MODEL_NUM` as our data model numbers. data model numbers should be less than your parallel gpu numbers (we suggest you to use 8 gpus just like us). And it should also be strictly less than your data model numbers, because we need the rest data models for visualization and test. If we set `DATA_MODEL_NUM` as 2, it will use data models of index 0 and 1 for meta-training.

### Select appropriate metanetwork for pruning

Like we mentioned in *Quick Reproduce*, we can visualize our metanetwork by run :
```bash
sh scripts/visualize.sh
```
Set `INDEX` to a index that haven't been used during meta-training (usually the largest index). And change `METANETWORK_INDEX` to search for the proper metanetwork in a binary search way.

### Pruning

We prune by run :
```bash
sh scripts/prune_after_metanetwork.sh
```
Before running, make sure to set the hyperparameters `INDEX`, `METANETWORK_INDEX`, `NAME`, and `SPEED_UP`. This time, both `INDEX` and `METANETWORK_INDEX` must be used during visualization. This is because pruning depends on the results from the visualization step — specifically, the feedforward through the metanetwork and subsequent finetuning. Therefore, if you want to prune a particular combination of `INDEX` and `METANETWORK_INDEX` that hasn't been visualized yet, you should visualize it first.

