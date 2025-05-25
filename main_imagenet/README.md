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


<!-- ## ✈️ Full reproduce

You can also do our experiments from scratch, generate data models, meta-train metanetworks and select the proper metanetwork for pruning. (This is the complete version of our experiments)

### Generate data models

Follow [generate_dataset/README.md](../generate_dataset/README.md) to generate data models for the following meta-training.

### Meta-Training

First we need to understand several configs of meta-training.

In ['configs/base.yaml'](configs/base.yaml) :
- `run` : running mode, for meta-training we set it to `meta_train`.
- `name` : name for save and output, choose a name you like.
- `task` : choose task to run, `resnet56_on_CIFAR10` or `VGG19_on_CIFAR100`.

We take `resnet56_on_CIFAR10` as example, in ['configs/task/resnet56_on_CIFAR10.yaml'](configs/task/resnet56_on_CIFAR10.yaml) :
- `meta_train` : set hyperparameters like epochs and lr for meta-training.
- `metanetwork` : set the size and res coefficient of metanetwork

To meta_train, run:
```bash
python main.py run=meta_train task=resnet56_on_CIFAR10 name=Test 
```

### Select appropriate metanetwork for pruning

To control the finetuning stages — both after metanetwork and after pruning — we use the `pruning` configuration in ['configs/task/resnet56_on_CIFAR10'](configs/task/resnet56_on_CIFAR10.yaml) . Within this configuration, the `finetuning` includes two key parameters: `after_pruning` and `after_metanetwork`. These parameters are used to set the hyperparameters for finetuning after metanetwork and finetuning after pruning.

We search for the most suitable metanetwork by visualizing its performance using a binary search strategy. Every time we visualize a metanetwork, we pass a data model through it, followed by finetuning with hyperparameters same as `pruning.finetune.after_metanetwork`, and then visualize the ``Test Accuracy vs.\ Speed-Up'' curve of the resulting model.
(A little trick is that if finetuning during visualize a metanetwork costs too much time, we can temporarily change the `epochs` and `lr_deacy_milestones` in configs to be smaller, and change them back while pruning, this can save lots of time in `VGG_on_CIFAR100`)

We take `resnet56_on_CIFAR10` as example. First, we want to visualize metanetwork at epoch 50, we should run :
```bash
python main.py task=resnet56_on_CIFAR10 name=Test run=visualize index=50
```
Because we are doing it in a binary search way, so based on the performance we may later run:
```bash
python main.py task=resnet56_on_CIFAR10 name=Test run=visualize index=25
# or
python main.py task=resnet56_on_CIFAR10 name=Test run=visualize index=75
```
We can also visualize many metanetworks at a time:
```bash
python main.py task=resnet56_on_CIFAR10 name=Test run=visualize index=[30,40,50]
```

### Pruning

To do final pruning, first we need to choose a unique `reproduce_index`. Here we use **3** as example.

First, we need to train a model for pruning. We run :
```bash
python main.py task=resnet56_on_CIFAR10 name=Test run=pretrain_final index=3
```

When finished, we have a directory like
```
main/
├── final/           
│   ├── resnet56_on_CIFAR10
│   │   ├── reproduce_3
│   │   │   └── model.pth
│   │   └── ... 
│   └── ...  
└── README.md               # You are here!
```

Assuming we use metanetwork at epoch 28 for final pruning, we should copy `epoch_18.pth` from `save/metanetwork/resnet56_on_CIFAR10/Test/level_0/epoch_18.pth` to `final/resnet56_on_CIFAR10/reproduce_3` and rename it as `metanetwork.pth`. So the final directory should look like :

```
main/
├── final/           
│   ├── resnet56_on_CIFAR10
│   │   ├── reproduce_3
│   │   │   ├── metanetwork.pth
│   │   │   └── model.pth
│   │   └── ... 
│   └── ...  
└── README.md               # You are here!
```

Finally, if we want to pruning with speed up 2.5x, we can run :
```bash
python main.py task=resnet56_on_CIFAR10 run=pruning_final name=Test reproduce_index=3 index=2.5
```
 -->



