# 📄 Experiments on Small Datasets (no parallel)

---

## 🔍 Overview

This directory includes codes for our method on small datasets without parallel (pruning ResNet56 on CIFAR10, VGG19 on CIFAR100). You can also easily add other experiments based on our framework.

---

## 🚀 Quick Reproduce

You can download our pretrained metanetworks and models to quickly reproduce the experiments in our paper. (This is the abbreviated version of our experiments)

### Download pretrained models

download the `download.zip` from [Latest Release](https://github.com/Yewei-Liu/MetaPruning/releases/latest) and unzip it. 

Then create a `final` directory and move `resnet56_on_CIFAR10` and `VGG19_on_CIFAR100` from your download to it in a way like this.

```
main/
├── final/      
│    ├── resnet56_on_CIFAR10
│    └── VGG19_on_CIFAR100
└── README.md                  # You are here!
```

Each file like `resnet56_on_CIFAR10/reproduce_0` contains a metanetwork for pruning and a model to be pruned. You can change `reproduce_0` to `reproduce_{k}` and setting `reproduce_index` to the corresponding `k` during pruning.

### Pruning

See ['scripts'](scripts/resnet56_on_CIFAR10.sh) to choose your scripts for pruning.

In each script : 
```bash
python main.py task=resnet56_on_CIFAR10 name=Final run=pruning_final reproduce_index=0 seed=7 index=2.3
```
You can Change:
- `task` : Specify which task to run. `resnet56_on_CIFAR10` or `VGG19_on_CIFAR100`.
- `name` : Set the name of output file.
- `reproduce_index` : Corresponding to your `reproduce_{k}` file name like mentioned above.
- `seed` : Set seed for reproduce.
- `index` : Pruning to a speed up larger than this value.

---


## ✈️ Full reproduce

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




