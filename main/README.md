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

First we need to under several configs of meta-training.

In ['configs/base.yaml'](configs/base.yaml) :
- `run` : running mode, for meta-training we set it to `meta_train`.
- `name` : name for save and output, choose a name you like.
- `task` : choose task to run, `resnet56_on_CIFAR10` or `VGG19_on_CIFAR100`.

We take resnet56_on_CIFAR10 as example, in ['configs/task/resnet56_on_CIFAR10.yaml'](configs/task/resnet56_on_CIFAR10.yaml) :
- `meta_train` : set hyperparameters like epochs and lr for meta-training.
- `metanetwork` : set the size and res coefficient of metanetwork

To meta_train, run:
```bash
python main.py run=meta_train task=resnet56_on_CIFAR10 name=Test 
```

