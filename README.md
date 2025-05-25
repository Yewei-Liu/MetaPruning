# 📄 Meta Pruning via Graph Metanetworks : A Meta Learning Framework for Network Pruning

<!-- **[Insert title here]**  
*Authors: [Your Name], [Co-Author(s)]*  
*Conference/Journal: [e.g., NeurIPS 2025, arXiv preprint]*  
*ArXiv Link: [insert link]*  
*Published Version: [insert DOI or link if available]* -->

## 🔍 Overview

This repository contains the codes for our paper. We implemented the 3 experimenets in our paper (pruning ResNet56 on CIFAR10, VGG10 on CIFAR100, ResNet50 on ImageNet). You can also easily add other experiments based on our framework.

We have referenced the [Depgraph](https://github.com/VainF/Torch-Pruning) codebase in part for our implementation. Thanks 😊.

---

## 🚀 Reproduce

### Generate Data Models

Use ['generate_dataset'](generate_dataset/) to generate data models for meta-training.



---


## 📁 Directory Structure

```
.
├── data_loaders/           # Get data loaders for datasets and data models
├── generate_dataset/       # Generate data models
├── main/                   # Experiment on CIFAR10 & CIFAR100 (no parallel)
├── main_imagenet/          # Experiment on ImageNet (with parallel)
├── nn/                     # Graph Neural Networks
├── utils/                  # Utility Functions
│   ├── imagenet_utils/     # Utility Functions for ImageNet from DepGraph
│   ├── convert.py/         # Conversions between networks and graphs
│   ├── meta_train.py       # Meta train (eval) our metanetwork
│   ├── pruner.py           # Get a pruner
│   ├── pruning.py          # Several different ways of pruning
│   ├── train.py            # Train (eval) our network
│   ├── visualize.py        # Visualize "Test Acc VS. Speed Up" curve
│   └── ...          
└── README.md               # You are here!
```

---


## 🧪 Usage

See ['main/README.md'](main/README.md) for our implementation on small datasets without parallel (ResNet56 on CIFAR10, VGG19 on CIFAR100)

See ['main_imagenet/README.md'](main_imagenet/README.md) for our implementation on big datasets with data parallel (ResNet50 on ImageNet)


---




