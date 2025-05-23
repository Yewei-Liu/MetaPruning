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

## 🚀 Getting Started

- git clone or download our code
- create the conda env
```bash
conda create -n MetaPruning python=3.9 -y
conda activate MetaPruning
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pyg==2.3.0 pytorch-scatter -c pyg -y
pip install hydra-core einops opencv-python 
pip install torch-pruning 
pip install datasets 
pip install importlib_metadata
pip install termcolor
pip install h5py
```

- install our package
```bash
cd MetaPruning
pip install -e .
```

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


## 📊 Results

### ResNet56 on CIFAR10

| Method | Base  | Pruned | $\Delta$ Acc | Pruned FLOPs | Speed Up |
|-----------------------------------|-------|----------------------------|--------------|-----------------------------|---------------------------|
| NISP                 | ---   | ---                        | ---          | 43.2\%                      | 1.76                      |
| Geometric        | 93.59 | 93.26                      | -0.33        | 41.2\%                      | 1.70                      |
| Polar                | 93.80 | 93.83                      | 0.03         | 46.8\%                      | 1.88                      |
| DCP-Adapt       | 93.80 | 93.81                      | 0.01         | 47.0\%                      | 1.89                      |
| CP                      | 92.80 | 91.80                      | -1.00        | 50.0\%                      | 2.00                      |
| AMC                  | 92.80 | 91.90                      | -0.90        | 50.0\%                      | 2.00                      |
| HRank               | 93.26 | 92.17                      | -1.09        | 50.0\%                      | 2.00                      |
| SFP                   | 93.59 | 93.36                      | -0.23        | 52.6\%                      | 2.11                      |
| ResRep             | 93.71 | 93.71                      | 0.00         | 52.8\%                      | 2.12                      |
| SCP                  | 93.69 | 93.23                      | -0.46        | 51.5\%                      | 2.06                      |
| FPGM               | 93.59 | 92.93                      | -0.66        | 52.6\%                      | 2.11                      |
| FPC| 93.59 | 93.24                      | -0.35        | 52.9\%                      | 2.12                      |
| DMC                    | 93.62 | 92.69                      | -0.93        | 50.0\%                      | 2.00                      |
| GNN-RL              | 93.49 | 93.59                      | 0.10         | 54.0\%                      | 2.17                      |
| DepGraph w/o SL   | 93.53 | 93.46                      | -0.07        | 52.6\%                      | 2.11                      |
| DepGraph with SL  | 93.53 | 93.77                      | 0.24         | 52.6\%                      | 2.11                      |
| ATO                  | 93.50 | 93.74                      | 0.24         | 55.0\%                      | 2.22                      |
| Meta-Pruning (ours)               | 93.51 | <u>**94.08** | 0.57         | **56.5\%**             | **2.30**             |
| Meta-Pruning (ours)               | 93.51 | 93.56                      | 0.05         | **56.5\%**             | **2.30**             |
| Meta-Pruning (ours)               | 93.51 | **93.81**             | 0.30         | <u>**56.9\%** | <u>**2.32** |
|-|-|-|-|-|-|
| GBN                   | 93.10 | 92.77                      | -0.33        | 60.2\%                      | 2.51                      |
| AFP                    | 93.93 | 92.94                      | -0.99        | 60.9\%                      | 2.56                      |
| C-SGD               | 93.39 | 93.44                      | 0.05         | 60.8\%                      | 2.55                      |
| Greg-1                | 93.36 | 93.18                      | -0.18        | 60.8\%                      | 2.55                      |
| Greg-2                | 93.36 | 93.36                      | 0.00         | 60.8\%                      | 2.55                      |
| DepGraph w/o SL   | 93.53 | 93.36                      | -0.17        | 60.2\%                      | 2.51                      |
| DepGraph with SL  | 93.53 | **93.64**            | 0.11         | 61.1\%                      | 2.57                      |
| ATO                   | 93.50 | 93.48                      | -0.02        | 65.3\%                      | 2.88                      |
| Meta-pruning (ours)               | 93.51 | <u>**93.69** | 0.18         | 65.6\%                      | 2.91                      |
| Meta-pruning (ours)               | 93.51 | 93.42                      | -0.09        | **66.0\%**             | **2.94**             |
| Meta-pruning (ours)               | 93.51 | 93.31                      | -0.20        | <u>**66.6\%** | <u>**2.99** |
| Meta-pruning (ours)               | 93.51 | <u>**93.25** | -0.26        | **66.9\%**             | **3.02**             |
| Meta-pruning (ours)               | 93.51 | **93.20**             | -0.31        | 66.7\%                      | 3.00                      |
| Meta-pruning (ours)               | 93.51 | 93.18                      | -0.33        | <u>**67.5\%** | <u>**3.08** |

---

### VGG19 on CIFAR100

| Method                            | Base  | Pruned                     | $\Delta$ Acc | Pruned FLOPs                  | Speed Up                  |
|-----------------------------------|-------|----------------------------|--------------|-------------------------------|---------------------------|
| OBD                    | 73.34 | 60.70                      | -12.64       | 82.55\%                       | 5.73                      |
| OBD                   | 73.34 | 60.66                      | -12.68       | 83.58\%                       | 6.09                      |
| EigenD                 | 73.34 | 65.18                      | -8.16        | 88.64\%                       | 8.80                      |
| Greg-1               | 74.02 | 67.55                      | -6.67        | 88.69\%                       | 8.84                      |
| Greg-2              | 74.02 | 67.75                      | -6.27        | 88.69\%                       | 8.84                      |
| DepGraph w/o SL   | 73.50 | 67.60                      | -5.44        | 88.73\%                       | 8.87                      |
| DepGraph with SL  | 73.50 | <u>**70.39** | -3.11        | 88.79\%                       | 8.92                      |
| Meta-Pruning (ours)               | 73.65 | **70.06**             | -3.59        | 88.85\%                       | 8.97                      |
| Meta-Pruning (ours)               | 73.65 | 69.19                      | -4.46        | **88.91\%**              | **9.02**             |
| Meta-Pruning (ours)               | 73.65 | 67.05                      | -6.60        | <u>**89.07\%** | <u>**9.15** |

---

### ResNet50 on ImageNet

| Method                    | Base Top-1(Top-5) | Pruned Top-1($\Delta$)                | Pruned Top-5($\Delta$)                | Pruned FLOPs                |
|---------------------------|-------------------|---------------------------------------|---------------------------------------|-----------------------------|
| DCP      | 76.01\%(92.93\%)  | 74.95\%(-1.06\%)                      | 92.32\%(-0.61\%)                      | 55.6\%                      |
| CCP            | 76.15\%(92.87\%)  | 75.21\%(-0.94\%)                      | 92.42\%(-0.45\%)                      | 54.1\%                      |
| FPGM        | 76.15\%(92.87\%)  | 74.83\%(-1.32\%)                      | 92.32\%(-0.55\%)                      | 53.5\%                      |
| ABCP         | 76.01\%(92.96\%)  | 73.86\%(-2.15\%)                      | 91.69\%(-1.27\%)                      | 54.3\%                      |
| DMC         | 76.15\%(92.87\%)  | 75.35\%(-0.80\%)                      | 92.49\%(-0.38\%)                      | 55.0\%                      |
| Random      | 76.15\%(92.87\%)  | 75.13\%(-1.02\%)                      | 92.52\%(-0.35\%)                      | 51.0\%                      |
| DepGraph  | 76.15\%(-)        | 75.83\%(-0.32\%)                      | -                                     | 51.7\%                      |
| ATO           | 76.13\%(92.86\%)  | <u>**76.59\%**(+0.46\%) | <u>**93.24\%**(+0.38\%) | 55.2\%                      |
| DTP    | 76.13\%(-)        | 75.55\%(-0.58\%)                      | -                                     | **56.7\%**             |
| ours                      | 76.14\%(93.11\%)  | 76.17\%(+0.03\%)                      | 92.94\%(-0.17\%)                      | <u>**56.9\%** |
| ours                      | 76.14\%(93.11\%)  | **76.27**\%(+0.13\%)             | **93.05**\%(-0.06\%)             | <u>**56.9\%** |
| ours                      | 76.14\%(93.11\%)  | 76.15\%(+0.01\%)                      | 92.95\%(-0.16\%)                      | <u>**56.9\%** |

---

<!-- ## 📄 Citation

Include a BibTeX entry for your paper.

If you find this code useful, please cite our paper:
```bibtex
@article{yourpaper2025,
  author = {Your Name and Co-Author},
  title = {Title of Your Paper},
  journal = {Conference or Journal Name},
  year = {2025}
}
``` -->


