```bash
conda create -n meta-pruning python=3.9
conda activate meta-pruning
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg==2.3.0 pytorch-scatter -c pyg
pip install hydra-core einops opencv-python
pip install torch-pruning
pip install tensorboard
```

```bash
git clone ??????????
cd meta-pruning
pip install -e .
```

if want to generate datasets
```bash
pip install datasets
```