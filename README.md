```bash
conda create -n meta-pruning python=3.9 -y
conda activate meta-pruning
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidian -y
conda install pyg==2.3.0 pytorch-scatter -c pyg -y
pip install hydra-core einops opencv-python 
pip install torch-pruning 
pip install datasets 
```

```bash
git clone ....
cd MetaPruning
pip install -e .
```
