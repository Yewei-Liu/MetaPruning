#!/bin/bash

#SBATCH -J liuyewei
#SBATCH -p IAI_SLURM_HGX
#SBATCH --qos=16gpu-hgx
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -o linkpred.out
#SBATCH -e linkpred.err

nvidia-smi