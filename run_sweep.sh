#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --array=1-100  # 50 parallel agents
#SBATCH -o /nfs/research/birney/users/esther/medaka-ziram/out/%x-%j.out

module purge
module load cuda/12.2.0

# Initialize Micromamba
MICROMAMBA_PATH=$(which micromamba)
MICROMAMBA_ENV=/hps/software/users/birney/esther/micromamba/envs/indigene-img-umap

# Ensure Micromamba is executable
chmod +x $MICROMAMBA_PATH

# Initialize Micromamba shell
eval "$($MICROMAMBA_PATH shell hook --shell=bash)"

# Activate the environment
micromamba activate $MICROMAMBA_ENV


### Run command (> wandb sweep sweep.yaml --project "Ziram VAE Training") on CLI first
python f0_vae_sweep.py --sweep_id 395cmn4b

# 395cmn4b = sweep with resnet-based vae, rebalanced train set to include 200 F2
# 6o8li1mg = sweep with resnet-based vae, rebalanced train set to include 500 F2
# mmosfr4n = sweep with resnet-based vae, original trainset with only F0
# 0a39n1bn = sweep with standard convolutional vae, original trainset