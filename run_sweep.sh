#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --array=1-32  # Number of parallel agents
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
python f0_vae_sweep.py --sweep_id vlsh39cb

# vlsh39cb = follow up grid sweep of ddb11dyz but with correct val dataloader
# udbrupn3 = follow up grid sweep of ddb11dyz
# ddb11dyz = sweep with resnet; kld annealing for kld > 1.0; cycle_epochs=n_epochs // 4, ramp_fraction=0.5
# 9bdl9deg = sweep with resnet-based vae; kld annealing (cycle_epochs=n_epochs / 10, ramp_fraction=0.5), 2000 epochs with no early stopping, lr reduce after 200 epochs
# xabocpvt = sweep with resnet-based vae, rebalanced train set to include 200 F2 and bad quality images removed; corrected recon loss + removed image norm from kld loss; best evaluate on best val loss + LR reduce on plateau
# vgkxorsp = sweep with resnet-based vae, rebalanced train set to include 200 F2 and bad quality images removed; corrected recon loss; best evaluate on best val loss + LR reduce on plateau
# q7pfwusn = same as r0bq9jm0; better best_recon_val_loss tracking; remove torch.clamp on kld?
# r0bq9jm0 = sweep with resnet-based vae, rebalanced train set to include 200 F2 and bad quality images removed; best evaluate on best val loss + LR reduce on plateau
# arx5uexy = sweep with resnet-based vae, rebalanced train set to include 200 F2 and bad quality images removed; best evaluate on best val loss + LR reduce on plateau
# hdpw034f = sweep with resnet-based vae, rebalanced train set to include 200 F2 and bad quality images removed; best evaluate on best val loss
# ibzu6scv = sweep with resnet-based vae, rebalanced train set to include 200 F2 and bad quality images removed
# 395cmn4b = sweep with resnet-based vae, rebalanced train set to include 200 F2
# 6o8li1mg = sweep with resnet-based vae, rebalanced train set to include 500 F2
# mmosfr4n = sweep with resnet-based vae, original trainset with only F0
# 0a39n1bn = sweep with standard convolutional vae, original trainset
