import os, sys, pickle
import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import Dataset
import seaborn as sns
import albumentations as A
import datetime
import yaml
sns.set()

sys.path.append(os.path.abspath("."))
from src_vae.train import train_vae
from src_vae.model import VAEModel
from src_vae.data import ZiramF0Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading data...")

# X_train = ZiramF0Dataset(img_type="brightfield", train=True)
# X_test = ZiramF0Dataset(img_type="brightfield", train=False)

X_train = ZiramF0Dataset(img_type="full_dataset", train=True, val=False, test=False, apply_transform=True)
X_val = ZiramF0Dataset(img_type="full_dataset", train=False, val=True, test=False, apply_transform=False)

print("Done loading data!")


# Get datetime for better model tracking
export_date = datetime.datetime.now().strftime("%Y%m%d-%H%M")


# Train VAE
export_dir = f"./results/{export_date}-vae_resnet_testkld/"
print(export_dir)

model, history = train_vae(
    X_train=X_train,
    X_test=X_val,
    latent_dim=128,
    batch_size=64,
    lr=1e-3,
    kld_b=1,
    n_epochs=2000,

    # capacity=32,
    # depth=5,
    # dropout_p=0.3,

    device=device,
    export_dir=export_dir,
    wandb_flag=True
)

print("Started time: ", export_date)
print("Ended: ", datetime.datetime.now().strftime("%Y%m%d-%H%M"))