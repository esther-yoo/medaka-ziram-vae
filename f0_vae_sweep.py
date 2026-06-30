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
import wandb
import argparse
sns.set()

sys.path.append(os.path.abspath("."))
from src_vae.train import train_vae
from src_vae.model import VAEModel
from src_vae.data import ZiramF0Dataset

def sweep_train():
    # Get datetime for better model tracking
    export_date = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    ### Load data
    print("Loading data...")
    X_train = ZiramF0Dataset(img_type="full_dataset", train=True, val=False, test=False, apply_transform=True)
    X_val = ZiramF0Dataset(img_type="full_dataset", train=False, val=True, test=False, apply_transform=True)
    print("Done loading data!")

    with wandb.init() as run:
        config = run.config
        model, history = train_vae(
            X_train=X_train,
            X_test=X_val,
            latent_dim=config.latent_dim,
            capacity=config.capacity,
            depth=config.depth,
            batch_size=config.batch_size,
            lr=config.lr,
            dropout_p=config.dropout_p,
            kld_b=config.kld_b,
            n_epochs=2000,
            device=device,
            export_dir=f"./results/sweep_{run.sweep_id}/{export_date}_{run.name}",
            wandb_flag=True
        )    

if __name__ == "__main__":
    ### Get sweep config
    with open('/nfs/research/birney/users/esther/medaka-ziram/sweep.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    ### For passing training parameters from shell script
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--project', type=str, default="Ziram VAE Training", help='The name of the wandb project to save the run to.')
    # # parser.add_argument('--dataset', type=str, default="full_dataset", help="The data used for training, value for 'img_type' parameter when loading dataset. Default is 'full_dataset' to use Fanny's data split.")
    # parser.add_argument('--latent_dim', type=int, default=64, help='The dimension of the latent space (default: 64).')
    # parser.add_argument('--capacity', type=int, default=32, help='Capacity of the encoder/decoder. Corresponds to the number of channels in first hidden layer of encoder/last hidden layer of decoder.')
    # parser.add_argument('--depth', type=int, default=5, help='The number of hidden layers in each of the encoder and decoder.')
    # parser.add_argument('--batch-size', type=int, default=128, help='The batch size (defaults to 128).')
    # parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate (defaults to 1e-4).')
    # parser.add_argument('--dropout_p', type=float, default=0.1, help='Dropout to add to layers.')
    # parser.add_argument('--kld_b', type=float, default=1.0, help='The weight of the KL divergence term of the loss.')
    # parser.add_argument('--epochs', type=int, default=None, help='The number of epochs.')
    # parser.add_argument('--export_dir', type=str, default=f"./results/{export_date}_temp-sweep/", help='The name of the export dir of the model.')
    # args = parser.parse_args()

    # print(str(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Initialize sweep (if sweep not initialized on CLI)
    # sweep_id = wandb.sweep(sweep=sweep_config,
    #                        project="Ziram VAE Training")
    # wandb.agent(sweep_id = sweep_id,
    #             function = sweep_train,
    #             # project = "Ziram VAE Training"
    #             count = 10)

    # If sweep already initialized on CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str)
    args = parser.parse_args()

    wandb.agent(sweep_id = args.sweep_id,
                function = sweep_train,
                project = "Ziram VAE Training",
                count = 2)