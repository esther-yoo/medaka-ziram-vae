import torch
import torch.nn as nn
from .vae import VAE
from .vae_resnet import VAE_ResNet


class VAEModel(nn.Module):
    def __init__(
        self,
        input_dim: tuple = (1, 2048, 2048),
        latent_dim: tuple | int = 20,
        capacity: int = 16,
        depth: int = 4,
        batch_size: int = 8,
        dropout_p: float = 0.0,
        kld_b: float = 1.0,
        device="cpu",
    ):

        super().__init__()
        self.device = torch.device(device)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.kld_b = kld_b

        # self.vae = VAE(
        #     input_dim = self.input_dim,
        #     latent_dim = self.latent_dim,
        #     capacity = capacity,
        #     depth = depth,
        #     batch_size = self.batch_size,
        #     dropout_p = self.dropout_p,
        #     device = self.device
        # )
        # print("Model is standard CNN")

        self.vae = VAE_ResNet(
            input_dim = self.input_dim,
            latent_dim = self.latent_dim,
            batch_size = self.batch_size,
            device = self.device
        )
        print("Model is ResNet")

        self.to(self.device)


    def batch_loss(self, batch_x):
        batch_x = batch_x.to(self.device)
        out_x = self.vae(batch_x)

        recon_loss, kld_loss = self.vae.compute_loss(batch_x)
        recon_loss = recon_loss #/ batch_x.shape[0]
        kld_loss = kld_loss 

        total = recon_loss + self.kld_b*kld_loss

        return total, recon_loss, kld_loss