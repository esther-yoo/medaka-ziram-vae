import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VAE(nn.Module):
    def __init__(
            self,
            input_dim: tuple = (1, 2048, 2048),
            latent_dim: int = 128,
            capacity: int = 16,
            depth: int = 4,
            batch_size: int = 8,
            dropout_p: float = 0.0,
            device: str = "cpu"
        ):
        
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.capacity = capacity
        self.depth = depth
        self.batch_size = batch_size
        self.dropout_p = dropout_p

        print("input_dim: ", input_dim)
        print("latent_dim: ", latent_dim)
        print("capacity: ", capacity)
        print("depth: ", depth)
        print("batch_size: ", batch_size)
        print("dropout: ", dropout_p)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.criterion = nn.MSELoss(reduction="none")

        ### Encoder
        for i in range(self.depth):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(input_dim[0] if i == 0 else self.capacity*(2**(i-1)), 
                              self.capacity if i == 0 else self.capacity*(2**i), 
                              kernel_size=4, 
                              stride=2, 
                              padding=1),
                    nn.BatchNorm2d(self.capacity*(2**i)),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout_p),
                    nn.Flatten() if i == self.depth - 1 else nn.Identity()
                )
            )
        
        ### Latents
        self.mean = nn.Linear(self.capacity * (2**(self.depth-1)) *
                                input_dim[-1] // (2**self.depth) *
                                input_dim[-1] // (2**self.depth),
                              latent_dim)
        self.var = nn.Linear(self.capacity * (2**(self.depth-1)) *
                                input_dim[-1] // (2**self.depth) *
                                input_dim[-1] // (2**self.depth),
                              latent_dim)

        ### Decoder
        self.decoder.append(
            nn.Linear(latent_dim,
                      self.capacity * (2**(self.depth-1)) *
                        input_dim[-1] // (2**self.depth) *
                        input_dim[-1] // (2**self.depth))
        )
        
        for i in range(self.depth-1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.capacity*(2**i), 
                                       self.capacity*(2**(i-1)), 
                                       kernel_size=4, 
                                       stride=2, 
                                       padding=1),
                    nn.BatchNorm2d(self.capacity*(2**(i-1))),
                    nn.ReLU()
                )
            )
        
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.capacity,
                                    input_dim[0],
                                    kernel_size=4, 
                                    stride=2, 
                                    padding=1),
                # nn.Sigmoid()
            )
        )
    

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    

    def get_latent(self, x):
        """
        Given an image, return the mean, logvar latent vectors
        """
        ### Apply encoder
        for i in range(self.depth):
            x = self.encoder[i](x)

        ### Get latents
        mean = self.mean(x)
        logvar = self.var(x)
        
        return mean, logvar
    

    def decode(self, mean, logvar):
        """
        Given the mean, logvar, return the reconstructed image
        """
        z = self.reparameterize(mean, logvar)

        ### Apply decoder
        for i in range(self.depth+1):
            z = self.decoder[i](z)
            if i == 0:
                z = z.reshape(mean.shape[0], # batch
                                self.capacity * (2**(self.depth-1)),
                                self.input_dim[-1] // (2**self.depth),
                                self.input_dim[-1] // (2**self.depth))

        return z


    def forward(self, x):
        ### Apply encoder
        for i in range(self.depth):
            x = self.encoder[i](x)

        ### Get latents
        mean = self.mean(x)
        logvar = self.var(x)
        logvar = torch.clamp(logvar, max = 10.0)
        
        ### Reparameterize
        z = self.reparameterize(mean, logvar)

        ### Apply decoder
        for i in range(self.depth+1):
            z = self.decoder[i](z)
            if i == 0:
                ### reshape to (B, 128, 128, 128) (or, the dimensions from last conv layer of encoder)
                z = z.reshape(x.shape[0],
                                self.capacity * (2**(self.depth-1)),
                                self.input_dim[-1] // (2**self.depth),
                                self.input_dim[-1] // (2**self.depth))
        return z
    
    def compute_loss(self, x):
        """
        Given input image, compute its associated losses
        """
        # Calculate mean, logvar of input image
        mean, logvar = self.get_latent(x)
        logvar = torch.clamp(logvar, max = 10.0)

        # Decode
        recon_x = self.decode(mean, logvar)

        # Without sigmoid
        # print(recon_x.min(), recon_x.max())

        # Calculate losses
        # print("mean: ", mean)
        # print("logvar: ", logvar)
        # recon_loss = (self.criterion(recon_x, x)).mean()
        # recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum").mean(dim=0) / (x.shape[2] * x.shape[3])
        recon_loss = F.mse_loss(recon_x, x, reduction="sum").mean(dim=0) / (x.shape[2] * x.shape[3])
        # print(x.shape)
        kld_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)).mean(dim=0) / (x.shape[2] * x.shape[3])
        # print("recon_loss: ", recon_loss)
        # print("kld_loss: ", kld_loss)

        return recon_loss, kld_loss