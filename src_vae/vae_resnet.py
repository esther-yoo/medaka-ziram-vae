import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class VAE_ResNet(nn.Module):
    def __init__(
            self,
            input_dim: tuple = (1, 352, 352),
            latent_dim: int = 128,
            batch_size: int = 8,
            dropout_p: float = 0.0,
            device: str = "cpu",
            capacity = 64,
            depth = 4
        ):
        
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.capacity = 64 # Capacity of first layer in ResNet
        self.depth = 4 # Number of layers in ResNet
        self.batch_size = batch_size
        self.dropout_p = dropout_p

        print("input_dim: ", input_dim)
        print("latent_dim: ", latent_dim)
        print("capacity: ", capacity)
        print("depth: ", depth)
        print("batch_size: ", batch_size)
        print("dropout: ", dropout_p)

        ### Encoder: ResNet-18 --> decoder_channels = 512
        self.encoder = models.resnet._resnet(models.resnet.BasicBlock, 
                                             [2, 2, 2, 2], 
                                             weights=None, 
                                             progress=True)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()

        self.decoder_channels = 512


        ### Latents
        self.mean = nn.Linear(512, self.latent_dim)
        self.var = nn.Linear(512, self.latent_dim)


        ### Decoder
        self.decoder = nn.ModuleList()

        # self.decoder.append(
        #     nn.Sequential(
        #         nn.Linear(latent_dim,
        #                   self.decoder_channels),
                
        #     )
        # )

        self.decoder.append(
            nn.Sequential(
                nn.Linear(latent_dim,               # Latent dim --> 512
                          self.decoder_channels),
                nn.ReLU(),
                nn.Linear(self.decoder_channels,    # 512 --> 512 * 11 * 11
                          self.decoder_channels * 
                            (self.input_dim[-1] // (2**(self.depth+1))) * 
                            (self.input_dim[-1] // (2**(self.depth+1))))
            # View((-1, self.decoder_channels, 1, 1))
            )
        )

        for i in range(self.depth-1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.capacity*(2**i), 
                                       self.capacity*(2**(i-1)), 
                                       kernel_size=4, 
                                       stride=2, 
                                       padding=1),
                    nn.ReLU()
                )
            )

        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.capacity,
                                    self.capacity,
                                    kernel_size=4, 
                                    stride=2, 
                                    padding=1),
                nn.ConvTranspose2d(self.capacity,
                                    input_dim[0],
                                    kernel_size=4, 
                                    stride=2, 
                                    padding=1),
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
        x = self.encoder(x)

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
                z = z.reshape(mean.shape[0],
                                self.capacity * (2**(self.depth-1)),
                                self.input_dim[-1] // (2**(self.depth+1)),
                                self.input_dim[-1] // (2**(self.depth+1)))

        return z


    def forward(self, x):
        ### Apply encoder
        x = self.encoder(x)

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
                # z = z.view(-1, self.decoder_channels, 1, 1)
                z = z.reshape(x.shape[0],
                                self.capacity * (2**(self.depth-1)),
                                self.input_dim[-1] // (2**(self.depth+1)),
                                self.input_dim[-1] // (2**(self.depth+1)))
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