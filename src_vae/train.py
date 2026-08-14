import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as transF
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import VAEModel
from .data import ZiramF0Dataset

import os
import pandas as pd
import pickle

import wandb
import yaml
import math

os.environ["WANDB_CACHE_DIR"] = "/hps/nobackup/birney/users/esther/"

# Function for denormalizing per-image normalized images (if needed)
# From: https://discuss.pytorch.org/t/why-do-images-look-weird-after-imagenet-normalization/92071/3
def renormalize(tensor):
    minFrom = tensor.min()
    maxFrom = tensor.max()
    minTo = 0
    maxTo=1
    return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))


# Function for visualizing reconstructions in wandb
def log_predictions(model, dataloader, device, epoch):
    model.eval()
    images_to_log = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0][0].to(device) # output torch.Size([1, 352, 352])
            recon = model(x.unsqueeze(0)).squeeze().unsqueeze(0).to(device) # output torch.Size([1, 1, 352, 352])

            # If used per-image normalization
            x = renormalize(x)
            recon = renormalize(recon)

            images_to_log.append(wandb.Image(transF.to_pil_image(x.cpu()), caption="Original"))
            images_to_log.append(wandb.Image(transF.to_pil_image(recon.cpu()), caption="Reconstructed")) # wandb image needs to be shape [1, 352, 352]
            if len(images_to_log) >= 6:  # Limita el número de imágenes
                break
    wandb.log({"examples_epoch_{}".format(epoch): images_to_log})


class CyclicalCosineAnnealing:
    def __init__(self, max_beta: float, cycle_epochs: int,
                 ramp_fraction: float = 0.5, start: float = 0.0):
        """
        max_beta:      the target beta value to ramp up to (e.g. 0.01)
        cycle_epochs:  number of epochs per full cycle (ramp + hold)
        ramp_fraction: fraction of each cycle spent ramping (0-1);
                       the remainder is held at max_beta
        start:         beta value at the start of each ramp (usually 0)
        """
        assert 0 < ramp_fraction <= 1.0, "ramp_fraction must be in (0, 1]"
        self.max_beta = max_beta
        self.cycle_epochs = cycle_epochs
        self.ramp_fraction = ramp_fraction
        self.start = start
        self.ramp_len = cycle_epochs * ramp_fraction
 
    def get_beta(self, epoch: int) -> float:
        pos_in_cycle = epoch % self.cycle_epochs
 
        if pos_in_cycle >= self.ramp_len:
            # Holding phase: stay at max_beta for the rest of the cycle
            return self.max_beta
 
        # Ramping phase: cosine ease-in from `start` to `max_beta`.
        # cos(pi) = -1 -> progress=0 at pos_in_cycle=0
        # cos(0)  =  1 -> progress=1 at pos_in_cycle=ramp_len
        progress = (1 - math.cos(math.pi * pos_in_cycle / self.ramp_len)) / 2
        return self.start + progress * (self.max_beta - self.start)

def train_vae(
    X_train,
    X_test,
    latent_dim: tuple | int = 20,
    capacity: int = 16,
    depth: int = 4,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_epochs: int = 2000,
    dropout_p: float = 0.0,
    kld_b: float = 1.0,
    device: str = "cpu",
    export_dir: str = "./results/temp/",
    wandb_flag: bool = False
):
    if wandb_flag:
        if wandb.run is None: # Check if there is already a wandb run initialized, eg. from sweep
            run = wandb.init(project="Ziram VAE Training", 
                            config={
                                "latent_dim": latent_dim,
                                "capacity": capacity,
                                "depth": depth,
                                "batch_size": batch_size,
                                "kld_b": kld_b,
                                "lr": lr,
                                "n_epochs": n_epochs,
                                "dropout_p": dropout_p
                            },
                            name=f"VAE_latent{latent_dim}_bs{batch_size}")

    device = torch.device(device)

    n_train = X_train.__len__()
    n_test = X_test.__len__()

    # --------- preprocess TRAIN ---------
    loader_train = DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)

    # --------- preprocess TEST using TRAIN stats ---------
    loader_test = DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True)

    # --------- model ---------
    model = VAEModel(
        input_dim=X_train[0][0].shape,
        latent_dim=latent_dim,
        capacity=capacity,
        depth=depth,
        batch_size=batch_size,
        dropout_p=dropout_p,
        kld_b=kld_b,
        device=device,
    )
    model.train()

    # --------- optimizers ---------
    kld_cycle = 8
    opt_vae   = torch.optim.Adam(model.vae.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(opt_vae, mode="min", factor=0.5, patience=((n_epochs // kld_cycle) + 5))

    # --------- kld beta schedule ----------
    # if kld_b >= 1e-2:
    if kld_b >= 1.0:
        kld_schedule = CyclicalCosineAnnealing(max_beta=kld_b, cycle_epochs=n_epochs // kld_cycle, ramp_fraction=0.5)

    # --------- history containers ---------
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kld": [],
        "test_loss": [],
        "test_recon": [],
        "test_kld": []
    }

    # --------- training loop ---------
    best_val_loss = float('inf')
    best_val_epoch = 0
    best_val_recon_loss = float('inf')
    best_val_recon_epoch = 0
    for epoch in range(1, n_epochs + 1):
        # ===== TRAIN PHASE =====
        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0

        # current_beta = kld_schedule.get_beta(epoch) if kld_b >= 1e-2 else kld_b
        current_beta = kld_schedule.get_beta(epoch) if kld_b >= 1.0 else kld_b

        for batch_y, _, _, _, _, _ in loader_train:
            batch_y = batch_y.to(device)

            opt_vae.zero_grad()

            (
                loss,
                recon_loss,
                kld_loss
            ) = model.batch_loss(batch_y)

            loss = recon_loss + current_beta*kld_loss

            loss.backward()

            opt_vae.step()

            bs = batch_y.shape[0]
            total_loss           += loss.item()          * bs
            total_recon          += recon_loss.item()    * bs
            total_kld            += kld_loss.item()      * bs

            if wandb_flag:
                # Log train batch to wandb
                wandb.log({"batch_train_loss": loss.item() * bs, 
                        "batch_train_recon_loss": recon_loss.item() * bs,
                        "batch_train_kld_loss": kld_loss.item() * bs
                        })

        lr_scheduler.step(best_val_recon_loss) # changed from best_val_loss to best_val_recon_loss
        current_lr = opt_vae.param_groups[0]["lr"]

        avg_loss         = total_loss         / n_train
        avg_recon        = total_recon        / n_train
        avg_kld          = total_kld          / n_train

        history["train_loss"].append(avg_loss)
        history["train_recon"].append(avg_recon)
        history["train_kld"].append(avg_kld)

        # ===== TEST REP PHASE =====
        model.eval()

        # Freeze model params
        for p in model.vae.parameters():
            p.requires_grad_(False)

        total_test_loss = 0.0
        total_test_recon = 0.0
        total_test_kld   = 0.0

        for batch_y_test, _, _, _, _, _ in loader_test:
            batch_y_test = batch_y_test.to(device)

            (
                loss_test,
                recon_test,
                kld_test
            ) = model.batch_loss(batch_y_test)

            loss_test = recon_test + current_beta*kld_test

            bs = batch_y_test.shape[0]
            total_test_loss         += loss_test.item()        * bs
            total_test_recon        += recon_test.item()       * bs
            total_test_kld          += kld_test.item()         * bs

            if wandb_flag:
                # Log validation batch to wandb
                wandb.log({"batch_val_loss": loss_test.item(), 
                        "batch_val_recon_loss": recon_test.item(),
                        "batch_val_kld_loss": kld_test.item()
                        })

        avg_test_loss         = total_test_loss         / n_test
        avg_test_recon        = total_test_recon        / n_test
        avg_test_kld          = total_test_kld          / n_test

        history["test_loss"].append(avg_test_loss)
        history["test_recon"].append(avg_test_recon)
        history["test_kld"].append(avg_test_kld)

        # Unfreeze model params
        for p in model.vae.parameters():
            p.requires_grad_(True)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={avg_loss:.4f} "
            f"(recon={avg_recon:.4f}, kld={avg_kld:.4f}) | "
            f"test loss={avg_test_loss:.4f} "
            f"(recon={avg_test_recon:.4f}, kld={avg_test_kld:.4f})"
        )

        if wandb_flag:
            # Log epoch to wandb
            wandb.log({"epoch": epoch,
                    "epoch_train_loss": avg_loss, 
                    "epoch_train_recon_loss": avg_recon,
                    "epoch_train_kld_loss": avg_kld,
                    "epoch_test_loss": avg_test_loss,
                    "epoch_test_recon_loss": avg_test_recon,
                    "epoch_test_kld_loss": avg_test_kld,
                    "best_test_loss": best_val_loss,
                    "best_test_epoch": best_val_epoch,
                    "best_test_recon_loss": best_val_recon_loss,
                    "best_test_recon_epoch": best_val_recon_epoch,
                    "lr": current_lr,
                    "kld_b": current_beta
                    })

            if epoch % 50 == 0:
                log_predictions(model.vae, loader_train, device, epoch)

        # Terminate training if validation loss has not improved for 100 epochs
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            best_val_epoch = epoch
            patience_counter = 0

            if not os.path.exists(export_dir):
                os.makedirs(export_dir+"/best_running_model/")
            
            print("Writing best overall model...")
            torch.save(model.state_dict(), export_dir+f"/best_running_model/best_model_val_loss.pt")
        else:
            patience_counter += 1
            if patience_counter >= ((n_epochs // kld_cycle) + 50):
                break

        if avg_test_recon < best_val_recon_loss:
            best_val_recon_loss = avg_test_recon
            best_val_recon_epoch = epoch

            if not os.path.exists(export_dir):
                os.makedirs(export_dir+"/best_running_model/")

            print("Writing best reconstructing model...")
            torch.save(model.state_dict(), export_dir+f"/best_running_model/best_model_val_recon_loss.pt")


    # Save results
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print("Saved to: ", export_dir)
    
    # Save the model
    torch.save(model.state_dict(), export_dir+"/model.pt")
    # Save the losses history
    history_df = pd.DataFrame(history)
    history_df.to_csv(export_dir + "/history.csv")

    if wandb_flag:
        wandb.finish()
    
    return model, history