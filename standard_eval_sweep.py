### The standard eval script, to be run across all runs of a sweep

import os, sys, pickle
import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import albumentations as A
import wandb
from pathlib import Path
from tqdm import tqdm
sns.set()

sys.path.append(os.path.abspath("."))
from src_vae.train import train_vae
from src_vae.model import VAEModel
from src_vae.data import ZiramF0Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

### Standard eval code; cross-validation on single data partition
CLFS = [('LogReg', LogisticRegression(max_iter=3000)),
        ('RF', RandomForestClassifier(n_estimators=300, random_state=0))]

def report(X, y, name):
    """Run the protocol; print and RETURN (metric rows, confusion matrices)."""
    X = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    y = np.asarray(y).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    n, dims = len(y), X.shape[1]
    rows, cms = [], {}
    print(f"\n=== {name}  (n={n}, dims={dims}, classes={np.bincount(y).tolist()}) ===")
    print("  5-class (severity SC0-4):")
    for nm, clf in CLFS:
        yp = cross_val_predict(clf, X, y, cv=cv)
        r = dict(method=name, task='5class', classifier=nm, n=n, dims=dims,
                 acc=accuracy_score(y, yp), macroF1=f1_score(y, yp, average='macro'),
                 weightedF1=f1_score(y, yp, average='weighted'),
                 quadKappa=cohen_kappa_score(y, yp, weights='quadratic'))
        rows.append(r); cms[f'5class_{nm}'] = confusion_matrix(y, yp)
        print(f"    {nm:7s} acc={r['acc']:.3f} macroF1={r['macroF1']:.3f} "
              f"weightedF1={r['weightedF1']:.3f} quadKappa={r['quadKappa']:.3f}")
    yb = (y > 0).astype(int)
    print("  binary (SC0 vs any-kink):")
    for nm, clf in CLFS:
        yp = cross_val_predict(clf, X, yb, cv=cv)
        r = dict(method=name, task='binary', classifier=nm, n=n, dims=dims,
                 acc=accuracy_score(yb, yp), macroF1=f1_score(yb, yp, average='macro'),
                 weightedF1=f1_score(yb, yp, average='weighted'),
                 quadKappa=cohen_kappa_score(yb, yp))   # binary: unweighted kappa
        rows.append(r); cms[f'binary_{nm}'] = confusion_matrix(yb, yp)
        print(f"    {nm:7s} acc={r['acc']:.3f} F1(affected)={f1_score(yb,yp):.3f} "
              f"macroF1={r['macroF1']:.3f} kappa={r['quadKappa']:.3f}")
    return rows, cms


### Load data
X_train = ZiramF0Dataset(img_type="full_dataset", train=True, val=False, test=False, apply_transform=False)
X_val = ZiramF0Dataset(img_type="full_dataset", train=False, val=True, test=False, apply_transform=False)

loader_train = DataLoader(X_train, batch_size=X_train.__len__(), shuffle=False, num_workers=1, pin_memory=True)
train_all = next(iter(loader_train))
loader_val = DataLoader(X_val, batch_size=X_val.__len__(), shuffle=False, num_workers=1, pin_memory=True)
val_all = next(iter(loader_val))

### Get all directory names in sweep folder
sweep_id = "0a39n1bn"

sweep_dir = f"/nfs/research/birney/users/esther/medaka-ziram/results/sweep_{sweep_id}/"
run_dirs = [p.name for p in Path(sweep_dir).iterdir() if p.is_dir()]

api = wandb.Api() # Initialize wandb api

# Get mapping info between run ID's and run names
runs_map_df = pd.read_csv(f"{sweep_dir}/runs_name_to_id.csv")
runs_map_dict = dict(zip(runs_map_df["run_name"], runs_map_df["run_id"]))

### Iterate through each of the sweep run directories
for i in tqdm(run_dirs):
    model_dir = f"/nfs/research/birney/users/esther/medaka-ziram/results/sweep_{sweep_id}/{i}/"
    
    ### Get run data
    run_name = os.path.basename(model_dir[:-1]).split("_", 1)[1]
    run_id = runs_map_dict[run_name]
    
    run = api.run(f"ey267-university-of-cambridge/Ziram VAE Training/{sweep_id}/{run_id}")

    ### Initialize and load model
    model = VAEModel(
        input_dim=X_train[0][0].shape,
        latent_dim=run.config['latent_dim'],
        capacity=run.config['capacity'],
        depth=run.config['depth'],
        kld_b=run.config['kld_b'],
        batch_size=run.config['batch_size'],
        dropout_p=run.config['dropout_p'],
        device=device
    )

    model_ckpt = torch.load(model_dir + "model.pt")
    model.load_state_dict(model_ckpt,
                        strict=False)
    model.eval()
            
    print(f"Loaded model {i}")

    
    ### Save example reconstructions of validation set
    idx = 0 # Sample index to reconstruct

    sample = X_val[idx]
    recon = model.vae(sample[0].unsqueeze(0).to(device))

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(sample[0].squeeze(0).cpu().detach(), cmap='gray')
    axs[0, 0].set_title(f'Original (idx {idx})')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(recon[0].squeeze(0).squeeze(0).cpu().detach(), cmap='gray')
    axs[0, 1].set_title('Reconstructed')
    axs[0, 1].axis('off')

    idx = 60 # Sample index to reconstruct
    sample = X_val[idx]
    recon = model.vae(sample[0].unsqueeze(0).to(device))

    axs[1, 0].imshow(sample[0].squeeze(0).cpu().detach(), cmap='gray')
    axs[1, 0].set_title(f'Original (idx {idx})')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(recon[0].squeeze(0).squeeze(0).cpu().detach(), cmap='gray')
    axs[1, 1].set_title('Reconstructed')
    axs[1, 1].axis('off')
    plt.savefig(f"{model_dir}/validation_reconstruction.png", dpi=300, bbox_inches='tight')


    ### Get VAE latent means
    train_all_means, train_all_logvars = model.vae.get_latent(
        train_all[0].to(device)
    )
    val_all_means, val_all_logvars = model.vae.get_latent(
        val_all[0].to(device)
    )
    train_all_means = train_all_means.detach().cpu().numpy()
    val_all_means = val_all_means.detach().cpu().numpy()

    ### Run standard evaluation
    std_eval_name = os.path.basename(model_dir[:-1])

    rows, cms = report(X = train_all_means, 
                    y = train_all[5]['severity_score_adjusted'].detach().cpu().numpy(),
                    name = std_eval_name)

    # --- save: per-run metrics, confusion matrices, and append to a master table ---
    output_dir = model_dir
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in std_eval_name)
    run_df = pd.DataFrame(rows)
    run_df.to_csv(os.path.join(output_dir, f'metrics_{safe}.csv'), index=False)
    for k, cm in cms.items():
        np.savetxt(os.path.join(output_dir, f'confusion_{safe}_{k}.csv'), cm, fmt='%d', delimiter=',')
    master = os.path.join(output_dir, 'all_methods_comparison.csv')
    prev = pd.read_csv(master) if os.path.exists(master) else pd.DataFrame()
    if len(prev):   # re-running a name overwrites its old rows
        key = ['method', 'task', 'classifier']
        prev = prev[~prev[key].apply(tuple, 1).isin(run_df[key].apply(tuple, 1))]
    pd.concat([prev, run_df], ignore_index=True).to_csv(master, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    ### Save confusion matrix
    for ax, (name, cm) in zip(axes, cms.items()):
        # Choose labels based on matrix size
        if cm.shape[0] == 2:
            labels = ["Negative", "Positive"]
        else:
            labels = [str(i) for i in range(cm.shape[0])]

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d"
        )

        ax.set_title(name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.grid(False)

    fig.suptitle(f"Confusion matrices for run {std_eval_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cm.png", dpi=300, bbox_inches='tight')

    print(f"\n[OK] saved metrics_{safe}.csv (+ confusion matrices) -> {output_dir}/")
    print(f"[OK] master comparison table -> {master}")
