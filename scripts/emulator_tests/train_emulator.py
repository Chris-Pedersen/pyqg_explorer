import fsspec
import xarray as xr
import matplotlib.pyplot as plt
import sys
import pickle
import copy
import os

import torch
import math
import numpy as np
import wandb
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import pyqg_explorer.models.base_model as base_model
import pyqg_explorer.util.pbar as pbar
import pyqg_explorer.util.transforms as transforms
import pyqg_explorer.util.divergence_dataset as divergence_dataset
import pyqg_explorer.util.powerspec as powerspec
from pytorch_lightning.loggers import WandbLogger

from sklearn.metrics import r2_score

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

## Arch:
## "FCNN"
## "ResNetChoke"
## "ResNetParallel"
## "ResNetSingle"

arch="FCNN"
lev="both"
seed=123
batch_size=64
input_channels=2
output_channels=2
activation="ReLU"
epochs=200
subsample=None
conv_layers=5
normalise="proper"
drop_spin_up="False"
residual_blocks=5
conv_filters=64
lr=0.0001

## Wandb config file
config={"lev":lev,
        "seed":seed,
        "lr":lr,
        "batch_size":batch_size,
        "input_channels":input_channels,
        "output_channels":output_channels,
        "activation":activation,
        "arch":arch,
        "residual_blocks":residual_blocks,
        "conv_filters":conv_filters,
        "conv_layers":conv_layers,
        "drop_spin_up":drop_spin_up,
        "epochs":epochs,
        "subsample":subsample,
        "normalise":normalise}

emulator_dataset=forcing_dataset.EmulatorDataset('/scratch/cp3759/pyqg_data/sims/1_step/all.nc',seed=config["seed"],normalise=config["normalise"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])

config["q_mean_upper"]=emulator_dataset.q_mean_upper
config["q_mean_lower"]=emulator_dataset.q_mean_lower
config["q_std_upper"]=emulator_dataset.q_std_upper
config["q_std_lower"]=emulator_dataset.q_std_lower
config["training_fields"]=len(emulator_dataset.test_idx)
config["validation_fields"]=len(emulator_dataset.valid_idx)

train_loader = DataLoader(
    emulator_dataset,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(emulator_dataset.train_idx),
)
valid_loader = DataLoader(
    emulator_dataset,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(emulator_dataset.valid_idx),
)

if config["arch"]=="FCNN":
    model_theta=base_model.AndrewCNN(config)
elif config["arch"]=="ResNetChoke":
    model_theta=base_model.ResNetChoke(config)
elif config["arch"]=="ResNetParallel":
    model_theta=base_model.ResNetParallel(config)
elif config["arch"]=="ResNetSingle":
    model_theta=base_model.ResNetSingle(config)
model_theta.to(device)

config["theta learnable parameters"]=sum(p.numel() for p in model_theta.parameters())

wandb.init(project="pyqg_emulator_spinup", entity="m2lines",config=config)
wandb.watch(model_theta, log_freq=1)

# optimizer parameters
beta1 = 0.5
beta2 = 0.999
lr = config["lr"]
wd = 0.001

logger = WandbLogger()

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=config["epochs"],
    logger=logger,
    enable_progress_bar=False,
    )

trainer.fit(model_theta, train_loader, valid_loader)

x_maps=torch.tensor(()).to("cpu")
y_true=torch.tensor(()).to("cpu")
y_pred=torch.tensor(()).to("cpu")

model_theta.eval()
model_theta.to("cpu")

for i, data in enumerate(valid_loader, 0):
    ## x_data is ordered in [pv, dqdt, s]
    x_data, y_data = data
    #print(x_data.device)
    #x_data=x_data.to(device)

    ## First network
    y_hat = model_theta(x_data[:,0:2,:,:]) ## Takes in PV, outputs S
    
    x_data_cpu=x_data.to("cpu")
    y_data_cpu=y_data.to("cpu")
    y_hat_cpu=y_hat.to("cpu")
    
    del x_data
    del y_hat
    
    x_maps=torch.cat((x_maps,x_data_cpu),dim=0)
    y_true=torch.cat((y_true,y_data_cpu),dim=0)
    y_pred=torch.cat((y_pred,y_hat_cpu),dim=0)

## Convert validation fields to numpy arrays
x_np=x_maps.squeeze().cpu().detach().numpy()
y_np=y_true.squeeze().cpu().detach().numpy()
y_pred_np=y_pred.squeeze().cpu().detach().numpy()

## Estimate R2
r2_upper=r2_score(y_np[:,0,:,:].flatten(),y_pred_np[:,0,:,:].flatten())
r2_lower=r2_score(y_np[:,1,:,:].flatten(),y_pred_np[:,1,:,:].flatten())

map_index=25

fig, axs = plt.subplots(2,4,figsize=(15,6))
ax=axs[0][0].imshow(x_np[map_index][0], cmap='bwr')
fig.colorbar(ax, ax=axs[0][0])
axs[0][0].set_xticks([]); axs[0][0].set_yticks([])
axs[0][0].set_title("q field")

ax=axs[0][1].imshow(y_np[map_index][0], cmap='bwr', interpolation='none')
fig.colorbar(ax, ax=axs[0][1])
axs[0][1].set_xticks([]); axs[0][1].set_yticks([])
axs[0][1].set_title("True i+1")

ax=axs[0][2].imshow(y_pred_np[map_index][0], cmap='bwr', interpolation='none')
fig.colorbar(ax, ax=axs[0][2])
axs[0][2].set_xticks([]); axs[0][2].set_yticks([])
axs[0][2].set_title("Emulated i+1")

ax=axs[0][3].imshow(y_np[map_index][0]-y_pred_np[map_index][0], cmap='bwr', interpolation='none')
fig.colorbar(ax, ax=axs[0][3])
axs[0][3].set_xticks([]); axs[0][3].set_yticks([])
axs[0][3].set_title("Residuals")
fig.tight_layout()

ax=axs[1][0].imshow(x_np[map_index][1], cmap='bwr')
fig.colorbar(ax, ax=axs[1][0])
axs[1][0].set_xticks([]); axs[1][0].set_yticks([])

ax=axs[1][1].imshow(y_np[map_index][1], cmap='bwr', interpolation='none')
fig.colorbar(ax, ax=axs[1][1])
axs[1][1].set_xticks([]); axs[1][1].set_yticks([])

ax=axs[1][2].imshow(y_pred_np[map_index][1], cmap='bwr', interpolation='none')
fig.colorbar(ax, ax=axs[1][2])
axs[1][2].set_xticks([]); axs[1][2].set_yticks([])

ax=axs[1][3].imshow(y_np[map_index][1]-y_pred_np[map_index][1], cmap='bwr', interpolation='none')
fig.colorbar(ax, ax=axs[1][3])
axs[1][3].set_xticks([]); axs[1][3].set_yticks([])
fig.tight_layout()

figure_fields=wandb.Image(fig)
wandb.log({"Random fields": figure_fields})

fig=plt.figure()
percent_error=(np.abs(((y_np-y_pred_np)/y_np)).flatten())*100
plt.title("Histogram of percentage error in each pixel over validation set")
plt.hist(percent_error,bins=250,range=[0,80]);
plt.xlabel("Percentage error on emulated field")
plt.yscale("log")

figure_hist=wandb.Image(fig)
wandb.log({"Hist of percentage error": figure_hist})

wandb.run.summary["r2_upper"]=r2_upper
wandb.run.summary["r2_lower"]=r2_lower
wandb.finish()
