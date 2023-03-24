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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import pyqg_explorer.models.base_model as base_model
import pyqg_explorer.models.resnet as resnet
import pyqg_explorer.util.pbar as pbar
import pyqg_explorer.util.transforms as transforms
import sys

time_horizon=int(sys.argv[1])

emulator_dataset=forcing_dataset.EmulatorDataset('/scratch/cp3759/pyqg_data/sims/%d_step/all.nc' % time_horizon,seed=42,subsample=None,drop_spin_up=True)

train_loader = DataLoader(
    emulator_dataset,
    num_workers=10,
    batch_size=64,
    sampler=SubsetRandomSampler(emulator_dataset.train_idx),
)
valid_loader = DataLoader(
    emulator_dataset,
    num_workers=10,
    batch_size=64,
    sampler=SubsetRandomSampler(emulator_dataset.valid_idx),
)

config={}
config["input_channels"]=2
config["output_channels"]=2
config["arch"]="res net"

q,s_true=emulator_dataset.__getitem__(emulator_dataset.valid_idx[0])
s_true=s_true.detach().squeeze().numpy()
q_plot=q.detach().squeeze().numpy()


def train():
    wandb.init()
    
    config["lr"]=wandb.config.lr
    config["wd"]=wandb.config.wd
    config["dropout"]=wandb.config.dropout
    config["residual_blocks"]=wandb.config.residual_blocks
    config["conv_layers"]=wandb.config.conv_layers
    config["conv_filters"]=wandb.config.conv_filters
    
    model=resnet.ResNet(config)
    wandb.config["theta learnable parameters"]=sum(p.numel() for p in model.parameters())
    wandb.watch(model, log_freq=1)
    
    logger = WandbLogger()
    lr_monitor=LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=200,
        logger=logger,
        enable_progress_bar=False,
        callbacks=[lr_monitor]
        )

    trainer.fit(model, train_loader, valid_loader)
    
    s_pred=model(q.unsqueeze(0)).detach().squeeze().numpy()
    
    fig, axs = plt.subplots(2,5,figsize=(13,4))
    ax=axs[0][0].imshow(q_plot[0], cmap='bwr')
    fig.colorbar(ax, ax=axs[0][0])
    axs[0][0].set_xticks([]); axs[0][0].set_yticks([])
    axs[0][0].set_title("q field")

    ax=axs[0][1].imshow(s_true[0], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[0][1])
    axs[0][1].set_xticks([]); axs[0][1].set_yticks([])
    axs[0][1].set_title("True i+1")

    ax=axs[0][2].imshow(s_pred[0], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[0][2])
    axs[0][2].set_xticks([]); axs[0][2].set_yticks([])
    axs[0][2].set_title("Emulated i+1")

    ax=axs[0][3].imshow(s_true[0]-s_pred[0], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[0][3])
    axs[0][3].set_xticks([]); axs[0][3].set_yticks([])
    axs[0][3].set_title("Emulator residuals")

    ax=axs[0][4].imshow(s_true[0]-q_plot[0], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[0][4])
    axs[0][4].set_xticks([]); axs[0][4].set_yticks([])
    axs[0][4].set_title("Timestep residuals")

    ax=axs[1][0].imshow(q_plot[1], cmap='bwr')
    fig.colorbar(ax, ax=axs[1][0])
    axs[1][0].set_xticks([]); axs[1][0].set_yticks([])

    ax=axs[1][1].imshow(s_true[1], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[1][1])
    axs[1][1].set_xticks([]); axs[1][1].set_yticks([])

    ax=axs[1][2].imshow(s_pred[1], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[1][2])
    axs[1][2].set_xticks([]); axs[1][2].set_yticks([])

    ax=axs[1][3].imshow(s_true[1]-s_pred[1], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[1][3])
    axs[1][3].set_xticks([]); axs[1][3].set_yticks([])

    ax=axs[1][4].imshow(s_true[1]-q_plot[1], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[1][4])
    axs[1][4].set_xticks([]); axs[1][4].set_yticks([])


    fig.tight_layout()

    figure_fields=wandb.Image(fig)
    wandb.log({"Random fields": figure_fields})
    wandb.finish()

project_name="resnet_timestep_sweeps"

sweep_configuration = {
    'method': 'random',
    'name': 'resnet_%d' % time_horizon,
    'metric': {
        'goal': 'minimize', 
        'name': 'valid_loss'
        },
    'parameters': {
        'residual_blocks': {'values': [4,6,8,10,12,16]},
        'conv_filters': {'values': [4,12,16,20,26,32,38,46,64]},
        'conv_layers': {'values': [2,3,4,5]},
        'lr': {'max': -2.303, 'min': -9.210,'distribution':'log_uniform'},
        'wd': {'max': -2.303, 'min': -11.513,'distribution':'log_uniform'},
        'dropout': {'max': -1.204, 'min': -6.908,'distribution':'log_uniform'}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="%s" % project_name, entity="m2lines")

wandb.agent(sweep_id, function=train, count=50)