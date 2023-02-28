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
from pytorch_lightning.loggers import WandbLogger


# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')


emulator_dataset=forcing_dataset.EmulatorDataset('/scratch/cp3759/pyqg_data/sims/1_step/all.nc',seed=42,normalise="proper",subsample=None,drop_spin_up=True)

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
#config["residual_blocks"]=3
#config["conv_filters"]=3
#config["lr"]=0.001
config["input_channels"]=2
config["output_channels"]=2

def train():
    wandb.init()
    
    config["residual_blocks"]=wandb.config.residual_blocks
    config["lr"]=wandb.config.lr
    config["conv_filters"]=wandb.config.conv_filters
    
    model=base_model.ResNetParallel(config)
    wandb.config["theta learnable parameters"]=sum(p.numel() for p in model.parameters())
    wandb.watch(model, log_freq=1)
    
    logger = WandbLogger()

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=200,
        logger=logger,
        enable_progress_bar=False,
        )

    trainer.fit(model, train_loader, valid_loader)
    wandb.finish()

project_name="resnet_sweeps"

sweep_configuration = {
    'method': 'bayes',
    'name': 'parallel_resnet',
    'metric': {
        'goal': 'minimize', 
        'name': 'valid_loss'
        },
    'parameters': {
        'residual_blocks': {'values': [4,12,24,32,64,128,256]},
        'conv_filters': {'values': [8,16,32,64,128]},
        'lr': {'max': -2.303, 'min': -9.210,'distribution':'log_uniform'}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="%s" % project_name, entity="m2lines")

wandb.agent(sweep_id, function=train, count=30)
