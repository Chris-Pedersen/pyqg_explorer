import fsspec
import xarray as xr
import wandb

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar

import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import pyqg_explorer.models.base_model as base_model

data_full=xr.open_zarr(fsspec.get_mapper(f'/scratch/zanna/data/pyqg/publication/eddy/forcing1.zarr'), consolidated=True)
data_forcing=data_full.q_subgrid_forcing.isel(lev=1)
data_forcing=data_forcing.stack(snapshot=("run","time"))
data_forcing=data_forcing.transpose("snapshot","y","x")

data_q=data_full.q.isel(lev=1)
data_q=data_q.stack(snapshot=("run","time"))
data_q=data_q.transpose("snapshot","y","x")

dataset=forcing_dataset.ForcingDataset(data_q,data_forcing)


train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=SubsetRandomSampler(dataset.train_idx),
)
valid_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=SubsetRandomSampler(dataset.valid_idx),
)

model=base_model.AndrewCNN(1,1)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=40,
    callbacks=None,
    logger=None
)

trainer.fit(model, train_loader, valid_loader)

print("done")