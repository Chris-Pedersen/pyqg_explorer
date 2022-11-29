import fsspec
import xarray as xr
import wandb
from sklearn.metrics import r2_score

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar

import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import pyqg_explorer.models.base_model as base_model
import pyqg_explorer.util.pbar as pbar

lev=1
forcing=1
seed=123
batch_size=64
input_channels=1
output_channels=1
activation="ReLU"
save_path="/scratch/cp3759/pyqg_data/models"
save_name="new_test_lower_100.pt"
arch="RossCNN"
epochs=100
subsample=None
normalise=True
lr=0.001

## Wandb config file
config={"lev":lev,
        "forcing":forcing,
        "seed":seed,
        "lr":lr,
        "batch_size":batch_size,
        "input_channels":input_channels,
        "output_channels":output_channels,
        "activation":activation,
        "save_name":save_name,
        "save_path":save_path,
        "arch":arch,
        "epochs":epochs,
        "subsample":subsample,
        "normalise":normalise}

data_full=xr.open_zarr(fsspec.get_mapper(f'/scratch/zanna/data/pyqg/publication/eddy/forcing%s.zarr' % forcing), consolidated=True)
data_forcing=data_full.q_subgrid_forcing.isel(lev=lev)
data_forcing=data_forcing.stack(snapshot=("run","time"))
data_forcing=data_forcing.transpose("snapshot","y","x")

data_q=data_full.q.isel(lev=lev)
data_q=data_q.stack(snapshot=("run","time"))
data_q=data_q.transpose("snapshot","y","x")

del data_full

dataset=forcing_dataset.ForcingDataset(data_q,data_forcing,config["seed"],normalise=config["normalise"],subsample=config["subsample"])
config["q_mean"]=dataset.q_mean
config["q_std"]=dataset.q_std
config["s_mean"]=dataset.s_mean
config["s_std"]=dataset.s_std
config["training_fields"]=len(dataset.test_idx)
config["validation_fields"]=len(dataset.valid_idx)

train_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(dataset.train_idx),
)
valid_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(dataset.valid_idx),
)

model=base_model.AndrewCNN(config)
wandb.init(project="pyqg_cnns", entity="chris-pedersen",config=config)
logger = WandbLogger()

trainer = pl.Trainer(
    default_root_dir=model.config["save_path"],
    accelerator="auto",
    max_epochs=model.config["epochs"],
    callbacks=pbar.ProgressBar(),
    logger=WandbLogger()
)

trainer.fit(model, train_loader, valid_loader)

model.save_model()

## Get R2
x_maps=torch.tensor(())
y_true=torch.tensor(())
y_pred=torch.tensor(())

for valid in valid_loader:
    x=valid[0]
    y=valid[1]
    y_hat=model(x)
    ## Concat to form tensors containing all maps
    x_maps=torch.cat((x_maps,x),dim=0)
    y_true=torch.cat((y_true,y),dim=0)
    y_pred=torch.cat((y_pred,y_hat),dim=0)
    
print("done")

## Convert to numpy
x_np=x_maps.squeeze().detach().numpy()
y_np=y_true.squeeze().detach().numpy()
y_pred_np=y_pred.squeeze().detach().numpy()

r2=r2_score(y_np.flatten(),y_pred_np.flatten())
wandb.run.summary["r2_score"]=r2
wandb.run.summary["learnable_parameters"]=sum(p.numel() for p in model.parameters())
wandb.finish()