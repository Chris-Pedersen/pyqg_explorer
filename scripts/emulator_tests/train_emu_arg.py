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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.util.pbar as pbar
import pyqg_explorer.util.transforms as transforms
import cmocean
from sklearn.metrics import r2_score


config=reg_sys.config
config["output_channels"]=2
config["arch"]="FCNN on residuals"
config["epochs"]=125
config["save_path"]="/scratch/cp3759/pyqg_data/models/emulator/fcnn_residuals/"
config["time_horizon"]=int(sys.argv[1])
config["batch_size"]=128
config["subsample"]=None
config["scheduler"]=True

def train(input_channels,sub_models):
    config["subgrid_models"]=sub_models
    config["input_channels"]=input_channels
    emulator_dataset=forcing_dataset.EmulatorForcingDataset('/scratch/cp3759/pyqg_data/sims/%d_step_forcing/' % config["time_horizon"],config["subgrid_models"],
                                                                     channels=config["input_channels"],seed=config["seed"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])

    config["q_mean_upper"]=emulator_dataset.q_mean_upper
    config["q_mean_lower"]=emulator_dataset.q_mean_lower
    config["q_std_upper"]=emulator_dataset.q_std_upper
    config["q_std_lower"]=emulator_dataset.q_std_lower
    config["training_fields"]=len(emulator_dataset.test_idx)
    config["validation_fields"]=len(emulator_dataset.valid_idx)
    config["save_name"]="fcnnr_%d_%d_step_%s.p" % (config["input_channels"],config["time_horizon"],config["subgrid_models"])

    train_loader = DataLoader(
        emulator_dataset,
        num_workers=10,
        batch_size=config["batch_size"],
        sampler=SubsetRandomSampler(emulator_dataset.train_idx),
    )
    valid_loader = DataLoader(
        emulator_dataset,
        num_workers=10,
        batch_size=config["batch_size"],
        sampler=SubsetRandomSampler(emulator_dataset.valid_idx),
    )

    model=fcnn.FCNN(config)
    print(model)
    wandb.init(project="pyqg_emulator", entity="m2lines",dir="/scratch/cp3759/pyqg_data/wandb_runs",config=config)
    wandb.config["theta learnable parameters"]=sum(p.numel() for p in model.parameters())
    wandb.watch(model, log_freq=1)
    
    system=reg_sys.ResidualRegressionSystem(model,config)
    
    logger = WandbLogger()
    lr_monitor=LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=config["epochs"],
        logger=logger,
        enable_progress_bar=False,
        callbacks=[lr_monitor]
        )

    trainer.fit(system, train_loader, valid_loader)
    model.save_model()
    def get_distribution(dataset,model,subgrid_model):
        valid_loader = DataLoader(
            emulator_dataset,
            num_workers=10,
            batch_size=64,
            sampler=SubsetRandomSampler(dataset.model_splits[subgrid_model]["valid"]),
        )
        y_hat=[]
        y_true=[]
        for data in valid_loader:
            x_data=data[0]
            y_data=data[1]
            y_pred=model(x_data)
            y_hat.append(y_pred)
            y_true.append(y_data-x_data[:,0:2,:,:])
        hat=torch.vstack(y_hat).detach().numpy().flatten()
        truth=torch.vstack(y_true).detach().numpy().flatten()
        r2=r2_score(truth,hat)

        fig=plt.figure()
        plt.title(r"%s, $R^2$=%.3f" % (subgrid_model,r2))
        plt.hist(truth,bins=100,alpha=0.5,label="truth");
        plt.hist(hat,bins=100,alpha=0.5,label="predicted");
        plt.legend()
        figure_dist=wandb.Image(fig)
        wandb.log({"Distribution %s" % subgrid_model: figure_dist})

        fig, axs = plt.subplots(2, 4,figsize=(15,6))
        ax=axs[0][0].imshow(x_data[0,0], cmap=cmocean.cm.balance)
        fig.colorbar(ax, ax=axs[0][0])
        axs[0][0].set_xticks([]); axs[0][0].set_yticks([])
        axs[0][0].set_title("i")

        ax=axs[0][1].imshow(y_data[0,0]-x_data[0,0], cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(ax, ax=axs[0][1])
        axs[0][1].set_xticks([]); axs[0][1].set_yticks([])
        axs[0][1].set_title("i+dt - i")

        ax=axs[0][2].imshow(y_pred[0,0].detach().numpy(), cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(ax, ax=axs[0][2])
        axs[0][2].set_xticks([]); axs[0][2].set_yticks([])
        axs[0][2].set_title("predicted i+dt -i")

        ax=axs[0][3].imshow(y_pred[0,0].detach().numpy()-y_data[0,0].detach().numpy()-x_data[0,0].detach().numpy(), cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(ax, ax=axs[0][3])
        axs[0][3].set_xticks([]); axs[0][3].set_yticks([])
        axs[0][3].set_title("True - predicted")
        fig.tight_layout()

        ax=axs[1][0].imshow(x_data[0,1], cmap=cmocean.cm.balance)
        fig.colorbar(ax, ax=axs[1][0])
        axs[1][0].set_xticks([]); axs[1][0].set_yticks([])

        ax=axs[1][1].imshow(y_data[0,1]-x_data[0,1], cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(ax, ax=axs[1][1])
        axs[1][1].set_xticks([]); axs[1][1].set_yticks([])

        ax=axs[1][2].imshow(y_pred[0,1].detach().numpy(), cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(ax, ax=axs[1][2])
        axs[1][2].set_xticks([]); axs[1][2].set_yticks([])

        ax=axs[1][3].imshow(y_pred[0,1].detach().numpy()-y_data[0,1].detach().numpy()-x_data[0,1].detach().numpy(), cmap=cmocean.cm.balance, interpolation='none')
        fig.colorbar(ax, ax=axs[1][3])
        axs[1][3].set_xticks([]); axs[1][3].set_yticks([])
        fig.tight_layout()

        figure_fields=wandb.Image(fig)
        wandb.log({"Random fields %s" % subgrid_model: figure_fields})
        return
    
    for sub in sub_models:
        get_distribution(emulator_dataset,model,sub)
    
    wandb.finish()

input_channels=[2,4]
smodels=[["CNN"],["ZB"],["BScat"],["CNN","ZB","BScat"],["HRC"],["CNN","ZB","BScat","HRC"]]

for in_chan in input_channels:
    for smodel in smodels:
        train(in_chan,smodel)

print("Finito")
