import pickle
import os
import functools
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.loops import FitLoop
import pyqg_explorer.util.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.util.misc as misc
import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import pyqg_explorer.util.performance as performance

model_string="/scratch/cp3759/pyqg_data/models/emulator/fcnn_residuals/fcnn_r_4_10_all.p"
model_beta=misc.load_model(model_string)

config=reg_sys.config
config["epochs"]=5
config["time_horizon"]=10
config["subgrid_models"]=["HRC"]
config["beta_loss"]=100
config["theta_loss"]=0
config["emulator_model"]=model_string

dataset=forcing_dataset.EmulatorForcingDataset('/scratch/cp3759/pyqg_data/sims/%d_step_forcing/' % config["time_horizon"],config["subgrid_models"],
                            channels=4,seed=config["seed"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])

train_loader = DataLoader(
    dataset,
    num_workers=10,
    batch_size=64,
    sampler=SubsetRandomSampler(dataset.train_idx),
)
valid_loader = DataLoader(
    dataset,
    num_workers=10,
    batch_size=64,
    sampler=SubsetRandomSampler(dataset.valid_idx),
)

config["q_mean_upper"]=dataset.q_mean_upper
config["q_mean_lower"]=dataset.q_mean_lower
config["q_std_upper"]=dataset.q_std_upper
config["q_std_lower"]=dataset.q_std_lower
config["s_mean_upper"]=dataset.s_mean_upper
config["s_mean_lower"]=dataset.s_mean_lower
config["s_std_upper"]=dataset.s_std_upper
config["s_std_lower"]=dataset.s_std_lower
config["training_fields"]=len(dataset.train_idx)
config["validation_fields"]=len(dataset.valid_idx)

model=fcnn.FCNN(config)

system=reg_sys.JointRegressionSystem(model,config,model_beta)
system.network_beta.requires_grad=False

wandb.init(project="joint_opt_dev", entity="m2lines",config=config)
wandb.config["theta learnable parameters"]=sum(p.numel() for p in model.parameters())
wandb.watch(model, log_freq=1)

logger = WandbLogger()

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=config["epochs"],
    logger=logger,
    enable_progress_bar=False,
    )

trainer.fit(system, train_loader, valid_loader)

perf=performance.ParameterizationPerformance(model,valid_loader,threshold=1000)

dist_fig=perf.get_distribution_2d()
figure_dist=wandb.Image(dist_fig)
wandb.log({"Distributions": figure_dist})

power_fig=perf.get_power_spectrum()
figure_power=wandb.Image(power_fig)
wandb.log({"Power spectra": figure_power})

field_fig=perf.get_fields()
figure_field=wandb.Image(field_fig)
wandb.log({"Random fields": figure_field})

online_fig=perf.online_comparison()
figure_online=wandb.Image(online_fig)
wandb.log({"Online test": figure_online})

wandb.finish()