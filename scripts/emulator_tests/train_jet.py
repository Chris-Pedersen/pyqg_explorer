import wandb
import copy

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.models.unet as unet
import pyqg_explorer.performance.emulator_performance as perf

import numpy as np

def train(rollout,subsample=None):
    config=copy.deepcopy(reg_sys.config)

    config["decay_coeff"]=0.1
    config["epochs"]=120
    config["increment"]=5
    config["rollout"]=rollout
    config["decay_coeff"]=0.2
    config["batch_size"]=128
    config["subsample"]=subsample
    config["eddy"]=False
    ## If using the unet we use for diffusion
    ## need to add time parameters
    config["dim_mults"]=[2,4]
    config["base_dim"]=32
    config["timesteps"]=2
    config["time_embedding_dim"]=2

    test_dataset=forcing_dataset.EmulatorDatasetTorch(config["increment"],config["rollout"],subsample=config["subsample"])

    ## Need to save renormalisation factors for when the CNN is plugged into pyqg
    config["q_mean_upper"]=test_dataset.q_mean_upper
    config["q_mean_lower"]=test_dataset.q_mean_lower
    config["q_std_upper"]=test_dataset.q_std_upper
    config["q_std_lower"]=test_dataset.q_std_lower
    config["training_fields"]=len(test_dataset.train_idx)
    config["validation_fields"]=len(test_dataset.valid_idx)

    train_loader = DataLoader(
        test_dataset,
        num_workers=10,
        batch_size=config["batch_size"],
        sampler=SubsetRandomSampler(test_dataset.train_idx),
    )
    valid_loader = DataLoader(
        test_dataset,
        num_workers=10,
        batch_size=config["batch_size"],
        sampler=SubsetRandomSampler(test_dataset.valid_idx),
    )

    wandb.init(project="torch_emu", entity="m2lines",config=config,dir="/scratch/cp3759/pyqg_data/wandb_runs")
    wandb.config["save_path"]=wandb.run.dir
    config["save_path"]=wandb.run.dir
    config["wandb_url"]=wandb.run.get_url()

    model=unet.Unet(config)
    #model=fcnn.FCNN(config)

    wandb.config["cnn learnable parameters"]=sum(p.numel() for p in model.parameters())
    wandb.watch(model, log_freq=1)

    system=reg_sys.ResidualRolloutTorch(model,config)

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

    emu_perf=perf.EmulatorMSE(model)
    fig_mse=emu_perf.get_short_MSEs()
    figure_mse=wandb.Image(fig_mse)
    wandb.log({"Short MSE": figure_mse})

    model.save_model()

    wandb.finish()

rollouts=[1,2,4,6]

for rollout in rollouts:
    train(rollout,subsample=70000)

print("Finito")
