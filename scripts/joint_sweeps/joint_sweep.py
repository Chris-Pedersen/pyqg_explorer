import wandb
import sys
import copy
import os
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.util.performance as performance
import pyqg_explorer.util.misc as misc
import pyqg_explorer.dataset.forcing_dataset as forcing_dataset

import pickle
import os
import functools
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.loops import FitLoop
import pyqg_explorer.util.transforms as transforms


def train(beta_loss):
    config=copy.deepcopy(reg_sys.config)
    time_hor=2
    config["drop_spin_up"]=True
    config["scheduler"]=False
    config["beta_gradients"]=False
    config["beta_loss"]=beta_loss
    config["time_horizon"]=2
    config["scheduler"]=True
    ## Emulator model used for forward prediction

    config["epochs"]=120
    config["subgrid_models"]=["HRC"]
    config["theta_loss"]=1

    ## Remove save_path if it exists..
    try:
        config.pop("save_path")
    except:
        pass

    config["model_string"]="/scratch/cp3759/pyqg_data/wandb_runs/wandb/run-20230830_234316-gnduwhx5/files/model_weights.pt" ## dt=2, trained ove 4 decaying rollouts
    model_beta=misc.load_model(config["model_string"])
    model_beta.requires_grad_(False)
    dataset=forcing_dataset.EmulatorForcingDataset('/scratch/cp3759/pyqg_data/sims/%d_step_forcing/' % config["time_horizon"],config["subgrid_models"],
                                channels=4,seed=config["seed"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])

    ## Fix emulator network gradients
    if config["beta_gradients"]==False:
        model_beta.requires_grad_(False)


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

    wandb.init(project="joint_dt2", entity="m2lines",config=config,dir="/scratch/cp3759/pyqg_data/wandb_runs")
    ## Have to update both the wandb config and the config dict that is passed to the CNN
    wandb.config["save_path"]=wandb.run.dir
    config["save_path"]=wandb.run.dir
    config["wandb_url"]=wandb.run.get_url()

    model=fcnn.FCNN(config)

    system=reg_sys.JointRegressionSystem(model,config,model_beta)

    wandb.config["theta learnable parameters"]=sum(p.numel() for p in model.parameters())
    wandb.watch(model, log_freq=1)

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

    perf=performance.ParameterizationPerformance(model,valid_loader,threshold=5000)

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

    subgrid_fig=perf.subgrid_energy()
    figure_subgrid=wandb.Image(subgrid_fig)
    wandb.log({"Subgrid energy": figure_subgrid})

    model.save_model()

    ## Run KE sims
    os.system("bash /home/cp3759/Projects/pyqg_explorer/scripts/KE_datasets/submit_ensembles.sh --wandb_dir %s --wandb_url %s" % (config["save_path"],config["wandb_url"]))

    wandb.finish()

loss_betas=[1,10,100,1000,10000,100000]
for beta_loss in loss_betas:
    train(beta_loss)

print("Finito")