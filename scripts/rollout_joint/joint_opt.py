from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb

import pyqg_explorer.util.misc as misc
import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.util.performance as performance
import pyqg_explorer.models.fcnn as fcnn
import time

import sys

config=reg_sys.config

config["num_sims"]=273
config["subgrid_forcing"]=True
config["epochs"]=120
config["time_horizon"]=2
config["rollout"]=4
config["decay_coeff"]=1
config["batch_size"]=128
config["input_channels"]=2
config["theta_loss"]=1
config["beta_loss"]=float(sys.argv[1])
config["beta_gradients"]=True


config["beta_model_string"]="/scratch/cp3759/pyqg_data/wandb_runs/wandb/run-20230830_234316-gnduwhx5/files/model_weights.pt"
model_beta=misc.load_model(config["beta_model_string"])

## Fix emulator network gradients
if config["beta_gradients"]==False:
    model_beta.requires_grad_(False)

t0 = time.time()
test_dataset=forcing_dataset.RolloutDataset(config["time_horizon"],config["rollout"],file_path="/scratch/cp3759/pyqg_data/sims/rollouts_joint/",num_sims=config["num_sims"],drop_spin_up=True,subgrid_forcing=config["subgrid_forcing"])
print(f"Loading dataset took {time.time() - t0:.1f} seconds.")

config["q_mean_upper"]=test_dataset.q_mean_upper
config["q_mean_lower"]=test_dataset.q_mean_lower
config["q_std_upper"]=test_dataset.q_std_upper
config["q_std_lower"]=test_dataset.q_std_lower
config["s_mean_upper"]=test_dataset.s_mean_upper
config["s_mean_lower"]=test_dataset.s_mean_lower
config["s_std_upper"]=test_dataset.s_std_upper
config["s_std_lower"]=test_dataset.s_std_lower
config["training_fields"]=len(test_dataset.train_idx)
config["validation_fields"]=len(test_dataset.valid_idx)

train_loader_roll = DataLoader(
    test_dataset,
    num_workers=10,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(test_dataset.train_idx),
)

valid_loader_roll = DataLoader(
    test_dataset,
    num_workers=10,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(test_dataset.valid_idx),
    drop_last=True
)


model=fcnn.FCNN(config)

system=reg_sys.JointRollout(model,config,model_beta)

wandb.init(project="joint_rollout", entity="m2lines",config=config,dir="/scratch/cp3759/pyqg_data/wandb_runs")
## Have to update both the wandb config and the config dict that is passed to the CNN
wandb.config["save_path"]=wandb.run.dir
config["save_path"]=wandb.run.dir
config["wandb_url"]=wandb.run.get_url()

wandb.config["cnn learnable parameters"]=sum(p.numel() for p in model.parameters())
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

trainer.fit(system, train_loader_roll, valid_loader_roll)

## Delete training dataset to clear up memory
del train_loader_roll
del valid_loader_roll
del test_dataset

## Test the jointly optimised CNN using the standard framework - can use the offline dataset for this
perf_dataset=forcing_dataset.OfflineDataset("/scratch/cp3759/pyqg_data/sims/0_step/all.nc",seed=config["seed"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])

valid_loader = DataLoader(
    perf_dataset,
    num_workers=10,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(perf_dataset.valid_idx),
)

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

subgrid_fig=perf.subgrid_energy()
figure_subgrid=wandb.Image(subgrid_fig)
wandb.log({"Subgrid energy": figure_subgrid})

online_fig=perf.online_comparison()
figure_online=wandb.Image(online_fig)
wandb.log({"Online test": figure_online})

## Batch of runs for KE(time) test
os.system("bash /home/cp3759/Projects/pyqg_explorer/scripts/KE_datasets/submit_ensembles.sh --wandb_dir %s --wandb_url %s" % (config["save_path"],config["wandb_url"]))


wandb.finish()
