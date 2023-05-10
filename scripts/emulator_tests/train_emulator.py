import wandb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.util.performance as performance
import pyqg_explorer.dataset.forcing_dataset as forcing_dataset


config=reg_sys.config
config["output_channels"]=2
config["arch"]="FCNN on residuals"
config["epochs"]=200
config["save_path"]="/scratch/cp3759/pyqg_data/models/emulator/fcnn_residuals/"
config["batch_size"]=128
config["subsample"]=None
config["scheduler"]=True
config["subgrid_models"]=["CNN","ZB","BScat","HRC"]
config["input_channels"]=4

def train(time_horizon):
    config["time_horizon"]=time_horizon
    emulator_dataset=forcing_dataset.EmulatorForcingDataset('/scratch/cp3759/pyqg_data/sims/%d_step_forcing/' % config["time_horizon"],config["subgrid_models"],
                                                                     channels=config["input_channels"],seed=config["seed"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])

    config["q_mean_upper"]=emulator_dataset.q_mean_upper
    config["q_mean_lower"]=emulator_dataset.q_mean_lower
    config["q_std_upper"]=emulator_dataset.q_std_upper
    config["q_std_lower"]=emulator_dataset.q_std_lower
    config["training_fields"]=len(emulator_dataset.test_idx)
    config["validation_fields"]=len(emulator_dataset.valid_idx)
    config["save_name"]="fcnnr_%d_%d_step_all_May.p" % (config["input_channels"],config["time_horizon"])

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
    wandb.init(project="pyqg_emulator_timehorizons", entity="m2lines",dir="/scratch/cp3759/pyqg_data/wandb_runs",config=config)
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

    fig_field=perf.get_fields()
    figure_fields=wandb.Image(fig_field)
    wandb.log({"Random fields": figure_fields})

    fig_dist=perf.get_distribution_2d()
    figure_dist=wandb.Image(fig_dist)
    wandb.log({"Distribution": figure_dist})

    wandb.finish()

time_horizons=[1,2,5,10,25,50]

for time_hor in time_horizons:
    train(time_hor)

print("Finito")
