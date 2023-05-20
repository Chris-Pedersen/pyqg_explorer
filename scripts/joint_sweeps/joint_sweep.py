import wandb
import sys
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


config=reg_sys.config
time_hor=int(sys.argv[1])
config["drop_spin_up"]=True

def train(time_horizon,beta_loss):
    config["beta_loss"]=beta_loss
    config["time_horizon"]=time_horizon
    ## Emulator model used for forward prediction
    config["model_string"]="/scratch/cp3759/pyqg_data/models/emulator/fcnn_residuals/fcnnr_4_%d_step_all_May.p" % config["time_horizon"]
    model_beta=misc.load_model(config["model_string"])
    config["epochs"]=100
    config["subgrid_models"]=["HRC"]
    config["theta_loss"]=1
    config["save_path"]="/scratch/cp3759/pyqg_data/models/joint_May_nospin"
    config["save_name"]="joint_beta%d_time%d.p" % (config["beta_loss"],config["time_horizon"])

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

    wandb.init(project="joint_opt_sweep", entity="m2lines",config=config,dir="/scratch/cp3759/pyqg_data/wandb_runs")
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

    model.save_model()
    wandb.finish()

loss_betas=[1,10,100,1000]
for beta_loss in loss_betas:
    train(time_hor,beta_loss)

print("Finito")
