import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
import cmocean

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.sampler import SubsetRandomSampler

import pyqg_explorer.performance.diffusion_performance as diff_performance
import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.models.diffusion as diffusion
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.models.unet as unet
import pyqg_explorer.dataset.forcing_dataset as forcing_dataset

parser = argparse.ArgumentParser(description="Diffusion model denoiser")
parser.add_argument("--base-dim", required=True, type=float, help="base dim of Unet")
parser.add_argument("--denoise-time",required=True,type=float,default=200,help="denoise level")
parser.add_argument("--subsample",type=int,help="subsample training data")
parser.add_argument("--jet",action=argparse.BooleanOptionalAction,
                default=False,help="Add this flag to run on jet config")
args = parser.parse_args()
print(args)

config=reg_sys.config
## Stuff we are varying
config["subsample"]=int(args.subsample)
config["base_dim"]=int(args.base_dim)
config["denoise_time"]=int(args.denoise_time)
config["eddy"]=not args.jet

## Stuff we are not varying (for now)
config["noise_sampling_coeff"]=0.35
config["dim_mults"]=[2,4]
config["lr"]=0.001
config["batch_size"]=128
config["epochs"]=100
config["timesteps"]=1000
config["model_ema_steps"]=0
config["model_ema_decay"]=0.995
config["time_embedding_dim"]=256
config["input_channels"]=2
config["output_channels"]=2
config["image_size"]=64
config["save_name"]="model_weights.pt"
config["valid_batch_size"]=20
print(config)

if config["eddy"]:
    sim_string="eddy"
else:
    sim_string="jet"

snapshot_dataset=forcing_dataset.OfflineDataset("/scratch/cp3759/pyqg_data/sims/torchqg_sims/0_step/all_%s.nc" % sim_string,
                            seed=config["seed"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])
print(sim_string)

## Save renormalisation factors for mapping NN predictions to and from physical units
config["q_mean_upper"]=snapshot_dataset.q_mean_upper
config["q_mean_lower"]=snapshot_dataset.q_mean_lower
config["q_std_upper"]=snapshot_dataset.q_std_upper
config["q_std_lower"]=snapshot_dataset.q_std_lower
config["s_mean_upper"]=snapshot_dataset.s_mean_upper
config["s_mean_lower"]=snapshot_dataset.s_mean_lower
config["s_std_upper"]=snapshot_dataset.s_std_upper
config["s_std_lower"]=snapshot_dataset.s_std_lower

train_loader = DataLoader(
    snapshot_dataset,
    num_workers=10,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(snapshot_dataset.train_idx),
    drop_last=True
)

valid_loader = DataLoader(
    snapshot_dataset,
    num_workers=10,
    batch_size=config["valid_batch_size"],
    sampler=SubsetRandomSampler(snapshot_dataset.valid_idx),
    drop_last=True
)

device="cuda"

wandb.init(project="denoise_qg",entity="m2lines",config=config,dir="/scratch/cp3759/pyqg_data/wandb_runs")

## Make sure save path, train set size, wandb url are passed to config before model is initialised!
## otherwise these important things aren't part of the model config property
config["save_path"]=wandb.run.dir
config["wandb_url"]=wandb.run.get_url()
config["train_set_size"]=len(train_loader.dataset)

model_cnn=unet.Unet(config)
model=diffusion.Diffusion(config, model=model_cnn).to(device)
config["num_params"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
wandb.config.update(config)
wandb.watch(model, log_freq=1)

optimizer=AdamW(model.parameters(),lr=config["lr"])
scheduler=OneCycleLR(optimizer,config["lr"],total_steps=config["epochs"]*len(train_loader),pct_start=0.25,anneal_strategy='cos')
loss_fn=nn.MSELoss(reduction='mean')

global_steps=0
for i in range(1,config["epochs"]+1):
    train_running_loss = 0.0
    model.train()
    ## Loop over batches
    for j,(image,target) in enumerate(train_loader):
        noise=torch.randn_like(image).to(device)
        image=image.to(device)
        pred=model(image,noise)
        loss=loss_fn(pred,noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        global_steps+=1
    
    ## Push loss values for each epoch to wandb
    log_dic={}
    log_dic["epoch"]=i
    log_dic["training_loss"]=(loss.detach().cpu().item()/(len(image)))
    
    model.eval()
    ## From a validation set, add noise to a set of heldout images
    ## Then denoise them, and run some quality metrics on the reconstructions
    valid_imgs=next(iter(valid_loader))
    valid_imgs=valid_imgs[0].to(device)
    noise=torch.randn_like(valid_imgs).to(device)
    t=(torch.ones(len(valid_imgs),dtype=torch.int64)*config["denoise_time"]).to(device)
    noised=model._forward_diffusion(valid_imgs,t,noise)
    denoised=model.denoising(noised,config["denoise_time"])
    
    ## Get losses of noise level and denoised images
    noise_loss=loss_fn(valid_imgs,noised)
    valid_denoise_loss=loss_fn(valid_imgs,denoised)
    
    fig_denoise=diff_performance.plot_fields(valid_imgs,noised,denoised,noise_loss,valid_denoise_loss,i)
    denoise_fig=wandb.Image(fig_denoise)
    wandb.log({"Denoise":denoise_fig})
    
    ## Pass clean and denoised fields to simulation objects, such that we can
    ## calculate diagnostics
    clean_sims,denoised_sims=diff_performance.field_to_sims(valid_imgs,denoised,config)
    fig_spectra=diff_performance.spectral_diagnostics(clean_sims,denoised_sims,i,config["eddy"])
    spectra_fig=wandb.Image(fig_spectra)
    wandb.log({"Spectra":spectra_fig})
    
    log_dic["denoise_loss"]=(valid_denoise_loss.detach().cpu().item()/(len(valid_imgs)))
    log_dic["noise_loss"]=(noise_loss.detach().cpu().item()/(len(valid_imgs)))
    wandb.log(log_dic)
    plt.close()

model.model.save_model()
wandb.finish()
print("done")
