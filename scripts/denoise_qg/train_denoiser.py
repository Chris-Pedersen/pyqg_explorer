import argparse
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.models.diffusion as diffusion
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.models.unet as unet
import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import matplotlib.pyplot as plt
import cmocean

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import math
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser(description="Diffusion model denoiser")
parser.add_argument(
    "--base-dim", required=True, type=float, help="base dim of Unet"
)
parser.add_argument(
    "--denoise-time",
    required=True,
    type=float,
    default=200,
    help="denoise level",
)
parser.add_argument("--subsample",type=int,
    help="subsample training data",
)
args = parser.parse_args()

config=reg_sys.config
## Stuff we are varying
config["subsample"]=int(args.subsample)
config["base_dim"]=int(args.base_dim)
config["dim_mults"]=[2,4]
config["noise_sampling_coeff"]=0.35
config["denoise_time"]=int(args.denoise_time)

## Stuff we are not varying (for now)
config["lr"]=0.001
config["batch_size"]=128
config["epochs"]=100
config["timesteps"]=1000
config["model_ema_steps"]=0
config["model_ema_decay"]=0.995
config["log_freq"]=50
config["no_clip"]=True # set to normal sampling method without clip x_0 which could yield unstable samples
config["time_embedding_dim"]=256
config["input_channels"]=2
config["output_channels"]=2
config["image_size"]=64
config["save_name"]="model_weights.pt"

snapshot_dataset=forcing_dataset.OfflineDataset("/scratch/cp3759/pyqg_data/sims/torchqg_sims/0_step/all_jet.nc",seed=config["seed"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])

## Need to save renormalisation factors for when the CNN is plugged into pyqg
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
    batch_size=20,
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
        #if global_steps%config["model_ema_steps"]==0:
        #    model_ema.update_parameters(model)
        global_steps+=1
        if j%config["log_freq"]==0:
            log_dic={}
            log_dic["training_loss"]=(loss.detach().cpu().item()/(len(image)))
            log_dic["lr"]=scheduler.get_last_lr()[0]
            wandb.log(log_dic)
            #print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,epochs,j,len(train_loader),
            #                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
    
    ## Push loss values for each epoch to wandb
    log_dic={}
    log_dic["epoch"]=i
    log_dic["training_loss"]=(loss.detach().cpu().item()/(len(image)))
    
    
    model.eval()
    
    valid_imgs=next(iter(valid_loader))
    valid_imgs=valid_imgs[0].to(device)
    noise=torch.randn_like(valid_imgs).to(device)
    t=(torch.ones(len(valid_imgs),dtype=torch.int64)*config["denoise_time"]).to(device)
    noised=model._forward_diffusion(valid_imgs,t,noise)
    denoised=model.denoising(noised,config["denoise_time"])
    
    plt.figure(figsize=(8,4))
    plt.suptitle("Denoised fields at epoch=%d" % i)
    
    noise_loss=loss_fn(valid_imgs,noised)
    valid_denoise_loss=loss_fn(valid_imgs,denoised)

    valid_idx=np.random.randint(len(valid_imgs))
    fig_denoise=plt.subplot(2,3,1)
    plt.title("Original field")
    plt.imshow(valid_imgs[valid_idx][0].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(valid_imgs[valid_idx][1].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()

    plt.subplot(2,3,2)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("Noised field: %.2f" % noise_loss)
    plt.imshow(noised[valid_idx][0].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(noised[valid_idx][1].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("Denoised field: %.2f" % valid_denoise_loss)
    plt.imshow(denoised[valid_idx][0].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(denoised[valid_idx][1].cpu().numpy(),cmap=cmocean.cm.balance)
    plt.colorbar()

    plt.tight_layout()
    wandb.log({"Denoise":fig_denoise})
    plt.close()
    
    log_dic["denoise_loss"]=(valid_denoise_loss.detach().cpu().item()/(len(valid_imgs)))
    log_dic["noise_loss"]=(noise_loss.detach().cpu().item()/(len(valid_imgs)))
    wandb.log(log_dic)

model.model.save_model()
wandb.finish()
print("done")
