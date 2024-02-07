import wandb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.models.diffusion as diffusion
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.models.unet as unet
import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import matplotlib.pyplot as plt

from utils import ExponentialMovingAverage
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

config=reg_sys.config


emulator_dataset=forcing_dataset.OfflineDataset("/scratch/cp3759/pyqg_data/sims/torchqg_sims/0_step/all.nc",seed=config["seed"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])

## Need to save renormalisation factors for when the CNN is plugged into pyqg
config["q_mean_upper"]=emulator_dataset.q_mean_upper
config["q_mean_lower"]=emulator_dataset.q_mean_lower
config["q_std_upper"]=emulator_dataset.q_std_upper
config["q_std_lower"]=emulator_dataset.q_std_lower
config["s_mean_upper"]=emulator_dataset.s_mean_upper
config["s_mean_lower"]=emulator_dataset.s_mean_lower
config["s_std_upper"]=emulator_dataset.s_std_upper
config["s_std_lower"]=emulator_dataset.s_std_lower

train_loader = DataLoader(
    emulator_dataset,
    num_workers=10,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(emulator_dataset.train_idx),
)

config["lr"]=0.001
config["batch_size"]=128
config["epochs"]=100
config["n_samples"]=12
config["timesteps"]=1000
config["model_ema_steps"]=0
config["model_ema_decay"]=0.995
config["log_freq"]=10
config["no_clip"]=True # set to normal sampling method without clip x_0 which could yield unstable samples
config["dim_mults"]=[2,4]
config["time_embedding_dim"]=256
config["input_channels"]=2
config["output_channels"]=2
config["base_dim"]=32
config["image_size"]=64
config["save_name"]=["model_weights.pt"]


device="cuda"
config["train_set_size"]=len(train_loader.dataset)

model_cnn=unet.Unet(config)
model=diffusion.Diffusion(config, model=model_cnn).to(device)

wandb.init(project="qg_diffusion",entity="chris-pedersen",config=config)
wandb.watch(model, log_freq=1)

config["save_path"]=wandb.run.dir
config["wandb_url"]=wandb.run.get_url()


#torchvision ema setting
#https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
#adjust = 1* config["batch_size"] * config["model_ema_steps"] / config["epochs"]
#alpha = 1.0 - config["model_ema_decay"]
#alpha = min(1.0, alpha * adjust)
#model_ema = diffusion.ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

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
    wandb.log(log_dic)

    #model_ema.eval()
    #samples=model_ema.module.sampling(config["n_samples"],clipped_reverse_diffusion=not config["no_clip"],device=device)
    
    model.eval()
    samples=model.sampling(config["n_samples"],clipped_reverse_diffusion=not config["no_clip"],device=device)
    
    ## Upload figure of generated samples
    fig_samples=plt.figure(figsize=(16,11))
    plt.suptitle("epoch=%d" % i)
    for aa in range(12):
        plt.subplot(3,4,aa+1)
        plt.imshow(samples[aa,0].cpu(),cmap="Purples")
        plt.colorbar()
    plt.tight_layout()
    wandb.log({"Samples upper":fig_samples})
    
    ## Upload figure of generated samples
    fig_samples=plt.figure(figsize=(16,11))
    plt.suptitle("epoch=%d" % i)
    for aa in range(12):
        plt.subplot(3,4,aa+1)
        plt.imshow(samples[aa,1].cpu(),cmap="Purples")
        plt.colorbar()
    plt.tight_layout()
    wandb.log({"Samples lower":fig_samples})
    #save_image(samples,"results/steps_{:0>8}.png".format(global_steps),nrow=int(math.sqrt(args.n_samples)))


model.model.save_model()
wandb.finish()

print("Finito")