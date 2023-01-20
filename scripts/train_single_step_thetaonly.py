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
from pytorch_lightning.callbacks import TQDMProgressBar

import pyqg_explorer.dataset.forcing_dataset as forcing_dataset
import pyqg_explorer.models.base_model as base_model
import pyqg_explorer.util.pbar as pbar
import pyqg_explorer.util.transforms as transforms
import pyqg_explorer.util.powerspec as powerspec

from sklearn.metrics import r2_score

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

lev=0
forcing=1
seed=123
batch_size=64
input_channels=1
output_channels=1
activation="ReLU"
arch="cnn_theta"
epochs=200
subsample=None
normalise="proper"
beta_steps=int(sys.argv[1])
beta_loss=float(sys.argv[2])
save_path="/scratch/cp3759/pyqg_data/models"
save_name="cnn_theta_beta%d_betaloss%d_lev%d_epoch%d.pt" % (beta_steps,beta_loss,lev,epochs)
lr=0.0001

## Wandb config file
config={"lev":lev,
        "forcing":forcing,
        "seed":seed,
        "lr":lr,
        "batch_size":batch_size,
        "input_channels":input_channels,
        "output_channels":output_channels,
        "activation":activation,
        "save_name":save_name,
        "save_path":save_path,
        "arch":arch,
        "beta_steps":beta_steps,
        "beta_loss":beta_loss,
        "epochs":epochs,
        "subsample":subsample,
        "normalise":normalise}

print("Config:", config)

data_full=xr.open_dataset('/scratch/cp3759/pyqg_data/sims/%d_step/all.nc' % beta_steps)
data_dqbar=data_full.dqdt_through_lores.isel(lev=config["lev"])
data_dqbar=data_dqbar.stack(snapshot=("run","time"))
data_dqbar=data_dqbar.transpose("snapshot","y","x")
#data_dqbar.isel(snapshot=200).plot()
data_forcing=data_full.q_forcing_advection.isel(lev=lev)
data_forcing=data_forcing.stack(snapshot=("run","time"))
data_forcing=data_forcing.transpose("snapshot","y","x")
#data_forcing.isel(snapshot=20).plot()
data_q=data_full.q.isel(lev=lev)
data_q=data_q.stack(snapshot=("run","time"))
data_q=data_q.transpose("snapshot","y","x")
#data_q.isel(snapshot=20).plot()

del data_full

single_dataset=forcing_dataset.SingleStepDataset(data_q,data_dqbar,data_forcing,normalise=config["normalise"],subsample=config["subsample"])

del data_forcing
del data_dqbar
del data_q

config["q_mean"]=single_dataset.q_mean
config["q_std"]=single_dataset.q_std
config["s_mean"]=single_dataset.s_mean
config["s_std"]=single_dataset.s_std
config["training_fields"]=len(single_dataset.test_idx)
config["validation_fields"]=len(single_dataset.valid_idx)


train_loader = DataLoader(
    single_dataset,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(single_dataset.train_idx),
)
valid_loader = DataLoader(
    single_dataset,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(single_dataset.valid_idx),
)

model_theta=base_model.AndrewCNN(config)

model_theta.to(device)

wandb.init(project="pyqg_beta_cnns", entity="m2lines",config=config)
wandb.watch([model_theta,model_beta], log_freq=1)

# optimizer parameters
beta1 = 0.5
beta2 = 0.999
lr = config["lr"]
wd = 0.001

optimizer = torch.optim.AdamW(list(model_theta.parameters())), lr=lr, weight_decay=wd, betas=(beta1, beta2))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=10)

criterion=nn.MSELoss()

for epoch in range(config["epochs"]):  # loop over the dataset multiple times

    train_samples = 0.0
    train_running_loss = 0.0
    train_theta_running_loss = 0.0

    valid_running_loss = 0.0
    valid_theta_running_loss = 0.0
    valid_samples = 0.0
    
    model_theta.train()
    
    for i, data in enumerate(train_loader, 0):
        ## x_data is ordered in [pv, dqdt, s]
        x_data, y_data = data
        x_data=x_data.to(device)
        y_data=y_data.to(device)
    
        ## zero the parameter gradients
        optimizer.zero_grad()

        ## First network
        output_theta = model_theta(x_data[:,0,:,:].unsqueeze(1)) ## Takes in Q, outputs \hat{S}
        
        loss_1 = criterion(output_theta.squeeze(), x_data[:,2,:,:])
        loss = loss_1
        loss.backward()
        optimizer.step()
        
        
        ## Track loss for wandb
        train_running_loss+=loss.detach()
        train_theta_running_loss+=loss_1.detach()
        train_samples+=x_data.shape[0]
        
        #print(train_running_loss)
    
    model_theta.eval()
    for i, data in enumerate(valid_loader, 0):
        ## x_data is ordered in [pv, dqdt, s]
        x_data, y_data = data
        x_data=x_data.to(device)
        y_data=y_data.to(device)
    
        ## zero the parameter gradients
        optimizer.zero_grad()

        ## First network
        output_theta = model_theta(x_data[:,0,:,:].unsqueeze(1)) ## Takes in PV, outputs S
        
        val_loss_1 = criterion(output_theta.squeeze(), x_data[:,2,:,:])
        val_loss = val_loss_1
        ## Track loss for wandb
        valid_running_loss+=val_loss.detach()
        valid_samples+=x_data.shape[0]
        valid_theta_running_loss+=val_loss_1.detach()
    
    ## Push loss values for each epoch to wandb
    log_dic={}
    log_dic["epoch"]=epoch
    log_dic["training_loss"]=train_running_loss/train_samples
    log_dic["training_theta_loss"]=train_theta_running_loss/train_samples
    log_dic["valid_loss"]=valid_running_loss/valid_samples
    log_dic["valid_theta_loss"]=valid_theta_running_loss/valid_samples
    wandb.log(log_dic)
    
    # verbose
    print('%03d %.3e %.3e '%(log_dic["epoch"], log_dic["training_loss"], log_dic["valid_loss"]), end='')
    print("")


model_theta.save_model()

x_maps=torch.tensor(()).to("cpu")
y_true=torch.tensor(()).to("cpu")
y_pred=torch.tensor(()).to("cpu")

model_theta.eval()
model_theta.to("cpu")

for i, data in enumerate(valid_loader, 0):
    ## x_data is ordered in [pv, dqdt, s]
    x_data, y_data = data
    #print(x_data.device)
    #x_data=x_data.to(device)

    ## zero the parameter gradients
    optimizer.zero_grad()

    ## First network
    y_hat = model_theta(x_data[:,0,:,:].unsqueeze(1)) ## Takes in PV, outputs S
    
    x_data_cpu=x_data.to("cpu")
    y_hat_cpu=y_hat.to("cpu")
    
    del x_data
    del y_hat
    
    x_maps=torch.cat((x_maps,x_data_cpu[:,0,:,:].unsqueeze(1)),dim=0)
    y_true=torch.cat((y_true,x_data_cpu[:,2,:,:].unsqueeze(1)),dim=0)
    y_pred=torch.cat((y_pred,y_hat_cpu),dim=0)

## Convert validation metrics to numpy arrays
x_np=x_maps.squeeze().cpu().detach().numpy()
y_np=y_true.squeeze().cpu().detach().numpy()
y_pred_np=y_pred.squeeze().cpu().detach().numpy()

## Estimate R2
r2=r2_score(y_np.flatten(),y_pred_np.flatten())

## Get power spectrum from validation set
power_true=[]
power_pred=[]

for aa in range(len(x_np)):
    power_true.append(powerspec.get_power_spectrum(y_np[aa]))
    power_pred.append(powerspec.get_power_spectrum(y_pred_np[aa]))
    
power_true=np.mean(np.stack(power_true,axis=1),axis=1)
power_pred=np.mean(np.stack(power_pred,axis=1),axis=1)

## Plot power spectrum
f=plt.title(r"Power spectrum of subgrid forcing: $R^2=%.2f$" % r2 )
plt.loglog(power_true,label="True")
plt.loglog(power_pred,label="From CNN")
plt.legend()
figure_power=wandb.Image(f)
wandb.log({"Power spectrum": figure_power})

map_index=2

fig, axs = plt.subplots(1, 4,figsize=(15,3))
ax=axs[0].imshow(x_np[map_index], cmap='bwr')
fig.colorbar(ax, ax=axs[0])
axs[0].set_xticks([]); axs[0].set_yticks([])
axs[0].set_title("PV field")

ax=axs[1].imshow(y_np[map_index], cmap='bwr', interpolation='none')
fig.colorbar(ax, ax=axs[1])
axs[1].set_xticks([]); axs[1].set_yticks([])
axs[1].set_title("True forcing")

ax=axs[2].imshow(y_pred_np[map_index], cmap='bwr', interpolation='none')
fig.colorbar(ax, ax=axs[2])
axs[2].set_xticks([]); axs[2].set_yticks([])
axs[2].set_title("Forcing from CNN")

ax=axs[3].imshow(y_np[map_index]-y_pred_np[map_index], cmap='bwr', interpolation='none')
fig.colorbar(ax, ax=axs[3])
axs[3].set_xticks([]); axs[3].set_yticks([])
axs[3].set_title("True forcing-CNN forcing")
fig.tight_layout()

figure_fields=wandb.Image(fig)
wandb.log({"Random fields": figure_fields})

wandb.run.summary["r2_score"]=r2
wandb.finish()