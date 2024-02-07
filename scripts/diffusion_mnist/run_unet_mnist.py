import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import os
import math
import argparse
import os, hashlib, requests, gzip
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import wandb
from torchvision.datasets import MNIST
from torchvision import transforms 
import matplotlib.pyplot as plt

import pyqg_explorer.models.diffusion as diffusion
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.models.unet as unet


def create_mnist_dataloaders(batch_size,image_size=28,num_workers=8):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

config={}

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
config["input_channels"]=1
config["output_channels"]=1
config["base_dim"]=32
config["image_size"]=28
config["save_name"]=["model_weights.pt"]

train_loader,test_loader=create_mnist_dataloaders(batch_size=config["batch_size"],image_size=28)
device="cuda"
config["train_set_size"]=len(train_loader.dataset)

model_cnn=unet.Unet(config)
model=diffusion.Diffusion(config, model=model_cnn).to(device)

wandb.init(project="mnist_diffusion",entity="chris-pedersen",config=config)
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
    wandb.log({"Samples":fig_samples})
    #save_image(samples,"results/steps_{:0>8}.png".format(global_steps),nrow=int(math.sqrt(args.n_samples)))


model.model.save_model()
wandb.finish()

print("Finito")