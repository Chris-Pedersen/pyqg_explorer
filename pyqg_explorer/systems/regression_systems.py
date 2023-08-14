import pickle
import os
import functools
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.loops import FitLoop
import pyqg_explorer.util.transforms as transforms


## Default config
config={## Dastaset config
        "seed":123,
        "subsample":None,
        "drop_spin_up":True,
        ## Training hyperparams
        "lr":0.001,
        "wd":0.05,
        "dropout":0.05,
        "batch_size":64,
        "epochs":200,
        "scheduler":True,
        ## Model config
        "input_channels":2,
        "output_channels":2,
        "activation":"ReLU",
        "save_name":"model_weights.pt",
        "conv_layers":5
        }


class BaseRegSytem(LightningModule):
    """ Base class to implement common methods. We leave the definition of the step method to child classes """
    def __init__(self,network,config:dict):
        super().__init__()
        self.config=config
        self.criterion=nn.MSELoss()
        self.network=network

    def forward(self,x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.config["lr"],weight_decay=self.config["wd"])
        if self.config["scheduler"]:
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
        else:
            return {"optimizer": optimizer}
        

    def step(self,batch,kind):
        raise NotImplementedError("To be defined by child class")

    def training_step(self, batch, batch_idx):
        return self.step(batch,"train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch,"valid")


class RegressionSystem(BaseRegSytem):
    """ Standard regression system - one model, one loss """
    def __init__(self,network,config:dict):
        super().__init__(network,config)

    def step(self,batch,kind):
        """ Evaluate loss function """
        x_data, y_data = batch
        output_theta = self(x_data) ## Takes in Q, outputs \hat{S}
        loss = self.criterion(output_theta, y_data)
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True)       
        return loss


class JointRegressionSystem(BaseRegSytem):
    """ Joint optimisation system. The `network` argument will be considered the offline forcing model,
        the `beta_network` argument is the forward-stepping network """
    def __init__(self,network,config:dict,network_beta):
        super().__init__(network,config)
        self.network_beta=network_beta

    def step(self,batch,kind):
        """ If we also have a beta network, run joint optimisation """
        x_data, y_data = batch
        output_theta = self(x_data[:,0:2,:,:]) ## Takes in Q, outputs \hat{S}
        output_beta = self.network_beta(torch.cat((x_data[:,0:2,:,:],output_theta),1))
        loss_theta = self.config["theta_loss"]*self.criterion(output_theta, x_data[:,2:4,:,:])
        loss_beta = self.config["beta_loss"]*self.criterion(output_beta, y_data)
        loss = loss_theta+loss_beta
        self.log(f"{kind}_theta_loss", loss_theta, on_step=False, on_epoch=True)
        self.log(f"{kind}_beta_loss", loss_beta, on_step=False, on_epoch=True)
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True)
        return loss


class ResidualRegressionSystem(BaseRegSytem):
    """ Define loss with respect to the residuals. Expect y_data *not* to be
    a residual value, but calculate residuals in the loss """
    def __init__(self,network,config:dict):
        super().__init__(network,config)

    def step(self,batch,kind):
        """ Evaluate loss function """
        x_data, y_data = batch
        output_theta = self(x_data) 
        loss = self.criterion(output_theta, y_data-x_data[:,0:2,:,:])
        loss_norm = self.criterion(output_theta+x_data[:,0:2,:,:], y_data)
        self.log(f"{kind}_loss", loss_norm, on_step=False, on_epoch=True)
        self.log(f"{kind}_target_loss", loss, on_step=False, on_epoch=True) 
        return loss


class ResidualRegressionSystemStd(BaseRegSytem):
    """ Expect y_data to be a residual value, independently normalised """
    def __init__(self,network,config:dict):
        super().__init__(network,config)

    def step(self,batch,kind):
        """ Loss here is defined with respect to the residuals """

        x_data, y_data = batch
        output_theta = self(x_data)
        loss = self.criterion(output_theta, y_data)

        def map_residual_to_q(field):
            up=field[:,0,:,:]
            low=field[:,1,:,:]

            ## Transform from residual space to physical space
            up_phys=transforms.denormalise_field(up,self.config["res_mean_upper"],self.config["res_std_upper"])+transforms.denormalise_field(x_data[:,0,:,:],self.config["q_mean_upper"],self.config["q_std_upper"])
            up_norm=transforms.normalise_field(up_phys,self.config["q_mean_upper"],self.config["q_std_upper"])
            ## Transform from residual space to physical space
            low_phys=transforms.denormalise_field(low,self.config["res_mean_lower"],self.config["res_std_lower"])+transforms.denormalise_field(x_data[:,1,:,:],self.config["q_mean_lower"],self.config["q_std_lower"])
            low_norm=transforms.normalise_field(low_phys,self.config["q_mean_lower"],self.config["q_std_lower"])

            return torch.cat((up_norm.unsqueeze(1),low_norm.unsqueeze(1)),1)

        ## Take prediction, map to physical space, add original field
        norm_true=map_residual_to_q(output_theta)
        norm_pred=map_residual_to_q(y_data)

        normal_loss=self.criterion(norm_true,norm_pred)

        ## Map this loss to a normalised loss 
        self.log(f"{kind}_residual_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{kind}_loss", normal_loss, on_step=False, on_epoch=True) 
        return loss


class ResidualRollout(BaseRegSytem):
    """ Define loss with respect to the residuals. Expect y_data *not* to be
    a residual value, but calculate residuals in the loss """
    def __init__(self,network,config:dict):
        super().__init__(network,config)

    def step(self,batch,kind):
        """ Evaluate loss function """
        x_data = batch
        
        loss=0
        
        for aa in range(0,x_data.shape[2]-1):
            if aa==0:
                x_t=x_data[:,:,0,:,:]
            else:
                x_t=x_dt+x_t
            x_dt=self(x_t)
            loss_dt=self.criterion(x_dt,x_data[:,:,aa+1,:,:]-x_data[:,:,aa,:,:])
            self.log(f"{kind}_loss_%d" % aa, loss_dt, on_step=False, on_epoch=True)
            loss+=loss_dt
            
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True) 
        return loss
