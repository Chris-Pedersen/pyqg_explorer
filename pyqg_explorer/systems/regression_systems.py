import pickle
import os
import functools
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
#from pytorch_lightning.loops import FitLoop
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
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.2)
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
        loss_beta = self.config["beta_loss"]*self.criterion(output_beta, y_data-x_data[:,0:2,:,:])
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


class RolloutTorch(BaseRegSytem):
    """ Train an emulator to predict the state of the field over some time horizon """
    def __init__(self,network,config:dict):
        super().__init__(network,config)
        
    def step(self,batch,kind):
        """ Evaluate loss function """
        x_data = batch
        
        loss=0
        
        for aa in range(0,x_data.shape[1]-1):
            if aa==0:
                x_pred=x_data[:,0,:,:,:]
            else:
                x_pred=self(x_pred)
            loss_dt=self.criterion(x_pred,x_data[:,aa+1,:,:,:])*np.exp(-aa*self.config["decay_coeff"])
            self.log(f"{kind}_loss_%d" % aa, loss_dt, on_step=False, on_epoch=True)
            loss+=loss_dt
            
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True) 
        return loss


class ResidualRolloutTorch(BaseRegSytem):
    """ Train an emulator to predict residuals over some time horizon. No subgrid forcing channel
        here - only emulating the state residuals """
    def __init__(self,network,config:dict):
        super().__init__(network,config)
        
    def step(self,batch,kind):
        """ Evaluate loss function """
        x_data = batch
        
        loss=0
        loss_state=0
        
        for aa in range(0,x_data.shape[1]-1):
            if aa==0:
                x_t=x_data[:,0,:,:,:]
            else:
                x_t=x_dt+x_t
            x_dt=self(x_t)
            loss_dt=self.criterion(x_dt,x_data[:,aa+1,:,:,:]-x_data[:,aa,:,:,:])*np.exp(-aa*self.config["decay_coeff"])
            ## Calculate state loss such that we can compare to the state-predicting emulator
            with torch.no_grad():
                loss_dt_state=self.criterion(x_dt+x_t,x_data[:,aa+1,:,:,:])*np.exp(-aa*self.config["decay_coeff"])
                loss_state+=loss_dt_state
            self.log(f"{kind}_loss_resid_%d" % aa, loss_dt, on_step=False, on_epoch=True)
            self.log(f"{kind}_loss_%d" % aa, loss_dt_state, on_step=False, on_epoch=True)
            loss+=loss_dt
            
        self.log(f"{kind}_loss_resid", loss, on_step=False, on_epoch=True)
        self.log(f"{kind}_loss", loss_state, on_step=False, on_epoch=True) 
        return loss


class ResidualRollout(BaseRegSytem):
    """ Train an emulator to predict residuals over some time horizon. Can either train to use
        just the resolved field, or resolved field + subgrid forcing field as input channels """
    def __init__(self,network,config:dict):
        super().__init__(network,config)
        
    def step_noforce(self,batch,kind):
        """ Evaluate loss function """
        x_data = batch
        
        loss=0
        
        for aa in range(0,x_data.shape[2]-1):
            if aa==0:
                x_t=x_data[:,:,0,:,:]
            else:
                x_t=x_dt+x_t
            x_dt=self(x_t)
            loss_dt=self.criterion(x_dt,x_data[:,:,aa+1,:,:]-x_data[:,:,aa,:,:])*np.exp(-aa*self.config["decay_coeff"])
            self.log(f"{kind}_loss_%d" % aa, loss_dt, on_step=False, on_epoch=True)
            loss+=loss_dt
            
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True) 
        return loss
    
    def step_force(self,batch,kind):
        """ Evaluate loss function """
        x_data,s_data = batch
        
        loss=0
        
        for aa in range(0,x_data.shape[2]-1):
            if aa==0:
                x_t=torch.cat((x_data[:,:,0,:,:],s_data[:,:,0,:,:]),1)
            else:
                x_t=x_dt+x_t[:,0:2,:,:]
                x_t=torch.cat((x_t,s_data[:,:,aa,:,:]),1)
            #print(x_t.shape)
            x_dt=self(x_t)
            loss_dt=self.criterion(x_dt,x_data[:,:,aa+1,:,:]-x_data[:,:,aa,:,:])*np.exp(-aa*self.config["decay_coeff"])
            self.log(f"{kind}_loss_%d" % aa, loss_dt, on_step=False, on_epoch=True)
            loss+=loss_dt
            
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True) 
        return loss

    def step_force4(self,batch,kind):
        """ Evaluate loss function for emulator that outputs both q_i+dt and s. We emulate the residuals of both of these quantities
            over some given time horizon """
        x_data,s_data = batch
        all_data=torch.cat((x_data,s_data),axis=1)
        
        loss=0
        
        for aa in range(0,x_data.shape[2]-1):
            if aa==0:
                x_t=all_data[:,:,0,:,:]
            else:
                x_t=x_dt+x_t
            x_dt=self(x_t)
            loss_dt=self.criterion(x_dt,all_data[:,:,aa+1,:,:]-all_data[:,:,aa,:,:])*np.exp(-aa*self.config["decay_coeff"])
            loss_q=self.criterion(x_dt[:,0:2],all_data[:,0:2,aa+1,:,:]-all_data[:,0:2,aa,:,:])
            loss_s=self.criterion(x_dt[:,2:4],all_data[:,2:4,aa+1,:,:]-all_data[:,2:4,aa,:,:])
            self.log(f"{kind}_loss_%d" % aa, loss_dt, on_step=False, on_epoch=True)
            self.log(f"{kind}_q_loss_%d" % aa, loss_q, on_step=False, on_epoch=True)
            self.log(f"{kind}_s_loss_%d" % aa, loss_s, on_step=False, on_epoch=True)
            loss+=loss_dt
            
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True) 
        return loss
    
    def step(self,batch,kind):
        """ Evaluate loss function """
        if self.config["subgrid_forcing"]==True:
            if self.network.config["output_channels"]==4:
                loss=self.step_force4(batch,kind)
            else:
                loss=self.step_force(batch,kind)
        else:
            loss=self.step_noforce(batch,kind)
        return loss


class JointRolloutO(BaseRegSytem):
    """ Regression system to train a jointly optimised subgrid model, over multiple time rollouts """
    def __init__(self,network,config:dict,network_beta):
        super().__init__(network,config)
        self.network_beta=network_beta
        
        assert self.network_beta.config["time_horizon"]==self.config["time_horizon"], "Different time horizons for dataset and beta network"

    def step(self,batch,kind):
        """ Evaluate loss function """
        x_data,y_data = batch
        
        loss=0
        
        for aa in range(0,x_data.shape[2]-1):
            if aa==0:
                x_t=x_data[:,:,0,:,:]
            else:
                x_t=x_dt+x_t
            output_theta = self(x_t)
            x_dt=self.network_beta(torch.cat((x_t,output_theta),1))
            loss_theta_dt=self.config["theta_loss"]*self.criterion(output_theta,y_data[:,:,aa,:,:])*np.exp(-aa*self.config["decay_coeff"])
            loss_beta_dt=self.config["beta_loss"]*self.criterion(x_dt,x_data[:,:,aa+1,:,:]-x_data[:,:,aa,:,:])*np.exp(-aa*self.config["decay_coeff"])
            loss_dt=loss_theta_dt+loss_beta_dt
            loss+=loss_dt
            self.log(f"{kind}_theta_loss_%d" % aa, loss_theta_dt, on_step=False, on_epoch=True)
            self.log(f"{kind}_beta_loss_%d" % aa, loss_beta_dt, on_step=False, on_epoch=True)
            self.log(f"{kind}_loss_%d", loss_dt, on_step=False, on_epoch=True)
            
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True) 
        return loss


class JointRollout(BaseRegSytem):
    """ Regression system to train a jointly optimised subgrid model, over multiple time rollouts """
    def __init__(self,network,config:dict,network_beta):
        super().__init__(network,config)
        self.network_beta=network_beta
        
        assert self.network_beta.config["time_horizon"]==self.config["time_horizon"], "Different time horizons for dataset and beta network"

    def step(self,batch,kind):
        """ Evaluate loss function """
        x_data,y_data = batch
        
        loss=0
        
        output_theta = self(x_data[:,:,0,:,:])

        for aa in range(0,x_data.shape[2]-1):
        ## Only evaluate offline loss at i timestep.
            if aa==0:
                x_t=x_data[:,:,0,:,:]
            else:
                x_t=x_dt+x_t
            output_theta = self(x_t)
            x_dt=self.network_beta(torch.cat((x_t,output_theta),1))
            loss_theta_dt=self.config["theta_loss"]*self.criterion(output_theta,y_data[:,:,aa,:,:])*np.exp(-aa*self.config["decay_coeff"])
            loss_beta_dt=self.config["beta_loss"]*self.criterion(x_dt,x_data[:,:,aa+1,:,:]-x_data[:,:,aa,:,:])*np.exp(-aa*self.config["decay_coeff"])
            loss_dt=loss_theta_dt+loss_beta_dt
            loss+=loss_dt
            self.log(f"{kind}_theta_loss_%d" % aa, loss_theta_dt, on_step=False, on_epoch=True)
            self.log(f"{kind}_beta_loss_%d" % aa, loss_beta_dt, on_step=False, on_epoch=True)
            self.log(f"{kind}_loss_%d", loss_dt, on_step=False, on_epoch=True)
            
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True) 
        return loss
