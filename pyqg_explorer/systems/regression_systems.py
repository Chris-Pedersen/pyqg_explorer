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
        "save_name":None,
        "save_path":None,
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
            return {"optimizer": optimizer, "lr_scheduler": scheduler,"monitor": "train_loss"}
        else:
            return {"optimizer": optimizer}
        

    def step(self,batch,kind):
        raise NotImplementedError("To be defined by child class")

    def training_step(self, batch, batch_idx):
        return self.step(batch,"train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch,"valid")
    
    def save_model(self):
        """ Save the model config, and optimised weights and biases. We create a dictionary
        to hold these two sub-dictionaries, and save it as a pickle file """
        if self.config["save_path"] is None:
            print("No save path provided, not saving")
            return
        save_dict={}
        save_dict["state_dict"]=self.state_dict() ## Dict containing optimised weights and biases
        save_dict["config"]=self.config           ## Dict containing config for the dataset and model
        save_string=os.path.join(self.config["save_path"],self.config["save_name"])
        with open(save_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved as %s" % save_string)
        return

    def pred(self, x):
        """ Method to call when receiving un-normalised data, when implemented as a pyqg
            parameterisation. Expects a 3D numpy array """

        x=torch.tensor(x).float()
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = transforms.normalise_field(x[0],self.config["q_mean_upper"],self.config["q_std_upper"])
        x_lower = transforms.normalise_field(x[1],self.config["q_mean_lower"],self.config["q_std_lower"])
        x = torch.stack((x_upper,x_lower),dim=0).unsqueeze(0)

        ## Pass the normalised fields through our network
        x = self(x)

        ## Map back from normalised space to physical units
        s_upper=transforms.denormalise_field(x[:,0,:,:],self.config["s_mean_upper"],self.config["s_std_upper"])
        s_lower=transforms.denormalise_field(x[:,1,:,:],self.config["s_mean_lower"],self.config["s_std_lower"])

        ## Reshape to match pyqg dimensions, and cast to numpy array
        s=torch.cat((s_upper,s_lower)).detach().numpy().astype(np.double)
        return s


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
        output_beta = self.network_beta(torch.cat((x_data[:,0:4,:,:],output_theta),1))
        loss_theta = self.criterion(output_theta, x_data[:,4:6,:,:])
        loss_beta = self.config["beta_loss"]*self.criterion(output_beta, y_data)
        loss = loss_theta+loss_beta
        self.log(f"{kind}_theta_loss", loss_theta, on_step=False, on_epoch=True)
        self.log(f"{kind}_beta_loss", loss_beta, on_step=False, on_epoch=True)
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True)
        return loss


class ResidualRegressionSystem(BaseRegSytem):
    def __init__(self,network,config:dict):
        super().__init__(network,config)

    def step(self,batch,kind):
        """ Loss here is defined with respect to the residuals """

        x_data, y_data = batch
        output_theta = self(x_data) ## Takes in Q, outputs \hat{S}
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
