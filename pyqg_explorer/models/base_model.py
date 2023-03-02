import pickle
import os
import functools
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.loops import FitLoop

import pyqg_explorer.util.transforms as transforms


class BaseModel(LightningModule):
    """ Class to store core model methods """
    def __init__(self,config:dict,model_beta=None):
        super().__init__()
        self.config=config
        self.criterion=nn.MSELoss()
        self.model_beta=model_beta ## CNN that predicts the system at some future time

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.config["lr"],weight_decay=self.config["wd"])
        return optimizer

    def step(self,batch,kind):
        """ If there is no additional beta network, just perform a standard optimisation """
        x_data, y_data = batch
        output_theta = self(x_data) ## Takes in Q, outputs \hat{S}
        loss = self.criterion(output_theta, y_data)
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True)       
        return loss
    
    def joint_step(self,batch,kind):
        """ If we also have a beta network, run joint optimisation """
        x_data, y_data = batch
        output_theta = self(x_data[:,0:2,:,:]) ## Takes in Q, outputs \hat{S}
        output_beta = self.model_beta(torch.cat((x_data[:,0:4,:,:],output_theta),1))
        loss_theta = self.criterion(output_theta, x_data[:,4:6,:,:])
        loss_beta = self.config["beta_loss"]*self.criterion(output_beta, y_data)
        loss = loss_theta+loss_beta
        self.log(f"{kind}_theta_loss", loss_theta, on_step=False, on_epoch=True)
        self.log(f"{kind}_beta_loss", loss_beta, on_step=False, on_epoch=True)
        self.log(f"{kind}_total_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        if self.model_beta is not None:
            self.model_beta.train()
            return self.joint_step(batch,"train")
        else:
            return self.step(batch,"train")

    def validation_step(self, batch, batch_idx):
        if self.model_beta is not None:
            self.model_beta.eval()
            return self.joint_step(batch,"valid")
        else:
            return self.step(batch,"valid")
    
    def save_model(self):
        """ Save the model config, and optimised weights and biases. We create a dictionary
        to hold these two sub-dictionaries, and save it as a pickle file """
        save_dict={}
        save_dict["state_dict"]=self.state_dict() ## Dict containing optimised weights and biases
        save_dict["config"]=self.config           ## Dict containing config for the dataset and model
        save_string=os.path.join(self.config["save_path"],self.config["save_name"])
        with open(save_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved as %s" % save_string)
