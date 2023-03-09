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
config={"lev":"both",
        "seed":123,
        "drop_spin_up":True,
        "lr":0.001,
        "wd":0.01,
        "batch_size":64,
        "input_channels":2,
        "output_channels":2,
        "activation":"ReLU",
        "save_name":None,
        "save_path":None,
        "arch":None,
        "conv_layers":12,
        "beta_steps":None,
        "epochs":200,
        "subsample":None}


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
