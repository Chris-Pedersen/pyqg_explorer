import torch.nn as nn
import os
import pickle

def process_block(latent_channels):
    """ Processor block, with dilated CNNs """
    process_block=nn.Sequential(nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=1),nn.BatchNorm2d(latent_channels),nn.ReLU(),
                                nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=2),nn.BatchNorm2d(latent_channels),nn.ReLU(),
                                nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=4),nn.BatchNorm2d(latent_channels),nn.ReLU(),
                                nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=8),nn.BatchNorm2d(latent_channels),nn.ReLU(),
                                nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=4),nn.BatchNorm2d(latent_channels),nn.ReLU(),
                                nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=2),nn.BatchNorm2d(latent_channels),nn.ReLU(),
                                nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=1),nn.BatchNorm2d(latent_channels),nn.ReLU())            
    return process_block
    


class DRN(nn.Module):
    """ My implementation of Dilated Res-Net, using the encode-process-decode paradigm from https://arxiv.org/abs/2112.15275 """
    def __init__(self,config):
        super(DRN, self).__init__()
        self.config=config
        self.input_channels=config["input_channels"]
        self.output_channels=config["output_channels"]
        self.latent_channels=config["latent_channels"]
        self.conv_encode=nn.Conv2d(self.input_channels,self.latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular')
        self.conv_decode=nn.Conv2d(self.latent_channels,self.output_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular')
        self.process1=process_block(self.latent_channels)
        self.process2=process_block(self.latent_channels)
        self.process3=process_block(self.latent_channels)
        self.process4=process_block(self.latent_channels)
        
    def forward(self,x):
        x=self.conv_encode(x)
        x=self.process1(x)+x
        x=self.process2(x)+x
        x=self.process3(x)+x
        x=self.process4(x)+x
        x=self.conv_decode(x)
        return x

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
