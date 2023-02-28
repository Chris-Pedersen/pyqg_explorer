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
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.config["lr"])
        return optimizer

    def loss_fn(self,x,y):
        return nn.MSELoss()(x, y)

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


## From Andrew/Pavel's code, function to create a CNN block
def make_block(in_channels: int, out_channels: int, kernel_size: int, 
        ReLU = 'ReLU', batch_norm = True) -> list:
    '''
    Packs convolutional layer and optionally ReLU/BatchNorm2d
    layers in a list
    '''
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
        padding='same', padding_mode='circular')
    block = [conv]
    if ReLU == 'ReLU':
        block.append(nn.ReLU())
    elif ReLU == 'LeakyReLU':
        block.append(nn.LeakyReLU(0.2))
    elif ReLU == 'False':
        pass
    else:
        print('Error: wrong ReLU parameter')
    if batch_norm:
        block.append(nn.BatchNorm2d(out_channels))
    return block


class AndrewCNN(BaseModel):
    def __init__(self,config,model_beta=None):
        '''
        Packs sequence of n_conv=config["conv_layers"] convolutional layers in a list.
        First layer has config["input_channels"] input channels, and last layer has
        config["output_channels"] output channels
        '''
        super().__init__(config,model_beta)

        blocks = []
        ## If the conv_layers key is missing, we are running
        ## with an 8 layer CNN
        if ("conv_layer" in self.config) == False:
            self.config["conv_layers"]=8
        blocks.extend(make_block(self.config["input_channels"],128,5,self.config["activation"])) #1
        blocks.extend(make_block(128,64,5,self.config["activation"]))                            #2
        if self.config["conv_layers"]==3:
            blocks.extend(make_block(64,self.config["output_channels"],3,'False',False))
        elif self.config["conv_layers"]==4:
            blocks.extend(make_block(64,32,3,self.config["activation"]))                            
            blocks.extend(make_block(32,self.config["output_channels"],3,'False',False))
        else: ## 5 layers or more
            blocks.extend(make_block(64,32,3,self.config["activation"])) ## 3rd layer
            for aa in range(4,config["conv_layers"]):
                ## 4th and above layer
                blocks.extend(make_block(32,32,3,self.config["activation"]))
            ## Output layer
            blocks.extend(make_block(32,self.config["output_channels"],3,'False',False))
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv(x)
        return x

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

####################### ResNet architectures #######################
def GetConvBlock(in_channels,intermediate_channels,out_channels,kernel_size=3):
    conv1=nn.Conv2d(in_channels,intermediate_channels,kernel_size,padding="same",padding_mode="circular")
    conv2=nn.Conv2d(intermediate_channels,out_channels,kernel_size,padding="same",padding_mode="circular")
    block=[conv1]
    block.append(nn.ReLU())
    block.append(nn.BatchNorm2d(intermediate_channels))
    block.append(conv2)
    return nn.Sequential(*block)


class ResidualBlock(nn.Module):
    """ Stacks 2 convolutional layers with a residual connection, forming a single
        residual block """
    def __init__(self,in_channels,intermediate_channels,out_channels,kernel_size):
        super().__init__()
        self.conv3x3=nn.ModuleList([*GetConvBlock(in_channels,intermediate_channels,out_channels,kernel_size)])
        self.batchnorm=nn.BatchNorm2d(in_channels)
        
    def forward(self,x):
        residual=x
        for layer in self.conv3x3:
            x=layer(x)
        x+=residual
        x=torch.nn.functional.relu(x)
        x=self.batchnorm(x)
        return x

    
class NarrowResBlock(nn.Module):
    """ Stacks 2 convolutional layers with a residual connection, forming a single
        residual block """
    def __init__(self,in_channels,intermediate_channels,out_channels,kernel_size):
        super().__init__()
        self.cut_idx=int(out_channels/2)
        self.conv3x3=nn.ModuleList([*GetConvBlock(in_channels,intermediate_channels,out_channels,kernel_size)])
        self.batchnorm=nn.BatchNorm2d(in_channels)
        
    def forward(self,x):
        residual=x
        for layer in self.conv3x3:
            x=layer(x)
        ## "Narrow" residual connection
        x=torch.cat(((x[:,:self.cut_idx,:,:]+residual[:,0,:,:].unsqueeze(1)),x[:,self.cut_idx:,:,:]+residual[:,0,:,:].unsqueeze(1)),dim=1)
        x=torch.nn.functional.relu(x)
        x=self.batchnorm(x)
        return x
    
    
class ResNetParallel(base_model.BaseModel):
    """ Resnet where the residual blocks run in parallel, after a joint convolutional first layer """
    def __init__(self,config,model_beta=None):
        super().__init__(config,model_beta)
        self.num_res_blocks=config["residual_blocks"]
        self.conv_filters=config["conv_filters"]
        self.first_conv=nn.Conv2d(config["input_channels"],128,kernel_size=3,padding="same",padding_mode="circular")
        self.batchnorm1=nn.BatchNorm2d(128)
        self.second_conv=nn.Conv2d(128,self.conv_filters,kernel_size=3,padding="same",padding_mode="circular")
        self.batchnorm2=nn.BatchNorm2d(self.conv_filters)
        self.resnets_1=nn.ModuleList()
        self.resnets_2=nn.ModuleList()
        for aa in range(self.num_res_blocks):
            self.resnets_1.append(ResidualBlock(self.conv_filters,self.conv_filters,self.conv_filters,3))
            self.resnets_2.append(ResidualBlock(self.conv_filters,self.conv_filters,self.conv_filters,3))
        self.final_conv_1=nn.Conv2d(self.conv_filters,1,kernel_size=3,padding="same",padding_mode="circular")
        self.final_conv_2=nn.Conv2d(self.conv_filters,1,kernel_size=3,padding="same",padding_mode="circular")
        
    def forward(self,x):
        residual=x
        x=self.first_conv(x)
        x=self.batchnorm1(x)
        x=torch.nn.functional.relu(x)
        x=self.second_conv(x)
        x=self.batchnorm2(x)
        
        x_1=x+residual[:,0,:,:].unsqueeze(1)
        x_2=x+residual[:,1,:,:].unsqueeze(1)
        
        x_1=torch.nn.functional.relu(x_1)
        x_2=torch.nn.functional.relu(x_2)
        
        for layer_index in range(len(self.resnets_1)):
            x_1=self.resnets_1[layer_index](x_1)
            x_2=self.resnets_2[layer_index](x_2)
            
        x_1=self.final_conv_1(x_1)
        x_2=self.final_conv_2(x_2)
        
        x=torch.cat((x_1,x_2),dim=1)
        
        return x
    
    
class ResNetSingle(base_model.BaseModel):
    """ Resnet with a single channel of residual blocks, splitting each filter set into upper and lower
        residual connections """
    def __init__(self,config,model_beta=None):
        super().__init__(config,model_beta)
        self.num_res_blocks=config["residual_blocks"]
        self.conv_filters=config["conv_filters"]
        self.first_conv=nn.Conv2d(config["input_channels"],128,kernel_size=3,padding="same",padding_mode="circular")
        self.batchnorm1=nn.BatchNorm2d(128)
        self.second_conv=nn.Conv2d(128,self.conv_filters,kernel_size=3,padding="same",padding_mode="circular")
        self.batchnorm2=nn.BatchNorm2d(self.conv_filters)
        self.resnets=nn.ModuleList()
        for aa in range(self.num_res_blocks):
            self.resnets.append(NarrowResBlock(self.conv_filters,self.conv_filters,self.conv_filters,3))
        self.final_conv=nn.Conv2d(64,2,kernel_size=3,padding="same",padding_mode="circular")
        
    def forward(self,x):
        residual=x
        x=self.first_conv(x)
        x=self.batchnorm1(x)
        x=torch.nn.functional.relu(x)
        x=self.second_conv(x)
        x=self.batchnorm2(x)
        
        ## Upper layer residual connection for the first half of the filters
        ## Lower layer for the second half
        x=torch.cat(((x[:,:int(self.conv_filters/2),:,:]+residual[:,0,:,:].unsqueeze(1)),x[:,int(self.conv_filters/2):,:,:]+residual[:,0,:,:].unsqueeze(1)),dim=1)
        for layer_index in range(len(self.resnets)):
            x=self.resnets[layer_index](x)
            
        x=self.final_conv(x)
        
        return x


class ResNetChoke(BaseModel):
    """ Resnet where each residual block is collapsed down to two channels, one for upper and one for
        lower layer """
    def __init__(self,config,model_beta=None):
        super().__init__(config,model_beta)
        self.network=nn.ModuleList([])
        for aa in range(config["conv_layers"]):
            self.network.append(ResidualBlock(2,config["intermediate_channels"],2,3))
        
    def forward(self,x):
        for module in self.network:
            x=module(x)
        return x
