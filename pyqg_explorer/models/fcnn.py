import torch
import torch.nn as nn
import numpy as np
import pyqg_explorer.util.transforms as transforms
import os
import pickle

class TimeMLP(nn.Module):
    '''
    MLP to proocess time embeddings at each conv layer of diffusion process
    '''
    def __init__(self,embedding_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim,out_dim))
        self.act=nn.SiLU()
    def forward(self,x,t):
        t_emb=self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        x=x+t_emb
  
        return self.act(x)


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,time_embedding_dim,final_layer=False):
        """ Conv block including time embeddings
            Each block consists of two convolutional layers before non-linearity """
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.time_embedding_dim=time_embedding_dim
        self.final_layer=final_layer
        self.conv0=nn.Sequential(nn.Conv2d(self.in_channels,self.out_channels,self.kernel_size,
                                padding="same", padding_mode="circular"),nn.BatchNorm2d(self.out_channels),
                                nn.Conv2d(self.out_channels,self.out_channels,self.kernel_size,
                                padding="same", padding_mode="circular"),nn.BatchNorm2d(self.out_channels),
                                nn.SiLU())
        self.time_mlp=TimeMLP(self.time_embedding_dim,self.out_channels,self.out_channels)
        self.conv1=nn.Sequential(nn.Conv2d(self.out_channels,self.out_channels,self.kernel_size,
                                padding="same", padding_mode="circular"),nn.BatchNorm2d(self.out_channels),
                                nn.Conv2d(self.out_channels,self.out_channels,self.kernel_size,
                                padding="same", padding_mode="circular"))
        if self.final_layer==False:
            self.conv1.append(nn.BatchNorm2d(self.out_channels))
            self.conv1.append(nn.SiLU())
        
    def forward(self,x,t):
        x=self.conv0(x)
        if self.final_layer==False:
            x=self.time_mlp(x,t)
        x=self.conv1(x)
        return x

class ConvBlockS(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,time_embedding_dim,final_layer=False):
        """ Conv block including time embeddings 
            Each block consists of a single convolutional layers before non-linearity """
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.time_embedding_dim=time_embedding_dim
        self.final_layer=final_layer
        self.conv0=nn.Sequential(nn.Conv2d(self.in_channels,self.out_channels,self.kernel_size,
                                padding="same", padding_mode="circular"),nn.BatchNorm2d(self.out_channels),
                                nn.SiLU())
        self.time_mlp=TimeMLP(self.time_embedding_dim,self.out_channels,self.out_channels)
        self.conv1=nn.Sequential(nn.Conv2d(self.out_channels,self.out_channels,self.kernel_size,
                                padding="same", padding_mode="circular"))
        if self.final_layer==False:
            self.conv1.append(nn.BatchNorm2d(self.out_channels))
            self.conv1.append(nn.SiLU())
        
    def forward(self,x,t):
        x=self.conv0(x)
        if self.final_layer==False:
            x=self.time_mlp(x,t)
        x=self.conv1(x)
        return x

class FCNNT(nn.Module):
    def __init__(self,config):
        '''
        CNN with time embeddings, used in diffusion model
        Packs sequence of n_conv=config["conv_layers"] convolutional layers in a list.
        First layer has config["input_channels"] input channels, and last layer has
        config["output_channels"] output channels
        
        timesteps: number of timesteps
        time_embedding_dim: size of time embedding
        '''
        super().__init__()
        self.config=config
        self.time_embedding=nn.Embedding(self.config["timesteps"],self.config["time_embedding_dim"])
        self.model_type="FCNN"

        blocks = []
        ## If the conv_layers key is missing, we are running
        ## with an 8 layer CNN
        if ("conv_layer" in self.config) == False:
            self.config["conv_layers"]=8

        self.conv=nn.ModuleList([ConvBlock(self.config["input_channels"],128,5,self.config["time_embedding_dim"])])
        self.conv.append(ConvBlock(128,64,5,self.config["time_embedding_dim"]))
        if self.config["conv_layers"]==3:
            self.conv.append(ConvBlock(64,self.config["output_channels"],3,self.config["time_embedding_dim"],final_layer=True))
        elif self.config["conv_layers"]==4:
            self.conv.append(ConvBlock(64,32,3,self.config["time_embedding_dim"]))
            self.conv.append(ConvBlock(32,self.config["output_channels"],3,self.config["time_embedding_dim"],final_layer=True))
        else:
            self.conv.append(ConvBlock(64,32,3,self.config["time_embedding_dim"]))
            for aa in range(4,self.config["conv_layers"]):
                self.conv.append(ConvBlock(32,32,3,self.config["time_embedding_dim"]))
            self.conv.append(ConvBlock(32,self.config["output_channels"],3,self.config["time_embedding_dim"],final_layer=True))

    def forward(self,x,t):
        t=self.time_embedding(t)
        for layer in self.conv:
            x=layer(x,t)
        return x

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
    elif ReLU == 'SiLU':
        block.append(nn.SiLU())
    elif ReLU == 'LeakyReLU':
        block.append(nn.LeakyReLU(0.2))
    elif ReLU == 'False':
        pass
    else:
        print('Error: wrong ReLU parameter:')
    if batch_norm:
        block.append(nn.BatchNorm2d(out_channels))
    return block


class FCNN(nn.Module):
    def __init__(self,config):
        '''
        Packs sequence of n_conv=config["conv_layers"] convolutional layers in a list.
        First layer has config["input_channels"] input channels, and last layer has
        config["output_channels"] output channels
        '''
        super().__init__()
        self.config=config

        blocks = []
        ## If the conv_layers key is missing, we are running
        ## with an 8 layer CNN
        if ("conv_layer" in self.config) == False:
            self.config["conv_layers"]=8
        ## Back batch_norm toggle backwards compatible - models trained pre-3rd April 2024
        ## won't have a batch_norm entry - they will all just have batch_norm though
        if ("batch_norm" in self.config) == False:
            self.config["batch_norm"]=True
        blocks.extend(make_block(self.config["input_channels"],128,5,self.config["activation"],batch_norm=self.config["batch_norm"])) #1
        blocks.extend(make_block(128,64,5,self.config["activation"],batch_norm=self.config["batch_norm"]))                            #2
        if self.config["conv_layers"]==3:
            blocks.extend(make_block(64,self.config["output_channels"],3,'False',False))
        elif self.config["conv_layers"]==4:
            blocks.extend(make_block(64,32,3,self.config["activation"],batch_norm=self.config["batch_norm"]))                            
            blocks.extend(make_block(32,self.config["output_channels"],3,'False',False))
        else: ## 5 layers or more
            blocks.extend(make_block(64,32,3,self.config["activation"],batch_norm=self.config["batch_norm"])) ## 3rd layer
            for aa in range(4,config["conv_layers"]):
                ## 4th and above layer
                blocks.extend(make_block(32,32,3,self.config["activation"],batch_norm=self.config["batch_norm"]))
            ## Output layer
            blocks.extend(make_block(32,self.config["output_channels"],3,'False',False))
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        if len(x.shape)==3:
            x=x.unsqueeze(0)
        x = self.conv(x)
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


class Sri_FCNN(nn.Module):
    def __init__(self,config):
        '''
        Packs sequence of n_conv=config["conv_layers"] convolutional layers in a list.
        First layer has config["input_channels"] input channels, and last layer has
        config["output_channels"] output channels
        '''
        super().__init__()
        self.config=config

        blocks = []
        ## If the conv_layers key is missing, we are running
        ## with an 8 layer CNN
        blocks.extend(make_block(self.config["input_channels"],self.config["n_filters"],5,self.config["activation"],False)) #1
        blocks.extend(make_block(self.config["n_filters"],2,5,'False',False)) 
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv(x)
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