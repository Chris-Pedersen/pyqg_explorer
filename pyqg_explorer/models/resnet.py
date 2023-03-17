
import torch
import torch.nn as nn
import pyqg_explorer.models.base_model as base_model


####################### Various residual block objects #######################
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


class LinearResidualBlock(nn.Module):
    """ Stacks 2 convolutional layers with a residual connection, forming a single
        residual block. Here the nonlinearity and batchnorm is applied *before*
        the residual connection """
    def __init__(self,in_channels,intermediate_channels,out_channels,kernel_size):
        super().__init__()
        self.conv3x3=nn.ModuleList([*GetConvBlock(in_channels,intermediate_channels,out_channels,kernel_size)])
        self.batchnorm=nn.BatchNorm2d(in_channels)
        
    def forward(self,x):
        residual=x
        x=torch.nn.functional.relu(x)
        x=self.batchnorm(x)
        for layer in self.conv3x3:
            x=layer(x)
        x+=residual
        return x


####################### ResNet architectures #######################
class ResNetParallel(base_model.BaseModel):
    """ Resnet where the residual blocks run in parallel, after a joint convolutional first layer """
    def __init__(self,config,model_beta=None,residual=False):
        super().__init__(config,model_beta,residual)
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
    def __init__(self,config,model_beta=None,residual=False):
        super().__init__(config,model_beta,residual)
        self.num_res_blocks=config["residual_blocks"]
        self.conv_filters=config["conv_filters"]
        self.first_conv=nn.Conv2d(config["input_channels"],128,kernel_size=3,padding="same",padding_mode="circular")
        self.batchnorm1=nn.BatchNorm2d(128)
        self.second_conv=nn.Conv2d(128,self.conv_filters,kernel_size=3,padding="same",padding_mode="circular")
        self.batchnorm2=nn.BatchNorm2d(self.conv_filters)
        self.resnets=nn.ModuleList()
        for aa in range(self.num_res_blocks):
            self.resnets.append(NarrowResBlock(self.conv_filters,self.conv_filters,self.conv_filters,3))
        self.final_conv=nn.Conv2d(self.conv_filters,2,kernel_size=3,padding="same",padding_mode="circular")
        
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


class ResNetChoke(base_model.BaseModel):
    """ Resnet where each residual block is collapsed down to two channels, one for upper and one for
        lower layer """
    def __init__(self,config,model_beta=None,residual=False):
        super().__init__(config,model_beta,residual)
        self.network=nn.ModuleList([])
        for aa in range(config["residual_blocks"]):
            self.network.append(ResidualBlock(2,config["conv_filters"],2,3))
        
    def forward(self,x):
        for module in self.network:
            x=module(x)
        return x


class ResNetLinear(base_model.BaseModel):
    """ Resnet where each residual block is collapsed down to two channels, one for upper and one for
        lower layer """
    def __init__(self,config,model_beta=None,residual=False):
        super().__init__(config,model_beta,residual)
        self.network=nn.ModuleList([])
        for aa in range(config["residual_blocks"]):
            self.network.append(LinearResidualBlock(2,config["conv_filters"],2,3))
        
    def forward(self,x):
        for module in self.network:
            x=module(x)
        return x
