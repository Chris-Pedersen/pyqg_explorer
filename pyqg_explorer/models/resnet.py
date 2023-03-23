
import torch
import torch.nn as nn
import pyqg_explorer.models.base_model as base_model


####################### Various residual block components #######################
def GetConvBlock(in_channels,intermediate_channels,out_channels,kernel_size=3,dropout=0.0):
    conv1=nn.Conv2d(in_channels,intermediate_channels,kernel_size,padding="same",padding_mode="circular")
    dropout=nn.Dropout2d(dropout)
    conv2=nn.Conv2d(intermediate_channels,out_channels,kernel_size,padding="same",padding_mode="circular")
    block=[conv1]
    block.append(nn.ReLU())
    block.append(nn.BatchNorm2d(intermediate_channels))
    block.append(conv2)
    return nn.Sequential(*block)

    
class WideResBlock(nn.Module):
    """ Map skip connections to N>2 intermediate layers, with the upper layer connected to the first N/2, 
    and the lower layer connected to the second N/2. For use in the WideResidualNetwork"""
    def __init__(self,in_channels,intermediate_channels,out_channels,kernel_size,dropout=0.0):
        super().__init__()
        self.cut_idx=int(out_channels/2)
        self.conv3x3=nn.ModuleList([*GetConvBlock(in_channels,intermediate_channels,out_channels,kernel_size,dropout)])
        self.batchnorm=nn.BatchNorm2d(in_channels)
        
    def forward(self,x):
        residual=x
        x=torch.nn.functional.relu(x)
        x=self.batchnorm(x)
        for layer in self.conv3x3:
            x=layer(x)
        ## "Narrow" residual connection
        x=torch.cat(((x[:,:self.cut_idx,:,:]+residual[:,0,:,:].unsqueeze(1)),x[:,self.cut_idx:,:,:]+residual[:,1,:,:].unsqueeze(1)),dim=1)
        return x


class ResidualBlock(nn.Module):
    """ Stacks 2 convolutional layers with a residual connection, forming a single
        residual block. Here the nonlinearity and batchnorm is applied *before*
        the residual connection, following 1603.05027 and 1605.07146 """
    def __init__(self,in_channels,intermediate_channels,out_channels,kernel_size,dropout=0.0):
        super().__init__()
        self.batchnorm=nn.BatchNorm2d(in_channels)
        self.conv3x3=nn.ModuleList([*GetConvBlock(in_channels,intermediate_channels,out_channels,kernel_size,dropout)])
        
    def forward(self,x):
        residual=x
        x=torch.nn.functional.relu(x)
        x=self.batchnorm(x)
        for layer in self.conv3x3:
            x=layer(x)
        x+=residual
        return x


####################### ResNet architectures #######################
class ResNet(base_model.BaseModel):
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


class WideResNet(base_model.BaseModel):
    """ Resnet where we have a larger number of intermediate convolutional layers - using skip connections
        from the upper layer to the first half of the intermediate fields, and from the lower layer to the second
        half. This mapping is performed in the torch.cat line of the WideResidualBlock.


        NB Not sure if we actually want to do this, as collapsing these layers back down into a single upper
        and lower layer prediction would struggle to map the identity. So setting this as NotImplemented for now
        and will delete it down the line if we don't come back to it."""

    def __init__(self,config,model_beta=None,residual=False):
        super().__init__(config,model_beta,residual)
        self.network=nn.ModuleList([])
        for aa in range(config["residual_blocks"]):
            self.network.append(WideResidualBlock(2,config["conv_filters"],2,3),config["dropout"])
        
    def forward(self,x):
        raise NotImplementedError
