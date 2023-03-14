
import torch
import torch.nn as nn
import pyqg_explorer.models.base_model as base_model

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


class FCNN(base_model.BaseModel):
    def __init__(self,config,model_beta=None,residual=False):
        '''
        Packs sequence of n_conv=config["conv_layers"] convolutional layers in a list.
        First layer has config["input_channels"] input channels, and last layer has
        config["output_channels"] output channels
        '''
        super().__init__(config,model_beta,residual)

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
