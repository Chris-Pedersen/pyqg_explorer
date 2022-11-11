from pytorch_lightning import LightningModule
import functools
import torch
import torch.nn as nn


class BaseModel(LightningModule):
    """ Class to store core model methods """
    def __init__(self,lr:float=0.001):
        super().__init__()
        self.lr=lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        return optimizer

    def loss_fn(self,x,y):
        return nn.MSELoss()(x, y)

    def step(self,batch,kind):
        x, y = batch[0],batch[1]
        y_hat = self(x)
        loss = self.loss_fn(y_hat,y)
        self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True)       
        return loss

    def training_step(self, batch, batch_idx):      
        return self.step(batch,"train")

    def validation_step(self, batch, batch_idx):      
        return self.step(batch,"valid")
        

##### Unet #########
class UnetGenerator(BaseModel):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, x_renorm=torch.tensor(1.), y_renorm=torch.tensor(1.), ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        ## Register normalisation factors as buffers
        self.register_buffer('x_renorm', x_renorm)
        self.register_buffer('y_renorm', y_renorm)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

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
    def __init__(self, n_in: int, n_out: int, x_renorm=torch.tensor(1.), y_renorm=torch.tensor(1.), ReLU = 'ReLU', lr=0.001) -> list:
        '''
        Packs sequence of 8 convolutional layers in a list.
        First layer has n_in input channels, and Last layer has n_out
        output channels
        '''
        super().__init__(lr=lr)
        self.lr=lr
        ## Register normalisation factors as buffers
        self.register_buffer('x_renorm', x_renorm)
        self.register_buffer('y_renorm', y_renorm)
        blocks = []
        blocks.extend(make_block(n_in,128,5,ReLU))                #1
        blocks.extend(make_block(128,64,5,ReLU))                  #2
        blocks.extend(make_block(64,32,3,ReLU))                   #3
        blocks.extend(make_block(32,32,3,ReLU))                   #4
        blocks.extend(make_block(32,32,3,ReLU))                   #5
        blocks.extend(make_block(32,32,3,ReLU))                   #6
        blocks.extend(make_block(32,32,3,ReLU))                   #7
        blocks.extend(make_block(32,n_out,3,'False',False))       #8
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv(x)
        return x
