from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
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
