from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
import torch
import torch.nn as nn


class BaseModel(LightningModule):
    """ Class to store core model methods """
    def __init__(self,lr:float=0.01):
        super().__init__()
        self.lr=lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self,batch,kind):
        """ Take a batch, perform a train, valid or test step """
        x,y=batch


class CNN(pl.LightningModule):
    def __init__(self,in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
                    nn.Conv2d(in_channels, 128, 5, 1, padding="same",padding_mode='circular'),
                    nn.ReLU(),
                    nn.BatchNorm2d(128))
    
    def forward(self, x):
        print(x.shape)
        x = self.model(x)
        return x


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


class AndrewCNN(pl.LightningModule):
    def __init__(self, n_in: int, n_out: int, ReLU = 'ReLU', lr=0.001) -> list:
        '''
        Packs sequence of 8 convolutional layers in a list.
        First layer has n_in input channels, and Last layer has n_out
        output channels
        '''
        super().__init__()
        self.lr=lr
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        return optimizer

    def loss_fn(self,x,y):
        return nn.MSELoss()(x, y)

    def step(self, batch, kind: str) -> dict:
        """Generic step function that runs the network on a batch and outputs loss
        nformation that will be aggregated at epoch end.
        This function is used to implement the training, validation, and test steps.
        """
        # run the model and calculate loss
        x,y = batch[0],batch[1]
        loss = self.loss_fn(self(x),y)

        total = len(y)

        batch_dict = {
            "loss": loss,
            "total": total,
        }
        return batch_dict

    def epoch_end(self, outputs, kind: str):
        """Generic function for summarizing and logging the loss and accuracy over an
        epoch.
        Creates log entries with name `f"{kind}_loss"` and `f"{kind}_accuracy"`.
        This function is used to implement the training, validation, and test epoch-end
        functions.
        """
        with torch.no_grad():
            # calculate average loss and average accuracy
            total_loss = sum(_["loss"] * _["total"] for _ in outputs)
            total = sum(_["total"] for _ in outputs)
            avg_loss = total_loss / total

        # log
        self.log(f"{kind}_loss", avg_loss)

    def compute_loss(self, x, ytrue):
        '''
        In case you want to use this block for training 
        as regression model with standard trainer cnn_tools.train()
        '''
        return {'loss': nn.MSELoss()(self.forward(x), ytrue)}

    def training_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, "test")