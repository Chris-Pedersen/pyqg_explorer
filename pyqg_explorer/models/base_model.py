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


class TestCNN(pl.LightningModule):
    def __init__(self):
      #image_size = 64
      super().__init__()
      self.block1=nn.BatchNorm2d(nn.RelLU(nn.Conv2d(1,128,5,padding=0)))
      self.block2=nn.BatchNorm2d(nn.RelLU(nn.Conv2d(128,64,5,padding=0)))
      self.block3=nn.BatchNorm2d(nn.RelLU(nn.Conv2d(64,32,3,padding=0)))
      self.block4=nn.BatchNorm2d(nn.RelLU(nn.Conv2d(32,32,3,padding=0)))
      self.block5=nn.BatchNorm2d(nn.RelLU(nn.Conv2d(32,32,3,padding=0)))
      self.block6=nn.BatchNorm2d(nn.RelLU(nn.Conv2d(32,32,3,padding=0)))
      self.cnv = nn.Conv2d(3,128,5,4)
      self.rel = nn.ReLU()
      self.bn = nn.BatchNorm2d(128)
      self.mxpool = nn.MaxPool2d(4)
      self.flat = nn.Flatten()
      self.fc1 = nn.Linear(1152,64)
      self.fc2 = nn.Linear(64,64)
      self.fc3 = nn.Linear(64,CLASSES)
      self.softmax = nn.Softmax()
      self.accuracy = pl.metrics.Accuracy()

    def forward(self,x):
      out = self.bn(self.rel(self.cnv(x)))
      out = self.flat(self.mxpool(out))
      out = self.rel(self.fc1(out))
      out = self.rel(self.fc2(out))
      out = self.fc3(out)
      return out

    def loss_fn(self,out,target):
      return nn.CrossEntropyLoss()(out.view(-1,CLASSES),target)
    
    def configure_optimizers(self):
      LR = 1e-3
      optimizer = torch.optim.AdamW(self.parameters(),lr=LR)
      return optimizer

    def training_step(self,batch,batch_idx):
      x,y = batch["x"],batch["y"]
      img = x.view(-1,3,IMG_SIZE,IMG_SIZE)
      label = y.view(-1)
      out = self(img)
      loss = self.loss_fn(out,label)
      self.log('train_loss', loss)
      return loss       

    def validation_step(self,batch,batch_idx):
      x,y = batch["x"],batch["y"]
      img = x.view(-1,3,IMG_SIZE,IMG_SIZE)
      label = y.view(-1)
      out = self(img)
      loss = self.loss_fn(out,label)
      out = nn.Softmax(-1)(out) 
      logits = torch.argmax(out,dim=1)
      accu = self.accuracy(logits, label)        
      self.log('valid_loss', loss)
      self.log('train_acc_step', accu)
      return loss, accu


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


class AndrewCNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, ReLU = 'ReLU', div=False) -> list:
        '''
        Packs sequence of 8 convolutional layers in a list.
        First layer has n_in input channels, and Last layer has n_out
        output channels
        '''
        super().__init__()
        self.div = div
        if div:
            n_out *= 2
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
        if self.div:
            # This parameter, 10000, just to improve convergence
            # Note it is not the part of the divergence procedure
            # Physically, it brings gradients from physical scale (1000km)
            # to non-dimensional scale
            x = 10000. * divergence(x)
        return x
    def compute_loss(self, x, ytrue):
        '''
        In case you want to use this block for training 
        as regression model with standard trainer cnn_tools.train()
        '''
        return {'loss': nn.MSELoss()(self.forward(x), ytrue)}