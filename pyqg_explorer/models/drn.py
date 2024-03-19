import torch.nn as nn

def process_block(latent_channels):
    """ Processor block, with dilated CNNs """
    process_block=nn.Sequential(nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=1),nn.ReLU(),
                              nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=2),nn.ReLU(),
                              nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=4),nn.ReLU(),
                              nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=8),nn.ReLU(),
                              nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=4),nn.ReLU(),
                              nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=2),nn.ReLU(),
                              nn.Conv2d(latent_channels,latent_channels,kernel_size=3,stride=1,padding="same",padding_mode='circular',dilation=1),nn.ReLU())
                              
    return process_block
    


class DRN(nn.Module):
    """ My implementation of Dilated Res-Net, using the encode-process-decode paradigm from https://arxiv.org/abs/2112.15275 """
    def __init__(self,input_channels=2,output_channels=2,latent_channels=48):
        super(DRN, self).__init__()
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.latent_channels=latent_channels
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
