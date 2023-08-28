import numpy as np
import torch
import torch.nn as nn
from itertools import *

""" Unet-style architecture taken from Adam Subel """

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class Conv_block(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 2,kernel_size = 3, num_layers=2):
        super().__init__()
        self.N_in = num_in

        layers = []
        layers.append(torch.nn.Conv2d(num_in,num_out,kernel_size,padding='same'))
        layers.append(torch.nn.BatchNorm2d(num_out))        
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers-1):
            layers.append(torch.nn.Conv2d(num_out,num_out,kernel_size,padding='same'))
            layers.append(torch.nn.BatchNorm2d(num_out))
            layers.append(torch.nn.ReLU())              

        self.layers = nn.ModuleList(layers)
        
    def forward(self,fts):
        for l in self.layers:
            fts= l(fts)
        return fts
    
class U_net(torch.nn.Module):
    def __init__(self,ch_width,n_out,config,kernel_size = 3):
        super().__init__()
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.config = config
        # going down
        layers = []
        for a,b in pairwise(ch_width):
            layers.append(Conv_block(a,b))
            layers.append(nn.MaxPool2d(2))
        layers.append(Conv_block(b,b))    
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        ch_width.reverse()
        for a,b in pairwise(ch_width[:-1]):
            layers.append(Conv_block(a,b))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(Conv_block(b,b))    
        layers.append(torch.nn.Conv2d(b,n_out,kernel_size,padding='same'))
        
        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width)-1)
        
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            fts= l(fts)
            if count < self.num_steps:
                if isinstance(l,Conv_block):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l,nn.Upsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(temp[int(2*self.num_steps-count-1)].shape[2:])
                    pads = (shape - crop)
                    pads = [pads[1]//2, pads[1]-pads[1]//2,
                            pads[0]//2, pads[0]-pads[0]//2]
                    fts = nn.functional.pad(fts,pads)
                    fts += temp[int(2*self.num_steps-count-1)]
                    count += 1
        return fts 

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
