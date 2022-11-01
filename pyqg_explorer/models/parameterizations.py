import pyqg
import torch
import numpy as np

class Parameterization(pyqg.QParameterization):
    """ pyqg subgrid parameterisation for the potential vorticity"""
    
    def __init__(self,models):
        """ Initialise with a list of torch models, one for each layer """
        self.models=models

    def __call__(self, m):
        """ 
            Inputs:
                - m: a pyqg model at the current timestep
            Outputs:
                - forcing: a numpy array of shape (nz, nx, ny) with the subgrid
                           forcing values """
        ## Extract potential vorticity values
        q1=m.q[0]
        q2=m.q[1]
        
        ## Convert to tensor, redistribute dimensions to work with the pytorch CNN,
        ## convert to float and then divide by the model's normalisation factors
        q1=((torch.tensor(q1).unsqueeze(0).unsqueeze(0)).float())/self.models[0].x_renorm
        q2=((torch.tensor(q2).unsqueeze(0).unsqueeze(0)).float())/self.models[1].x_renorm
        
        ## Pass input potential vorticity fields into pytorch model
        ## Renormalise using the model's normalisation factors
        ## Remove unused dimensions required by torch, and convert to numpy array
        s_1=((self.models[0](q1))*self.models[0].y_renorm).squeeze().detach().numpy()
        s_2=((self.models[1](q2))*self.models[1].y_renorm).squeeze().detach().numpy()
        
        ## Stack into a single array of shape (nz, nx, ny) and convert to double
        ## to be read by pyqg
        forcing=np.stack((s_1,s_2),axis=0).astype(np.double)
        return forcing