import pyqg
import torch
import copy
import numpy as np

class Parameterization(pyqg.QParameterization):
    """ pyqg subgrid parameterisation for the potential vorticity"""
    
    def __init__(self,models,normalise=True):
        """ Initialise with a list of torch models, one for each layer """
        self.models=models
        self.normalise=normalise

    def __call__(self, m):
        """ 
            Inputs:
                - m: a pyqg model at the current timestep
            Outputs:
                - forcing: a numpy array of shape (nz, nx, ny) with the subgrid
                           forcing values """

        ## Extract potential vorticity values
        q1=m.q[0] ## Upper layer q1
        q2=m.q[1] ## Lower layer q2

        if np.isnan(q1.any()) or np.isnan(q2.any()):
            print("NaNs in pv field")

        ## Convert to tensor, redistribute dimensions to work with the pytorch CNN,
        ## convert to float and then divide by the model's normalisation factors
        q1=((torch.tensor(q1).unsqueeze(0).unsqueeze(0)).float())
        q2=((torch.tensor(q2).unsqueeze(0).unsqueeze(0)).float())

        ## Pass input potential vorticity fields into pytorch model
        ## Using pred method which will account for normalisations
        ## Remove unused dimensions required by torch, and convert to numpy array
        s_1=((self.models[0].pred(q1)).squeeze())
        s_2=((self.models[1].pred(q2)).squeeze())

        s_1=s_1.detach().numpy()
        s_2=s_2.detach().numpy()

        print(s_1)
        print(s_2)
        
        ## Rescale to 0 mean if required
        if self.normalise:
            s_1=s_1-np.mean(s_1)
            s_2=s_2-np.mean(s_2)

        ## Stack into a single array of shape (nz, nx, ny) and convert to double
        ## to be read by pyqg
        forcing=np.stack((s_1,s_2),axis=0).astype(np.double)

        return forcing
