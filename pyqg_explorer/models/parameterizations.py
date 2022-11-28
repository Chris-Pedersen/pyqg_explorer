import pyqg
import torch
import copy
import numpy as np
import pyqg_explorer.util.transforms as transforms

class Parameterization(pyqg.QParameterization):
    """ pyqg subgrid parameterisation for the potential vorticity"""
    
    def __init__(self,models,normalise=True):
        """ Initialise with a list of torch models, one for each layer """
        self.models=models
        self.normalise=normalise

        self.x_transforms_upper=(models[0].config["x_mean"],models[0].config["x_std"])
        self.y_transforms_upper=(models[0].config["y_mean"],models[0].config["y_std"])
        self.x_transforms_lower=(models[1].config["x_mean"],models[1].config["x_std"])
        self.y_transforms_lower=(models[1].config["y_mean"],models[1].config["y_std"])

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

        if np.isnan(q1.any()) or np.isnan(q2.any()):
            print("NaNs in pv field")

        ## Convert to tensor, redistribute dimensions to work with the pytorch CNN,
        ## convert to float and then divide by the model's normalisation factors
        q1=((torch.tensor(q1).unsqueeze(0).unsqueeze(0)).float())
        q2=((torch.tensor(q2).unsqueeze(0).unsqueeze(0)).float())

        ## Renormalise input (q) field
        q1=transforms.normalise_field(q1,self.x_transforms_upper[0],self.x_transforms_upper[1])
        q2=transforms.normalise_field(q2,self.x_transforms_lower[0],self.x_transforms_lower[1])

        ## Pass input potential vorticity fields into pytorch model
        ## Renormalise using the model's normalisation factors
        ## Remove unused dimensions required by torch, and convert to numpy array
        s_1=((self.models[0](q1)).squeeze())
        s_2=((self.models[1](q2)).squeeze())

        ## Renormalise output (S) field
        s_1=transforms.denormalise_field(s_1,self.y_transforms_upper[0],self.y_transforms_upper[1])
        s_2=transforms.denormalise_field(s_2,self.y_transforms_lower[0],self.y_transforms_lower[1])

        s_1=s_1.detach().numpy()
        s_2=s_2.detach().numpy()
        
        ## Rescale to 0 mean if required
        if self.normalise:
            s_1=s_1-np.mean(s_1)
            s_2=s_2-np.mean(s_2)

        ## Stack into a single array of shape (nz, nx, ny) and convert to double
        ## to be read by pyqg
        forcing=np.stack((s_1,s_2),axis=0).astype(np.double)

        return forcing
