import pyqg
import torch
import copy
import numpy as np

class Parameterization(pyqg.QParameterization):
    """ pyqg subgrid parameterisation for the potential vorticity"""
    
    def __init__(self,model,normalise=True):
        """ Initialise with a list of torch models, one for each layer """
        self.model=model
        self.model.eval() ## Ensure we are in eval
        self.normalise=normalise

    def __call__(self, m):
        """ 
            Inputs:
                - m: a pyqg model at the current timestep
            Outputs:
                - forcing: a numpy array of shape (nz, nx, ny) with the subgrid
                           forcing values """


        s=self.model.pred(m.q)

        if self.normalise:
            means=np.mean(s,axis=(1,2))
            s[0]=s[0]-means[0]
            s[1]=s[1]-means[1]

        return s