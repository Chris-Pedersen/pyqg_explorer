from re import L
import pyqg
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statsmodels.tsa.stattools import acf
import random
from tqdm import tqdm
from IPython.display import HTML


class Simulation:
    def __init__(nx=256,tmax=10,twrite=10000,tavestart=5,dt=7200):
        self.year=24*60*60*360.
        self.nx=nx
        self.tmax=tmax*self.year
        self.twrite=twrite
        self.tavestart=tavestart*self.year
        self.dt=7200

        self.model=pyqg.QGModel(nx=self.nx,tmax=self.tmax,twrite=self.twrite,tavestart=self.tavestart,dt=self.dt)

    
    def run_sim(freq=1,tstart):

        datasets = [[],[]]
        kw = dict(dims=('x','y'), coords={'x': model.x[0], 'y': model.y[:,0]})
        for t in tqdm(range(self.tmax)):
            self.model._step_forward()
            if t % freq == 0 and t > t_start:
                for layer in range(len(self.model.u)):
                    u = xr.DataArray(np.array(self.model.ufull[layer]), **kw)
                    v = xr.DataArray(np.array(self.model.vfull[layer]), **kw)
                    q = xr.DataArray(np.array(self.model.q[layer]), **kw)
                    datasets[layer].append(xr.Dataset(data_vars=dict(
                    x_velocity=u, y_velocity=v, potential_vorticity=q)))
                    
            t += 1
            
        self.time_data = xr.concat(datasets[0], 'time')