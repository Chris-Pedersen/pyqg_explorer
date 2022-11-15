import pyqg
import numpy as np
import xarray as xr
import os

def concat_in_time(datasets):
    '''
    Concatenation of snapshots in time:
    - Concatenate everything
    - Store averaged statistics
    - Discard complex vars
    - Reduce precision
    '''
    from time import time
    # Concatenate datasets along the time dimension
    tt = time()
    ds = xr.concat(datasets, dim='time')
    if float(time() - tt) > 100:
        print('Line 27 took', time() - tt, 'seconds')
        print('Len of the datasets', len(datasets), '\n')
        
        print('Individual coords:')
        for d in datasets:
            print(d.coords)
        print('Total coord:')
        print(ds.coords)

        print('\n')
        print('individual variables:')
        for d in datasets:
            print(d.variables)
        print('total variables:')
        print(ds.variables)

        raise ValueError('Concatenation of datasets took too long time')
    
    # Diagnostics get dropped by this procedure since they're only present for
    # part of the timeseries; resolve this by saving the most recent
    # diagnostics (they're already time-averaged so this is ok)
    for key,var in datasets[-1].variables.items():
        if key not in ds:
            ds[key] = var.isel(time=-1)

    # To save on storage, reduce double -> single
    # And discard complex vars
    for key,var in ds.variables.items():
        if var.dtype == np.float64:
            ds[key] = var.astype(np.float32)
        elif var.dtype == np.complex128:
            ds = ds.drop_vars(key)

    ds = ds.rename({'p': 'psi'}) # Change for conventional name
    ds['time'] = ds.time.values / 86400
    ds['time'].attrs['units'] = 'days'

    return ds


class Simulation:
    def __init__(self,nx=256,tmax=311040000.0,twrite=1000,tavestart=155520000.0,
                             taveint=86400.0,dt=3600,log_level=1, parameterization=None):
        self.nx=nx
        self.twrite=twrite
        self.tmax=int(tmax)
        self.tavestart=tavestart
        self.taveint=taveint
        self.dt=dt
        self.log_level=log_level
        self.parameterization=parameterization

        ## Initialise model
        self.model=pyqg.QGModel(nx=self.nx,tmax=self.tmax,twrite=self.twrite,tavestart=self.tavestart,taveint=self.taveint,dt=self.dt,log_level=self.log_level,parameterization=self.parameterization)

    
    def run_sim(self,start=0.5,freq=2):
        """ Evolve the pyqg model
        tstart: When to start saving snapshots, as a fraction of tmax
        freq:   Frequency of evolved timesteps we want to store """

        ds=[]
        while self.model.t < self.tmax:
            self.model._step_forward()
            if self.model.t % freq == 0 and self.model.t > int(start*self.tmax):
                ds.append(self.model.to_dataset().copy(deep=True))

        self.sim_data = concat_in_time(ds).astype('float32')
        #out.attrs['pyqg_params'] = str(pyqg_params)

    def save_sim(self,savename,path="/scratch/cp3759/pyqg_data/sims/"):
        print('Saving to file')
        self.sim_data.to_netcdf(os.path.join(path, savename+".nc"))
