import pyqg_explorer.simulations.util as util
from pyqg.diagnostic_tools import calc_ispec
import numpy as np


def run_simulation(m, sampling_freq=1000, sampling_dist='uniform'):
    snapshots = []
    while m.t < m.tmax:
        if m.tc % sampling_freq == 0:
            ds=m.to_dataset().copy(deep=True)
            KEspec=(m.wv2*np.abs(m.ph)**2/m.M**2)
            KEs_upper = calc_ispec(m, KEspec[0])
            KEs_lower = calc_ispec(m, KEspec[1])
            KEs_both=np.vstack((KEs_upper[1],KEs_lower[1]))
            ds['KE_ispec']=(["lev","ispec_k"],KEs_both)
            snapshots.append(ds)
        m._step_forward()
    ## Concat snapshots into one dataset
    d_cat=util.concat_and_convert(snapshots)
    ## Add ispec online metrics
    d_cat=util.add_ispecs(d_cat,snapshots,m)
    return d_cat


def generate_dataset(sampling_freq=1000, sampling_dist='uniform', parameterization=None, **kwargs):
    if parameterization is not None:
        if isinstance(parameterization, str):
            model_cnn=misc.load_model(parameterization)
            parameterization=parameterizations.Parameterization(model_cnn)
    m = util.initialize_pyqg_model(parameterization=parameterization,**kwargs)
    return run_simulation(m, sampling_freq=sampling_freq, sampling_dist=sampling_dist)

    