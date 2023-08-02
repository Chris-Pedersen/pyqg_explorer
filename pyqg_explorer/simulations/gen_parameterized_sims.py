import pyqg
import json
import numpy as np
import pyqg_explorer.simulations.util as util


def run_parameterized_simulation(m, increment, rollout=1, sampling_freq=1000):
    """ increment: what size dt to store at each sample
        rollout: number of increment steps to store
        sampling_freq: how many timesteps to wait in total before starting a new sampling point
        """
    snapshots = []
    while m.t < m.tmax:
        if increment == 0:
            ## If increment == 0, just sample at each sampling freq
            should_sample = (m.tc % sampling_freq == 0)
        else:
            ## If sampling at increments, identify indices to sample at
            should_sample = (m.tc % sampling_freq == 0) or ((m.tc % sampling_freq % increment == 0) and m.tc % sampling_freq <= rollout*increment)

        ## Don't sample from t=0 (we have no subgrid forcing for very first snapshot)
        if m.tc<500:
            should_sample=False

        ## If sample, save fields
        if should_sample:
            snapshots.append(m.to_dataset().copy(deep=True))
            def save_var(key, val):
                zero = snapshots[-1].q*0
                if len(val.shape) == 3: val = val[np.newaxis]
                snapshots[-1][key] = zero + val
            forcing=m.q_parameterization.get_cached_forcing()
            save_var("q_subgrid_forcing",forcing)
        m._step_forward()
    ## Concat snapshots into one dataset
    d_cat=util.concat_and_convert(snapshots)
    ## Add ispec online metrics
    #d_cat=util.add_ispecs(d_cat,snapshots,m)
    return d_cat

def generate_parameterized_dataset(sampling_freq=1000,increment=0,rollout=1, parameterization=None, **kwargs):
    """ Run a low-res simulation with parameterisation. Save both the coarse resolution field, and the parameterisation at
        some desired set of timestep intervals """
    if parameterization is not None:
        if isinstance(parameterization, str):
            model_cnn=misc.load_model(parameterization)
            parameterization=parameterizations.Parameterization(model_cnn)
    m = util.initialize_pyqg_model(parameterization=parameterization,**kwargs)

    ## Prepare dict containing increment and rollout config to save as ds.attrs
    params = dict(util.DEFAULT_PYQG_PARAMS)
    params.update(kwargs)
    params["sampling_freq"]=sampling_freq
    params["increment"]=increment
    params["rollout"]=rollout

    ds=run_parameterized_simulation(m, increment=increment, rollout=rollout, sampling_freq=sampling_freq)
    return ds.assign_attrs(pyqg_params=json.dumps(params))

