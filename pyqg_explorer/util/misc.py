import numpy as np
import io
import torch
import pickle
import pyqg_explorer.models.fcnn as fcnn
import xarray as xr

""" Store some miscellaneous helper methods that are frequently used """


########################## Loading models ################################
## Torch models trained using cuda and then pickled cannot be loaded
## onto cpu using the normal pickle methods: https://github.com/pytorch/pytorch/issues/16797
## This method replaces the pickle.load(input_file), using the same syntax
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def load_model(file_string):
    """ Load a pickled model, either on gpu or cpu """
    with open(file_string, 'rb') as fp:
        if torch.cuda.is_available():
            model_dict = pickle.load(fp)
        else:
            model_dict = CPU_Unpickler(fp).load()
    ## Hardcoded to only work with FCNN for now. The loading of state_dict
    ## should flag errors if we accidentally try and load something else though
    model=fcnn.FCNN(model_dict["config"])
    model.load_state_dict(model_dict["state_dict"])
    return model

########################## Filtering functions ###########################
def spectral_filter_and_coarsen(hires_var, m1, m2, filtr='builtin'):
    if not isinstance(m1, pyqg.QGModel):
        m1 = pse_dataset.Dataset.wrap(m1).m
        m2 = pse_dataset.Dataset.wrap(m2).m

    if hires_var.shape == m1.q.shape:
        return m2.ifft(spectral_filter_and_coarsen(m1.fft(hires_var), m1, m2, filtr))
    elif hires_var.shape == m1.qh.shape:
        if filtr == 'guan2020':
            filtr = np.exp(-m2.wv**2 * (2*m2.dx)**2 / 24)
        elif filtr == 'builtin':
            rfiltr = m2.filt
        elif filtr == 'none':
            filtr = np.ones_like(m2.filtr)
        keep = m2.qh.shape[1]//2
        return np.hstack((
            hires_var[:,:keep,:keep+1],
            hires_var[:,-keep:,:keep+1]
        )) * filtr / (m1.nx / m2.nx)**2
    else:
        raise ValueError


def realspace_filter_and_coarsen(hires_var, m1, m2, filtr='gcm'):
    if not isinstance(m1, pyqg.QGModel):
        m1 = pse_dataset.Dataset.wrap(m1).m
        m2 = pse_dataset.Dataset.wrap(m2).m

    if hires_var.shape == m1.q.shape:
        scale = (m1.nx / m2.nx)
        assert(scale == int(scale))
        scale = int(scale)

        if filtr == 'gcm':
            import gcm_filters
            gcm_filter = gcm_filters.Filter(
                filter_scale=scale,
                dx_min=1,
                filter_shape=gcm_filters.FilterShape.GAUSSIAN,
                grid_type=gcm_filters.GridType.REGULAR,
            )
            filtr = lambda x: gcm_filter.apply(x, dims=['y','x'])
        elif filtr == 'none':
            filtr = lambda x: x

        da = pse_dataset.Dataset.wrap(m1).isel(time=-1).real_var(hires_var)

        return filtr(da).coarsen(y=scale, x=scale).mean().data

    elif hires_var.shape == m1.qh.shape:
        return m2.fft(realspace_filter_and_coarsen(m1.ifft(hires_var), m1, m2, filtr))
    else:
        raise ValueError


## Concatenate a list of datasets along time axis while running pyqg sims
def concat_and_convert(datasets, drop_complex=1):
    # Concatenate datasets along the time dimension
    d = xr.concat(datasets, dim='time')

    # Diagnostics get dropped by this procedure since they're only present for
    # part of the timeseries; resolve this by saving the most recent
    # diagnostics (they're already time-averaged so this is ok)
    for k,v in datasets[-1].variables.items():
        if k not in d:
            d[k] = v.isel(time=-1)

    # To save on storage, reduce double -> single
    for k,v in d.variables.items():
        if v.dtype == np.float64:
            d[k] = v.astype(np.float32)
        elif v.dtype == np.complex128:
            d[k] = v.astype(np.complex64)

    # Potentially drop complex variables
    if drop_complex:
        complex_vars = [k for k,v in d.variables.items() if np.iscomplexobj(v)]
        d = d.drop_vars(complex_vars)

    return d

#### Scripts to concat a suite of pyqg sims into one xarray file ####
def get_string(horizon,param,aa):
    return "/scratch/cp3759/pyqg_data/sims/%d_step_forcing/%d_step_%s%d.nc" % (horizon,horizon,param,aa)

def concat_arrays(horizon,param):
    data_list=[]
    for aa in range(1,276):
        filestring=get_string(horizon,param,aa)
        print(filestring)
        data_new=xr.open_dataset(filestring)
        data_list.append(data_new)
    data_all = xr.concat(data_list,dim="run")
    save_string="/scratch/cp3759/pyqg_data/sims/%d_step_forcing/all_%s.nc" % (horizon,param)
    data_all.to_netcdf(save_string)
    print("Saved to %s" % save_string)
    print("done")
    