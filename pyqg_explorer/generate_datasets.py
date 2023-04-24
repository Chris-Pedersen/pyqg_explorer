import os
import sys
import glob
import pyqg
import pickle
import numpy as np
import xarray as xr
import json
from scipy.stats import pearsonr
from pyqg.xarray_output import spatial_dims
from pyqg.diagnostic_tools import calc_ispec
import pyqg_subgrid_experiments.dataset as pse_dataset

import pyqg_explorer.util.misc as misc
import pyqg_explorer.parameterizations.parameterizations as parameterizations

YEAR = 24*60*60*360.

DEFAULT_PYQG_PARAMS = dict(nx=64, dt=3600., tmax=10*YEAR, tavestart=5*YEAR)


FORCING_ATTR_DATABASE = dict(
    uq_subgrid_flux=dict(
        long_name=r"x-component of advected PV subgrid flux, $\overline{u}\,\overline{q} - \overline{uq}$",
        units="meters second ^-2",
    ),
    vq_subgrid_flux=dict(
        long_name=r"y-component of advected PV subgrid flux, $\overline{v}\,\overline{q} - \overline{vq}$",
        units="meters second ^-2",
    ),
    uu_subgrid_flux=dict(
        long_name=r"xx-component of advected velocity subgrid flux, $\overline{u}^2 - \overline{u^2}$",
        units="meters second ^-2",
    ),
    uv_subgrid_flux=dict(
        long_name=r"xy-component of advected velocity subgrid flux, $\overline{u}\,\overline{v} - \overline{uv}$",
        units="meters second ^-2",
    ),
    vv_subgrid_flux=dict(
        long_name=r"yy-component of advected velocity subgrid flux, $\overline{v}^2 - \overline{v^2}$",
        units="meters second ^-2",
    ),
    q_forcing_advection=dict(
        long_name=r"PV subgrid forcing from advection, $\overline{(\mathbf{u} \cdot \nabla)q} - (\overline{\mathbf{u}} \cdot \overline{\nabla})\overline{q}$",
        units="second ^-2",
    ),
    u_forcing_advection=dict(
        long_name=r"x-velocity subgrid forcing from advection, $\overline{(\mathbf{u} \cdot \nabla)u} - (\overline{\mathbf{u}} \cdot \overline{\nabla})\overline{u}$",
        units="second ^-2",
    ),
    v_forcing_advection=dict(
        long_name=r"y-velocity subgrid forcing from advection, $\overline{(\mathbf{u} \cdot \nabla)v} - (\overline{\mathbf{u}} \cdot \overline{\nabla})\overline{v}$",
        units="second ^-2",
    ),
    dqdt_through_lores=dict(
        long_name="PV tendency from passing downscaled high-res initial conditions through low-res simulation",
        units="second ^-2",
    ),
    dqdt_through_hires_downscaled=dict(
        long_name="Downscaled PV tendency from passing high-res initial conditions through high-res simulation",
        units="second ^-2",
    ),
    q_forcing_total=dict(
        long_name="Difference between downscaled high-res tendency and low-res tendency",
        units="second ^-2"
    ),
    energy_transfer=dict(
        long_name="Energy transfer as in fig 9c of Pavel's paper",
        units="m^-1, m^3/second^3"
    ),
)

def spatial_var(var, ds):
    return xr.DataArray(var, coords=dict([(d, ds.coords[d]) for d in spatial_dims]), dims=spatial_dims)

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

def add_ispecs(d_cat,snapshots,m):
    """ For a dataset (d_cat), calculate the isotropic energy transfer and depth-averaged
    KE, and add these and the isotropic grid to the dataset
    Doing this at runtime to avoid having to recalculate this quantity constantly later on """

    ## Take last element for most complete average
    ## Squeeze to fit ispec dims
    ds_test=snapshots[-1].squeeze()
    ispec_k,ispec_energy_transfer=calc_ispec(m, (ds_test.KEflux+ds_test.APEflux+ds_test.paramspec).values)

    ## Take depth-averaged quantity
    avegd=ave_lev(snapshots[-1],m.delta)
    ispec_k_avegd,ispec_KEspec_avegd=calc_ispec(m, avegd.KEspec.squeeze())

    ## Isotropically averaged spectra to the dataset
    d_cat=d_cat.assign_coords(coords={"ispec_k":ispec_k})
    d_cat["ispec_energy_transfer"]=xr.DataArray(ispec_energy_transfer,dims="ispec_k")
    d_cat["ispec_KEspec_avegd"]=xr.DataArray(ispec_KEspec_avegd,dims="ispec_k")

    return d_cat

def ave_lev(arr: xr.DataArray, delta):
    '''
    Average over depth xarray
    delta = H1/H2
    H = H1+H2
    Weights are:
    Hi[0] = H1/H = H1/(H1+H2)=H1/H2/(H1/H2+1)=delta/(1+delta)
    Hi[1] = H2/H = H2/(H1+H2)=1/(1+delta)
    '''
    if 'lev' in arr.dims:
        Hi = xr.DataArray([delta/(1+delta), 1/(1+delta)], dims=['lev'])
        out  = (arr*Hi).sum(dim='lev')
        out.attrs = arr.attrs
        return out
    else:
        return arr

def initialize_pyqg_model(parameterization=None,**kwargs):
    pyqg_kwargs = dict(DEFAULT_PYQG_PARAMS)
    pyqg_kwargs.update(**kwargs)
    ## Check if tmax and tavestart are defined in years or seconds
    ## and rescale accordingly
    if pyqg_kwargs["tmax"]<1000:
        pyqg_kwargs["tmax"]=pyqg_kwargs["tmax"]*YEAR
    if pyqg_kwargs["tavestart"]<1000:
        pyqg_kwargs["tavestart"]=pyqg_kwargs["tavestart"]*YEAR

    return pyqg.QGModel(parameterization=parameterization,**pyqg_kwargs)

def generate_dataset(sampling_freq=1000, sampling_dist='uniform', parameterization=None, **kwargs):
    if parameterization is not None:
        if isinstance(parameterization, str):
            model_cnn=misc.load_model(parameterization)
            parameterization=parameterizations.Parameterization(model_cnn)
    m = initialize_pyqg_model(parameterization=parameterization,**kwargs)
    return run_simulation(m, sampling_freq=sampling_freq, sampling_dist=sampling_dist)

def generate_forcing_dataset(hires=256, lores=64, increment=0, **kw):
    forcing_params = {}
    pyqg_params = {}
    for k, v in kw.items():
        if k in ['downscaling','filtr','sampling_dist', 'sampling_freq']:
            forcing_params[k] = v
        else:
            pyqg_params[k] = v

    params1 = dict(DEFAULT_PYQG_PARAMS)
    params1.update(pyqg_params)
    params1['nx'] = hires

    params2 = dict(DEFAULT_PYQG_PARAMS)
    params2.update(pyqg_params)
    params2['nx'] = lores

    m1 = pyqg.QGModel(**params1)
    m2 = pyqg.QGModel(**params2)

    ds = run_forcing_simulations(m1, m2, increment, **forcing_params)
    return ds.assign_attrs(pyqg_params=json.dumps(params2))

def run_simulation(m, sampling_freq=1000, sampling_dist='uniform'):
    snapshots = []
    while m.t < m.tmax:
        if m.tc % sampling_freq == 0:
            snapshots.append(m.to_dataset().copy(deep=True))
        m._step_forward()
    ## Concat snapshots into one dataset
    d_cat=concat_and_convert(snapshots)
    ## Add ispec online metrics
    d_cat=add_ispecs(d_cat,snapshots,m)
    return d_cat

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
            filtr = m2.filtr
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

def run_forcing_simulations(m1, m2, increment, sampling_freq=1000, sampling_dist='uniform', downscaling='spectral', **kw):
    def downscaled(hires_var):
        if downscaling == 'spectral':
            return spectral_filter_and_coarsen(hires_var, m1, m2, **kw)
        else:
            return realspace_filter_and_coarsen(hires_var, m1, m2, **kw)

    def advected(var):
        if var.shape == m1.q.shape:
            m = m1
        elif var.shape == m2.q.shape:
            m = m2
        else:
            raise ValueError
        ik = -np.array(m._ik)[np.newaxis,np.newaxis,:]
        il = -np.array(m._il)[np.newaxis,:,np.newaxis]
        return m.ifft(ik * m.fft(m.ufull * var)) + m.ifft(il * m.fft(m.vfull * var))

    # Set the diagnostics of the coarse simulation to be those of the hi-res
    # simulation, but downscaled
    old_inc = m2._increment_diagnostics
    def new_inc():
        original_lores_q = np.array(m2.q)
        m2.set_q1q2(*m2.ifft(downscaled(m1.qh)))
        m2._invert()
        old_inc()
        m2.set_q1q2(*original_lores_q)
        m2._invert()
    m2._increment_diagnostics = new_inc

    # Arrays to hold the datasets we'll sample over the course of the
    # simulation 
    snapshots = []

    # If we're sampling irregularly, pick the time index for the next sample
    # from an exponential distribution
    if sampling_dist == 'exponential':
        next_sample = int(np.random.exponential(sampling_freq))

    while m1.t < m1.tmax:
        if sampling_dist == 'exponential':
            # If we're sampling irregularly, check if we've hit the next
            # interval
            should_sample = m1.tc >= next_sample
            if should_sample:
                next_sample = m1.tc + int(np.random.exponential(sampling_freq))
        else:
            # If we're sampling regularly, check if we're at that fixed
            # interval
            should_sample = (m1.tc % sampling_freq == 0) or (m1.tc % sampling_freq == increment)

        if should_sample:
            # Update the PV of the low-resolution model to match the downscaled
            # high-resolution PV
            m2.set_q1q2(*m2.ifft(downscaled(m1.qh)))
            m2._invert() # recompute velocities
            m2._calc_derived_fields()

            # Convert the low-resolution model to an xarray dataset
            ds = m2.to_dataset().copy(deep=True)

            # Compute various versions of the subgrid forcing defined in terms
            # of the advection and downscaling operators
            def save_var(key, val):
                zero = ds.q*0
                if len(val.shape) == 3: val = val[np.newaxis]
                ds[key] = zero + val

            m1._invert()
            def uv_diff(m):
                diff = m.u[0]*m.v[1] - m.u[1]*m.v[0]
                return np.array([m.F1 * diff, -m.F2 * diff])

            save_var('q_forcing_advection', downscaled(advected(m1.q)) - advected(m2.q))
            save_var('u_forcing_advection', downscaled(advected(m1.ufull)) - advected(m2.ufull))
            save_var('v_forcing_advection', downscaled(advected(m1.vfull)) - advected(m2.vfull))
            save_var('streamfunction', m2.p)

            save_var('b_forcing_advection', downscaled(uv_diff(m1)) - uv_diff(m2))

            def corr(a,b):
                def data(x):
                    if isinstance(x,xr.DataArray):
                        return x.data.ravel()
                    else:
                        return x.ravel()
                from scipy.stats import pearsonr
                return pearsonr(data(a),data(b))[0]

            save_var('uq_subgrid_flux', m2.ufull * m2.q - downscaled(m1.ufull * m1.q))
            save_var('vq_subgrid_flux', m2.vfull * m2.q - downscaled(m1.vfull * m1.q))

            save_var('uu_subgrid_flux', m2.ufull**2 - downscaled(m1.ufull**2))
            save_var('vv_subgrid_flux', m2.vfull**2 - downscaled(m1.vfull**2))
            save_var('uv_subgrid_flux', m2.ufull * m2.vfull - downscaled(m1.ufull * m1.vfull))

            # Now, step both models forward (which recomputes ∂q/∂t)
            m1._step_forward()
            m2._step_forward()

            # Store the resulting values of ∂q/∂t
            save_var('dqdt_through_lores', m2.ifft(m2.dqhdt))
            save_var('dqdt_through_hires_downscaled', m2.ifft(downscaled(m1.dqhdt)))

            # Finally, store the difference between those two quantities (which
            # serves as an alternate measure of subgrid forcing, that takes
            # into account other differences in the simulations beyond just
            # hi-res vs. lo-res advection)
            ds['q_forcing_total'] = ds['dqdt_through_hires_downscaled'] - ds['dqdt_through_lores']

            # Add attributes and units to the xarray dataset
            for key, attrs in FORCING_ATTR_DATABASE.items():
                if key in ds:
                    ds[key] = ds[key].assign_attrs(attrs)

            # Save the datasets
            if 'dqdt' in ds: ds = ds.drop('dqdt')
            snapshots.append(ds)
        else:
            # If we aren't sampling at this index, just step both models
            # forward (with the second model continuing to evolve in lock-step)
            #m2.set_q1q2(*downscaled_hires_q())
            m1._step_forward()
            m2._step_forward()

    # Concatenate the datasets along the time dimension
    return concat_and_convert(snapshots).assign_attrs(hires=m1.nx, lores=m2.nx)

def generate_parameterized_dataset(sampling_freq=1000, sampling_dist='uniform', parameterization=None, increment=0, **kwargs):
    if parameterization is not None:
        if isinstance(parameterization, str):
            model_cnn=misc.load_model(parameterization)
            parameterization=parameterizations.Parameterization(model_cnn)
    m = initialize_pyqg_model(parameterization=parameterization,**kwargs)
    return run_parameterized_simulation(m, sampling_freq=sampling_freq, increment=increment, sampling_dist=sampling_dist)


def run_parameterized_simulation(m, sampling_freq=1000, increment=0, sampling_dist='uniform'):
    snapshots = []
    while m.t < m.tmax:
        if ((m.tc % sampling_freq == 0) or (m.tc % sampling_freq == increment)) and m.tc>500:
            snapshots.append(m.to_dataset().copy(deep=True))
            def save_var(key, val):
                zero = snapshots[-1].q*0
                if len(val.shape) == 3: val = val[np.newaxis]
                snapshots[-1][key] = zero + val
            forcing=m.q_parameterization.get_cached_forcing()
            save_var("q_subgrid_forcing",forcing)
        m._step_forward()
    ## Concat snapshots into one dataset
    d_cat=concat_and_convert(snapshots)
    ## Add ispec online metrics
    d_cat=add_ispecs(d_cat,snapshots,m)
    return d_cat

def correlation_decay_curve(m1, m2, thresh=0.25, perturbation_sd=1e-10, max_timesteps=100000, coarsening='spectral', **kw):
    def coarsened(hires_var):
        if coarsening == 'spectral':
            return spectral_filter_and_coarsen(hires_var, m1, m2, **kw)
        else:
            return realspace_filter_and_coarsen(hires_var, m1, m2, **kw)    

    def possibly_coarsened(var):
        if m1.nx == m2.nx:
            return var
        else:
            return coarsened(var)

    def correlation(qa, qb):
        return pearsonr(qa.ravel(), qb.ravel())[0]
    
    for _ in range(3):
        m2.set_q1q2(*possibly_coarsened(m1.q))
        m1._step_forward()
        m2._step_forward()
        
    initial_conditions = possibly_coarsened(m1.q)
    perturbed_conditions = initial_conditions + np.random.normal(size=initial_conditions.shape) * perturbation_sd
    
    assert correlation(initial_conditions, perturbed_conditions) > 0.9
    
    m2.set_q1q2(*perturbed_conditions)
    
    corrs = []
    
    while len(corrs) < max_timesteps:
        m1._step_forward()
        m2._step_forward() 
        corrs.append(correlation(m2.q, possibly_coarsened(m1.q)))
        if corrs[-1] <= thresh:
            break
    
    return corrs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_to', type=str)
    parser.add_argument('--run_number', type=int)
    parser.add_argument('--parameterization', action="store_true")
    args, extra = parser.parse_known_args()

    # Setup parameters for dataset generation functions
    kwargs = dict()
    for param in extra:
        key, val = param.split('=')
        try:
            val = float(val)
        except:
            pass
        kwargs[key.replace('--', '')] = val
    files = [f.strip() for f in args.save_to.split(',')]
    files = [f for f in files if f]

    if args.parameterization:
        model_cnn=misc.load_model("/scratch/cp3759/pyqg_data/models/cnn_theta_ONLY_forcing1_both_epoch200.pt")
        parameterization=parameterizations.Parameterization(model_cnn,cache_forcing=True)

    for save_file in files:
        ## Add run number to save file name
        save_file=list(save_file)
        save_file.insert(-3,str(args.run_number))
        save_file="".join(save_file)
        os.system(f"mkdir -p {os.path.dirname(os.path.realpath(save_file))}")
        if args.parameterization:
            ds = generate_parameterized_dataset(parameterization=parameterization,increment=args.increment)
        else:
            ds = generate_forcing_dataset(**kwargs)
        ds.to_netcdf(save_file)
