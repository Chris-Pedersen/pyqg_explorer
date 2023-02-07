import os
import sys
import glob
import pyqg
import pickle
import numpy as np
import xarray as xr
import json
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pyqg.xarray_output import spatial_dims
import pyqg_subgrid_experiments as pse
import pyqg_explorer.util.misc
import pyqg_explorer.models.parameterizations as parameterizations

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
)

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

def spectral_filter_and_coarsen(hires_var, m1, m2, filtr='builtin'):
    if not isinstance(m1, pyqg.QGModel):
        m1 = pse.Dataset.wrap(m1).m
        m2 = pse.Dataset.wrap(m2).m

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
        m1 = pse.Dataset.wrap(m1).m
        m2 = pse.Dataset.wrap(m2).m

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

        da = pse.Dataset.wrap(m1).isel(time=-1).real_var(hires_var)

        return filtr(da).coarsen(y=scale, x=scale).mean().data

    elif hires_var.shape == m1.qh.shape:
        return m2.fft(realspace_filter_and_coarsen(m1.ifft(hires_var), m1, m2, filtr))
    else:
        raise ValueError

def run_divergence_simulations(m1, m2, sampling_freq=1000, sampling_dist='uniform', downscaling='spectral', **kw):
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

    snapshots = []

    steps=0
    while steps < 1000:
        """ After divergence start time, save every snapshot for both the LR and HRC fields """
        # Convert the low-resolution model to an xarray dataset
        ds = m2.to_dataset().copy(deep=True)
        m1_lrc=m2.ifft(downscaled(m1.qh))

        ## Inner function to add things to the dataset
        def save_var(key, val):
            zero = ds.q*0
            if len(val.shape) == 3: val = val[np.newaxis]
            ds[key] = zero + val

        ## Save LRC field
        save_var('q_hrc', m1_lrc)
        ## Save snapshot
        snapshots.append(ds)
            
        ## Evolve each sim forward
        m1._step_forward()
        m2._step_forward()
        steps+=1
    # Concatenate the datasets along the time dimension
    return concat_and_convert(snapshots).assign_attrs(hires=m1.nx, lores=m2.nx)

def generate_divergence_dataset(hires=256, lores=64, q_init=None, parameterization=None, downscaling='spectral', **kw):
    def downscaled(hires_var):
        if downscaling == 'spectral':
            return spectral_filter_and_coarsen(hires_var, m1, m2)
        else:
            return realspace_filter_and_coarsen(hires_var, m1, m2)
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
    m2 = pyqg.QGModel(parameterization=parameterization,**params2)
    
    ## If we pass a set of initial q1q2, use these to initialise the sim
    if q_init is not None:
        m1.q=q_init.astype('float64')
        m1._invert()
        m1_lrc=m2.ifft(downscaled(m1.qh))
        m2.q=m1_lrc
        m2._invert()

    ds = run_divergence_simulations(m1, m2, **forcing_params)
    return ds.assign_attrs(pyqg_params=json.dumps(params2))

def test_model_divergence(model_cnn):
    ## Load pyqg ICs
    data_ICs=xr.open_dataset("/scratch/cp3759/pyqg_data/sims/highres_ICs/10_ics.nc")
    q_ics=data_ICs.q.data

    ## Load parameterisation we want to test
    parameterization=parameterizations.Parameterization(model_cnn)

    kwargs={}
    kwargs["tmax"]=5*24*60*60*360.+(3600*100) ### Set a time to go some number of timesteps beyond start_div
    
    l2_baseline=np.empty((10,1000))
    l2_param=np.empty((10,1000))

    for aa, ic in enumerate(q_ics):
        ds = generate_divergence_dataset(q_init=ic,parameterization=None,**kwargs)
        ds_p = generate_divergence_dataset(q_init=ic,parameterization=parameterization,**kwargs)

        ## Convert to numpy arrays
        data_q=ds.q.data
        data_q_hrc=ds.q_hrc.data

        ## Parameterised sims
        data_q_p=ds_p.q.data
        data_q_hrc_p=ds_p.q_hrc.data

        for bb in range(1000):
            l2_baseline[aa][bb]=np.linalg.norm(data_q[bb]-data_q_hrc[bb])
            l2_param[aa][bb]=np.linalg.norm(data_q_p[bb]-data_q_hrc_p[bb])
            
    mean_l2=np.mean(l2_baseline,axis=0)
    std_l2=np.std(l2_baseline,axis=0)

    mean_l2_param=np.mean(l2_param,axis=0)
    std_l2_param=np.std(l2_param,axis=0)

    fig, axs = plt.subplots(1, 3,figsize=(18,5))

    axs[0].plot(mean_l2[:15],label="No parameterisation",color="blue")
    axs[0].fill_between(np.linspace(0,14,15),mean_l2[:15]+std_l2[:15],mean_l2[:15]-std_l2[:15],alpha=0.1,color="blue")
    axs[0].plot(mean_l2_param[:15],label="Parameterised sim",color="orange")
    axs[0].fill_between(np.linspace(0,14,15),mean_l2_param[:15]+std_l2_param[:15],mean_l2_param[:15]-std_l2_param[:15],alpha=0.1,color="orange")
    axs[0].legend()
    axs[0].set_title("L2 error between high-res coarsened sim, and low-res sim")
    axs[0].set_ylabel("L2 error")
    axs[0].set_xlabel("Timestep")


    axs[1].plot(mean_l2[:100],label="No parameterisation",color="blue")
    axs[1].fill_between(np.linspace(0,99,100),mean_l2[:100]+std_l2[:100],mean_l2[:100]-std_l2[:100],alpha=0.1,color="blue")
    axs[1].plot(mean_l2_param[:100],label="Parameterised sim",color="orange")
    axs[1].fill_between(np.linspace(0,99,100),mean_l2_param[:100]+std_l2_param[:100],mean_l2_param[:100]-std_l2_param[:100],alpha=0.1,color="orange")
    axs[1].legend()
    axs[1].set_title("L2 error between high-res coarsened sim, and low-res sim")
    axs[1].set_ylabel("L2 error")
    axs[1].set_xlabel("Timestep")

    axs[2].plot(mean_l2,label="No parameterisation",color="blue")
    axs[2].fill_between(np.linspace(0,999,1000),mean_l2+std_l2,mean_l2-std_l2,alpha=0.1,color="blue")
    axs[2].plot(mean_l2_param,label="Parameterised sim",color="orange")
    axs[2].fill_between(np.linspace(0,999,1000),mean_l2_param+std_l2_param,mean_l2_param-std_l2_param,alpha=0.1,color="orange")
    axs[2].legend()
    axs[2].set_title("L2 error between high-res coarsened sim, and low-res sim")
    axs[2].set_ylabel("L2 error")
    axs[2].set_xlabel("Timestep")

    return fig