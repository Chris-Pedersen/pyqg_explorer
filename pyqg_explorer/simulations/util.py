import xarray as xr
import numpy as np
import pyqg
from pyqg.diagnostic_tools import calc_ispec

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