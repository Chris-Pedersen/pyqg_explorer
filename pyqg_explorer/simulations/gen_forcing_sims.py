import pyqg
import numpy as np
import pyqg_explorer.simulations.util as util


def run_forcing_simulations(m1, m2, increment, rollout=1, sampling_freq=1000, sampling_dist='uniform', downscaling='spectral', **kw):
    def downscaled(hires_var):
        if downscaling == 'spectral':
            return util.spectral_filter_and_coarsen(hires_var, m1, m2, **kw)
        else:
            return util.realspace_filter_and_coarsen(hires_var, m1, m2, **kw)

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

    while m1.t < m1.tmax:
        if sampling_dist == 'exponential':
            # If we're sampling irregularly, check if we've hit the next
            # interval
            should_sample = m1.tc >= next_sample
            if should_sample:
                next_sample = m1.tc + int(np.random.exponential(sampling_freq))
        elif increment == 0:
            ## If increment == 0, just sample at each sampling freq
            should_sample = (m1.tc % sampling_freq == 0)
        else:
            ## If sampling at increments, identify indices to sample at
            should_sample = (m1.tc % sampling_freq == 0) or ((m1.tc % sampling_freq % increment == 0) and m1.tc % sampling_freq <= rollout*increment)

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
            #save_var('u_forcing_advection', downscaled(advected(m1.ufull)) - advected(m2.ufull))
            #save_var('v_forcing_advection', downscaled(advected(m1.vfull)) - advected(m2.vfull))
            #save_var('streamfunction', m2.p)

            #save_var('b_forcing_advection', downscaled(uv_diff(m1)) - uv_diff(m2))

            def corr(a,b):
                def data(x):
                    if isinstance(x,xr.DataArray):
                        return x.data.ravel()
                    else:
                        return x.ravel()
                from scipy.stats import pearsonr
                return pearsonr(data(a),data(b))[0]

            #save_var('uq_subgrid_flux', m2.ufull * m2.q - downscaled(m1.ufull * m1.q))
            #save_var('vq_subgrid_flux', m2.vfull * m2.q - downscaled(m1.vfull * m1.q))

            #save_var('uu_subgrid_flux', m2.ufull**2 - downscaled(m1.ufull**2))
            #save_var('vv_subgrid_flux', m2.vfull**2 - downscaled(m1.vfull**2))
            #save_var('uv_subgrid_flux', m2.ufull * m2.vfull - downscaled(m1.ufull * m1.vfull))

            # Now, step both models forward (which recomputes ∂q/∂t)
            m1._step_forward()
            m2._step_forward()

            # Store the resulting values of ∂q/∂t
            #save_var('dqdt_through_lores', m2.ifft(m2.dqhdt))
            #save_var('dqdt_through_hires_downscaled', m2.ifft(downscaled(m1.dqhdt)))

            # Finally, store the difference between those two quantities (which
            # serves as an alternate measure of subgrid forcing, that takes
            # into account other differences in the simulations beyond just
            # hi-res vs. lo-res advection)
            #ds['q_forcing_total'] = ds['dqdt_through_hires_downscaled'] - ds['dqdt_through_lores']

            # Add attributes and units to the xarray dataset
            for key, attrs in util.FORCING_ATTR_DATABASE.items():
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
    return util.concat_and_convert(snapshots).assign_attrs(hires=m1.nx, lores=m2.nx)


def generate_forcing_dataset(hires=256, lores=64, increment=0, rollout=1, **kw):
    forcing_params = {}
    pyqg_params = {}
    for k, v in kw.items():
        if k in ['downscaling','filtr','sampling_dist', 'sampling_freq']:
            forcing_params[k] = v
        else:
            pyqg_params[k] = v

    params1 = dict(util.DEFAULT_PYQG_PARAMS)
    params1.update(pyqg_params)
    params1['nx'] = hires

    params2 = dict(util.DEFAULT_PYQG_PARAMS)
    params2.update(pyqg_params)
    params2['nx'] = lores

    m1 = pyqg.QGModel(**params1)
    m2 = pyqg.QGModel(**params2)

    ds = run_forcing_simulations(m1, m2, increment, **forcing_params)

    params2["sampling_freq"]=sampling_freq
    params2["increment"]=increment
    params2["rollout"]=rollout
    return ds.assign_attrs(pyqg_params=json.dumps(params2))

