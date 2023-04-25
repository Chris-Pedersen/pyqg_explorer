import pyqg
import torch
import copy
import numpy as np

class Parameterization(pyqg.QParameterization):
    """ pyqg subgrid parameterisation for the potential vorticity"""
    
    def __init__(self,model,normalise=True,cache_forcing=False):
        """ Initialise with a list of torch models, one for each layer """
        self.model=model
        self.model.eval() ## Ensure we are in eval
        self.normalise=normalise
        self.cache_forcing=cache_forcing
        self.cached_forcing=None

    def get_cached_forcing(self):
        return self.cached_forcing

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

        if self.cache_forcing:
            self.cached_forcing=s

        return s

class ZannaBolton2020Q(pyqg.parameterizations.QParameterization):
    r"""PV parameterization derived from equation discovery by `Zanna and
    Bolton 2020`_ (Eq. 6).
    .. _Zanna and Bolton 2020: https://doi.org/10.1029/2020GL088376

    Modified the implementation in pyqg/parameterizations.py to output
    Q subgrid forcing as opposed to velocities.
    """

    def __init__(self, constant=-46761284,cache_forcing=False):
        r"""
        Parameters
        ----------
        constant : number
            Scaling constant :math:`\kappa_{BC}`. Units: meters :sup:`-2`.
            Defaults to :math:`\approx -4.68 \times 10^7`, a value obtained by
            empirically minimizing squared error with respect to the subgrid
            forcing that results from applying the filtering method of `Guan et
            al. 2022`_ to a
            two-layer QGModel with default parameters.
            .. _Guan et al. 2022: https://doi.org/10.1016/j.jcp.2022.111090
        """

        self.constant = constant
        self.cache_forcing=cache_forcing
        self.cached_forcing=None
        
    def get_cached_forcing(self):
        return self.cached_forcing

    def __call__(self, m):
        # Compute ZB2020 basis functions
        uh = m.fft(m.u)
        vh = m.fft(m.v)
        vx = m.ifft(vh * m.ik)
        vy = m.ifft(vh * m.il)
        ux = m.ifft(uh * m.ik)
        uy = m.ifft(uh * m.il)
        rel_vort = vx - uy
        shearing = vx + uy
        stretching = ux - vy
        # Combine them in real space and take their FFT
        rv_stretch = m.fft(rel_vort * stretching)
        rv_shear = m.fft(rel_vort * shearing)
        sum_sqs = m.fft(rel_vort**2 + shearing**2 + stretching**2) / 2.0
        # Take spectral-space derivatives and multiply by the scaling factor
        kappa = self.constant
        du = kappa * m.ifft(m.ik*(sum_sqs - rv_shear) + m.il*rv_stretch)
        dv = kappa * m.ifft(m.il*(sum_sqs + rv_shear) + m.ik*rv_stretch)
        ## Take curl to convert to potential vorticity forcing
        dq = -m.ifft(m.l*1j*m.fft(du))+m.ifft(m.k*1j*m.fft(dv))
        if self.cache_forcing:
            self.cached_forcing=dq
        return dq

    def __repr__(self):
        return f"ZannaBolton2020(κ={self.constant:.2e})"


class BackscatterBiharmonic(pyqg.parameterizations.QParameterization):
    r"""PV parameterization based on `Jansen and Held 2014`_ and
    `Jansen et al.  2015`_ (adapted by Pavel Perezhogin). Assumes that a
    configurable fraction of Smagorinsky dissipation is scattered back to
    larger scales in an energetically consistent way.
    .. _Jansen and Held 2014: https://doi.org/10.1016/j.ocemod.2014.06.002
    .. _Jansen et al. 2015: https://doi.org/10.1016/j.ocemod.2015.05.007
    """

    def __init__(self, smag_constant=0.08, back_constant=0.99, eps=1e-32,cache_forcing=False):
        r"""
        Parameters
        ----------
        smag_constant : number
            Smagorinsky constant :math:`C_S` for the dissipative model.
            Defaults to 0.08.
        back_constant : number
            Backscatter constant :math:`C_B` describing the fraction of
            Smagorinsky-dissipated energy which should be scattered back to
            larger scales. Defaults to 0.99. Normally should be less than 1,
            but larger values may still be stable, e.g. due to additional
            dissipation in the model from numerical filtering.
        eps : number
            Small constant to add to the denominator of the backscatter formula
            to prevent division by zero errors. Defaults to 1e-32.
        """

        self.smagorinsky = pyqg.parameterizations.Smagorinsky(smag_constant)
        self.back_constant = back_constant
        self.eps = eps
        self.cache_forcing=cache_forcing
        self.cached_forcing=None

    def get_cached_forcing(self):
        return self.cached_forcing

    def __call__(self, m):
        lap = m.ik**2 + m.il**2
        psi = m.ifft(m.ph)
        lap_lap_psi = m.ifft(lap**2 * m.ph)
        dissipation = -m.ifft(lap * m.fft(lap_lap_psi * m.dx**2 * self.smagorinsky(m,
            just_viscosity=True)))
        backscatter = -self.back_constant * lap_lap_psi * (
            (np.sum(m.Hi * np.mean(psi * dissipation, axis=(-1,-2)))) /
            (np.sum(m.Hi * np.mean(psi * lap_lap_psi, axis=(-1,-2))) + self.eps)) 
        dq = dissipation + backscatter
        if self.cache_forcing:
            self.cached_forcing=dq
        return dq

    def __repr__(self):
        return f"BackscatterBiharmonic(Cs={self.smagorinsky.constant}, "\
                                     f"Cb={self.back_constant})"
