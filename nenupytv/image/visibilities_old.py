#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
"""


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2020, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Visibilities'
    ]


import numpy as np
from astropy.time import Time

from nenupytv.read import Crosslets
from nenupytv.uvw import SphUVW
from nenupytv.astro import eq_zenith, to_lmn, rephase, nenufar_loc
from nenupytv.calibration import Skymodel
from nenupytv.image import Grid_Simple, Dirty


# ============================================================= #
# ----------------------- Visibilities ------------------------ #
# ============================================================= #
class Visibilities(object):
    """
    """

    def __init__(self, crosslets):
        self.flag = None
        self.time = None
        self.freq = None
        self.cal_vis = None
        self.vis = None
        self.uvw = None
        self.grid = None
        self.cross = crosslets


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def cross(self):
        return self._cross
    @cross.setter
    def cross(self, c):
        if isinstance(c, str):
            c = Crosslets(c)
        if not isinstance(c, Crosslets):
            raise TypeError(
                'Crosslets object expected'
            )
        self._cross = c
        self._get_vis()
        self._compute_uvw()
        return


    @property
    def time(self):
        return self._time
    @time.setter
    def time(self, t):
        if t is None:
            pass
        elif not isinstance(t, Time):
            raise TypeError(
                'Time object expected'
            )
        else:
            if t.shape[0] != self.vis.shape[0]:
                raise ValueError(
                    'Time shape mismatch'
                )
        self._time = t
        return


    @property
    def freq(self):
        return self._freq
    @freq.setter
    def freq(self, f):
        if f is None:
            pass
        elif not isinstance(f, np.ndarray):
            raise TypeError(
                'np.ndarray object expected'
            )
        else:
            if f.shape[0] != self.vis.shape[1]:
                raise ValueError(
                    'freq shape mismatch'
                )
        self._freq = f
        return    


    @property
    def vis(self):
        if self.flag is None:
            self.flag = np.zeros(
                self._vis.shape[:-1],
                dtype=bool
            )
        return np.ma.masked_array(
            self._vis,
            mask=np.tile(np.expand_dims(self.flag, axis=4), 4)
            )
    @vis.setter
    def vis(self, v):
        if v is None:
            pass
        elif not isinstance(v, np.ndarray):
            raise TypeError(
                'np.ndarray expected'
            )
        self._vis = v
        return


    @property
    def uvw(self):
        if self.flag is None:
            self.flag = np.zeros(
                self._uvw.shape[:-1],
                dtype=bool
            )
        return np.ma.masked_array(
            self._uvw,
            mask=np.tile(np.expand_dims(self.flag, axis=4), 3)
            )
    @uvw.setter
    def uvw(self, u):
        if u is None:
            pass
        elif not isinstance(u, np.ndarray):
            raise TypeError(
                'np.ndarray expected'
            )
        else:
            if not self.vis.shape[:-1] == u.shape[:-1]:
                raise ValueError(
                    'vis and uvw have shape discrepancies'
                )
        self._uvw = u
        return


    @property
    def phase_center(self):
        """ Phase center (time, (RA, Dec)) in degrees
        """
        return np.array(list(map(eq_zenith, self.time)))


    @property
    def time_mean(self):
        """
        """
        dt = self.time[-1] - self.time[0]
        return self.time[0] + dt/2


    @property
    def freq_mean(self):
        """
        """
        return np.mean(self.freq)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def uvcut(self, uvmin=None, uvmax=None):
        """
        """
        if uvmin is None:
            uvmin = self.uvdist.min()
        if uvmax is None:
            uvmax = self.uvdist.max()
        self.flag = (self.uvdist < uvmin) | (self.uvdist > uvmax)
        return


    def calibrate(self):
        """
        """
        # We search for sources around 35 deg of the zenith
        # no need to be very precise as they are fixed (RA, Dec)
        sk = Skymodel(
            center=eq_zenith(self.time_mean),
            radius=35,
            freq=self.freq_mean,
            method='gsm',
            cutoff=150
        )

        # The sky model does not contain polarization!
        model_vis = self._model_vis(sk.skymodel)

        # glm, Glm = _create_G_LM(self.vis, model_vis)
        # self.cal_vis = Glm**(-1) * self.vis

        self.vis = self.vis[:, 0, :, :, 0]
        model_vis = model_vis[:, 0, :, :, 0]
        gains = self._gain_cal(model_vis)
        self.cal_vis = gains**(-1) * self.vis
        return


    def average(self):
        """
        """
        return


    def make_dirty(self, fov=60, robust=-2, coord=None):
        """
        """
        avg_vis = np.mean(self.vis, axis=(0, 1))
        avg_uvw = np.mean(self.uvw, axis=(0, 1))

        if coord is not None:
            transfo, origtransfo, finaltransfo, dw = rephase(
                ra=coord[0],
                dec=coord[1],
                time=self.time_mean,
                loc=nenufar_loc(),
                dw=True
            )
            # phase = np.dot( avg_uvw, np.dot( dw.T, origtransfo).T)
            # dphi = np.exp( phase * 2 * np.pi * 1j)# / wavelength[idx1:idx2, chan])
            # avg_vis *= dphi
            # avg_uvw = np.dot(avg_uvw, transfo.T)

            avg_uvw = np.dot(avg_uvw, origtransfo.T)#finaltransfo.T)
            avg_vis *= np.exp( np.dot(avg_uvw, -dw) * 2 * np.pi * 1j)

        self.grid = Grid_Simple(
            vis=avg_vis,
            uvw=avg_uvw,
            freq=self.freq_mean,
            fov=fov,
            robust=robust,
            convolution=None # 'gaussian'
        )
        self.grid.populate()

        dirty = Dirty(self.grid, self.cross)
        dirty.compute()
        return dirty


    def make_image(self):
        """
        """
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_vis(self):
        """
        """
        self.vis = self._cross.reshape(
            fmean=False,
            tmean=False
        )
        self.time = self._cross.time
        self.freq = self._cross.meta['freq']
        return


    def _model_vis(self, skymodel):
        """
        """
        vis_model = np.zeros(
            self.vis.shape,
            dtype='complex'
        )

        # compute the zenith coordinates for every time step
        zen = self.phase_center #np.array(list(map(eq_zenith, self.time)))
        ra_0 = zen[:, 0]
        dec_0 = zen[:, 1]

        # pointers to u, v, w coordinates
        u = self.uvw[..., 0]
        v = self.uvw[..., 1]
        w = self.uvw[..., 2]

        # loop over skymodel sources
        na = np.newaxis
        for k in range(skymodel.shape[0]):
            flux = skymodel[k, 0] # Jy
            ra, dec = skymodel[k, 1], skymodel[k, 2]
            l, m, n = to_lmn(ra, dec, ra_0, dec_0)
            ul = u*l[:, na, na, na]
            vm = v*m[:, na, na, na]
            nw = (n[:, na, na, na] - 1)*w
            phase = np.exp(-2*np.pi*1j*(ul + vm))# + nw))
            # adding the w component mess up with subsequent plots
            vis_model += flux * phase[..., na]
        return vis_model


    def _gain_cal(self, model):
        """
        """
        from scipy.optimize import least_squares #leastsq
        
        gains = np.zeros(
            self.vis.shape,
            dtype='complex'
        )

        def err_func(gain, data, model):
            shape = self.vis.shape[1:]
            gain = np.reshape(gain, shape)
            data = np.reshape(data, shape)
            model = np.reshape(model, shape)

            calmodel = gain * model
            calmodel = calmodel * gain.conj()

            # scipy optimize doesn't like complex numbers
            a = (data - calmodel).ravel()
            return a.real**2 + a.imag**2

        for t in range(self.time.size):
            print(t)
            # res = leastsq(
            #     err_func,
            #     np.ones(
            #         self.vis[t, ...].size,
            #     ),
            #     args=(self.vis[t, ...].ravel(), model[t, ...].ravel())
            # )
            res = least_squares(
                err_func,
                np.ones(
                    self.vis[t, ...].size,
                ),
                args=(self.vis[t, ...].ravel(), model[t, ...].ravel()),
                verbose=2
            )
            gains[t, ...] = res.x # res

        return gains

    # def _create_G_LM(self, D, M):
    #     """ This function finds argmin G ||D-GMG^H|| using Levenberg-Marquardt.
    #         It uses the optimize.leastsq scipy to perform
    #         the actual minimization.
    #         D/self.vis is your observed visibilities matrx.
    #         M is your predicted visibilities.
    #         g the antenna gains.
    #         G = gg^H.
    #     """
    #     from scipy.optimize import leastsq
    #     def err_func(g, d, m):
    #         """ Unpolarized direction independent calibration entails
    #             finding the G that minimizes ||R-GMG^H||. 
    #             This function evaluates D-GMG^H.
    #             g is a vector containing the real and imaginary components of the antenna gains.
    #             d is a vector containing a vecotrized R (observed visibilities), real and imaginary.
    #             m is a vector containing a vecotrized M (predicted), real and imaginary.
    #             r is a vector containing the residuals.
    #         """
    #         Nm = len(d)//2
    #         N = len(g)//2
            
    #         G = np.diag(g[0:N] + 1j*g[N:])
            
    #         D = np.reshape(d[0:Nm],(N,N)) + np.reshape(d[Nm:],(N,N))*1j #matrization
    #         M = np.reshape(m[0:Nm],(N,N)) + np.reshape(m[Nm:],(N,N))*1j
            
    #         T = np.dot(G, M)
    #         T = np.dot(T, G.conj())
            
    #         R = D - T
    #         r_r = np.ravel(R.real) #vectorization
    #         r_i = np.ravel(R.imag)
    #         r = np.hstack([r_r, r_i])
    #         return r

    #     nant = D.shape[0] #number of antennas
    #     temp = np.ones(
    #         (nant, nant), # MAYBE FALSE CHECK D.SHAPE[1]
    #         dtype='complex'
    #     )
    #     G = np.zeros(
    #         D.shape, #(ant,ant,time)
    #         dtype='complex'
    #     )
    #     g = np.zeros(
    #         (self.time.size, nant),
    #         dtype='complex'
    #     )
        
    #     # perform calibration per time-slot
    #     for t in range(self.time.size):
    #         g_0 = np.ones((2*nant)) # first antenna gain guess 
    #         g_0[nant:] = 0
    #         d_r = np.ravel(D[t, ...].real) #vectorization of observed + seperating real and imag
    #         d_i = np.ravel(D[t, ...].imag)
    #         d = np.hstack([d_r,d_i])
    #         m_r = np.ravel(M[t, ...].real) #vectorization of model + seperating real and imag
    #         m_i = np.ravel(M[t, ...].imag)
    #         m = np.hstack([m_r, m_i])
    #         g_lstsqr_temp = leastsq(
    #             err_func,
    #             g_0,
    #             args=(d, m)
    #         )
    #         g_lstsqr = g_lstsqr_temp[0]          
               
    #         G_m = np.dot(np.diag(g_lstsqr[0:nant] + 1j*g_lstsqr[nant:]), temp)
    #         G_m = np.dot(G_m, np.diag((g_lstsqr[0:nant] + 1j*g_lstsqr[nant:]).conj()))           

    #         g[t, :] = g_lstsqr[0:nant] + 1j*g_lstsqr[nant:] #creating antenna gain vector       
    #         G[t, ...] = G_m
             
    #     return g, G


    def _compute_uvw(self):
        """
        """
        uvw = SphUVW()
        uvw.from_crosslets(self._cross)
        self.uvw = uvw._uvw
        self.uvdist = uvw.uvdist
        return
# ============================================================= #

