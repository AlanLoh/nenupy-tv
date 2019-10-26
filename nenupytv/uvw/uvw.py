#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
    Test :
    from nenupytv.uvw import UVW;  from astropy.time import Time, TimeDelta
    u = UVW()
    u.track(freq=70, tmin=Time('2019-06-18 17:10:00'), tmax=Time('2019-06-18 17:40:00'), dt=TimeDelta(60, format='sec'), ra=0, dec=90)
"""

__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'UVW'
    ]


import numpy as np
from astropy.constants import c as lspeed
from astropy import units as u

from nenupytv.instru import NenuFAR
from nenupytv.astro import lha, eq_zenith, rotz


# ============================================================= #
# ------------------------ Crosslets -------------------------- #
# ============================================================= #
class UVW(object):
    """
    """

    def __init__(self, NenuFAR_instance=NenuFAR()):
        self.instru = NenuFAR_instance
        
        self.bsl = self.instru.baselines
        self.positions = self.instru.pos
        self.lat = np.radians(self.instru.lat)

        self.uvw = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def instru(self):
        """ Instrument instance
        """
        return self._instru
    @instru.setter
    def instru(self, i):
        if not isinstance(i, NenuFAR):
            raise TypeError(
                'NenuFAR object required'
                )
        self._instru = i
        return


    @property
    def bsl(self):
        """ Basline array
        """
        return self._bsl
    @bsl.setter
    def bsl(self, b):
        if not isinstance(b, np.ndarray):
            raise TypeError(
                'Numpy array expected.'
                )
        if not all([len(bb)==2 for bb in b]):
            raise IndexError(
                'Some baseline tuples are not of length 2.'
                )
        self._bsl = b
        return


    @property
    def positions(self):
        """ Mini-Array positions
        """
        return self._positions
    @positions.setter
    def positions(self, p):
        if not isinstance(p, np.ndarray):
            raise TypeError(
                'Numpy array expected.'
                )
        if p.shape[1] != 3:
            raise ValueError(
                'Problem with position dimensions.'
                )
        self._positions = p
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute(self, freq, time, ra=None, dec=None):
        """ Compute UVW for a given time with an array phased
            towards (ra, dec).
            If ra and dec are None, local zenith is assumed.

            If uvw has already been calculated, the new values
            are stacked to the previous ones, as for a tracking.

            Parameters
            ----------
            ra : float
                Right Ascension in degrees
            dec : float
                Declination in degrees
            time : `astropy.time.Time`
                Time at which observation happened
            freq : float
                Frequency in MHz 
        """
        if (ra is None) and (dec is None):
            ra, dec = eq_zenith(
                time=time,
                location=self.instru.coord
                )

        freq *= u.MHz
        freq = freq.to(u.Hz)
        wavel = lspeed.value / freq.value

        uvw = np.zeros(
            (self.bsl.shape[0], 3, 1) # .., nfreq
            )
        
        rot_cel = self._celestial()
        rot_uvw = self._uvwplane(
            time=time,
            ra=ra,
            dec=dec
            )

        for k, (i, j) in enumerate(self.bsl):
            dpos = self.positions[i] - self.positions[j]
            xyz = rot_cel * np.matrix(dpos).T
            uvw_k = rot_uvw * xyz
            uvw[k, ...] = rotz(uvw_k.T, 90).T / wavel
        if self.uvw is None:
            self.uvw = uvw
        else:
            self.uvw = np.vstack((self.uvw, uvw))

        return


    def track(self, freq, tmin, tmax, dt, ra=None, dec=None):
        """ Compute UV corrdiantes for a tracking
        """
        period = tmax - tmin
        ntimes = int(period.sec // dt.sec)

        for i in range(ntimes):
            time = tmin + i * dt
            self.compute(
                freq=freq,
                time=time,
                ra=ra,
                dec=dec
                )

        return


    def plot(self):
        """ Plot the UV distribution
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() first.'
                )

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = plt.subplots(figsize=(7, 7))

        u, v = self._uvplot()

        hbins = ax.hexbin(
            x=u,
            y=v,
            C=None,
            cmap='YlGnBu',
            mincnt=1,
            bins='log',#None,
            gridsize=200,
            vmin=0.1,
            xscale='linear',
            yscale='linear',
            edgecolors='face',
            linewidths=0,
            vmax=None)

        ax.set_aspect('equal')
        ax.margins(0)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.15, pad=0.2)
        cb = fig.colorbar(hbins, cax=cax)
        
        cb.set_label('Histogram')
        # ax.set_title(title)
        ax.set_xlabel('u ($\\lambda$)')
        ax.set_ylabel('v ($\\lambda$)')

        lim = np.max(
                (np.abs(ax.get_xlim()).max(),
                np.abs(ax.get_ylim()).max())
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.show()
        plt.close('all')

        return


    def plot_radial(self):
        """ Plot the radial cut ont the UV distribution
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() first.'
                )

        import matplotlib.pyplot as plt
        from astropy.modeling import models, fitting

        u, v = self._uvplot()
        pos_v = v > 0.
        u = u[pos_v]
        v = v[pos_v]
        uv_dist = np.sqrt(u**2 + v**2)

        dist = []
        density = []
        ddist = 3 # lambda unit
        min_dists = np.arange(
            uv_dist.min(),
            uv_dist.max(),
            ddist
            )
        max_dists = np.arange(
            uv_dist.min() + ddist,
            uv_dist.max() + ddist,
            ddist
            )
        for min_dist, max_dist in zip(min_dists, max_dists):
            mask = (uv_dist>=min_dist) & (uv_dist<=max_dist)
            dist.append( np.mean([min_dist, max_dist]) )
            density.append( u[mask].size )
        density = np.array(density)/np.max(density)

        plt.bar(
            dist,
            height=density,
            width=ddist,
            edgecolor='black',
            linewidth=0.5
            )
        
        try:
            gaussian_init = models.Gaussian1D(
                amplitude=1.,
                mean=0,
                stddev=0.68 * max(dist),
                bounds={'mean': (0., 0.)}
                )
            fit_gaussian = fitting.LevMarLSQFitter()
            gaussian = fit_gaussian(gaussian_init, dist, density)
            gstd = gaussian.stddev.value

            x = np.linspace(min(dist), max(dist), 100)
            plt.plot(
                x,
                gaussian(x),
                linestyle=':',
                color='black',
                linewidth=1,
                label='HWHM = {:.2f} $\\lambda$'.format(gstd))
            plt.legend()
        except:
            pass

        plt.title('Radial profile')
        plt.xlabel('UV distance ($\\lambda$)')
        plt.ylabel('Density') 
        plt.show()

        return


    def plot_azimuthal(self):
        """ Plot the azimuthal cut ont the UV distribution
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() first.'
                )

        import matplotlib.pyplot as plt

        u, v = self._uvplot()
        pos_v = v > 0.
        u = u[pos_v]
        v = v[pos_v]
        uv_dist = np.sqrt(u**2 + v**2)
        uv_ang = np.degrees( np.arccos(u/uv_dist) )

        ang = []
        density = []
        dang = 5
        min_angs = np.arange(
            0,
            180,
            dang
            )
        max_angs = np.arange(
            0 + dang,
            180 + dang,
            dang
            )
        for min_ang, max_ang in zip(min_angs, max_angs):
            mask = (uv_ang>=min_ang) & (uv_ang<=max_ang)
            ang.append( np.mean([min_ang, max_ang]) )
            density.append( u[mask].size )
        plt.bar(
            ang,
            height=np.array(density)/np.max(density),
            width=dang,
            edgecolor='black',
            linewidth=0.5)
        plt.title('Azimuthal profile')
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Density') 
        plt.show()

        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _celestial(self):
        """ Transformation matrix to celestial pole
        """
        cos = np.cos(self.lat)
        sin = np.sin(self.lat)
        transfo = np.matrix([   
                [0, -sin, cos],
                [1,    0,   0],
                [0,  cos, sin]
            ])
        return transfo


    def _uvwplane(self, time, ra, dec):
        """  Transformation to uvw plane

            Parameters
            ----------
            ra : float
                Right Ascension in degrees
            dec : float
                Declination in degrees
            time : `astropy.time.Time`
                Time at which observation happened
        """
        ha = lha(
            time=time,
            location=self.instru.coord,
            ra=ra
            )

        ha = np.radians(ha)
        dec = np.radians(dec)

        sinr = np.sin(ha)
        cosr = np.cos(ha)
        sind = np.sin(dec)
        cosd = np.cos(dec)
        transfo = np.matrix([
                [      sinr,       cosr,    0],
                [-sind*cosr,  sind*sinr, cosd],
                [ cosd*cosr, -cosd*sinr, sind]
            ])
        return transfo


    def _uvplot(self):
        """ Return UV for plotting purposes
        """
        # dont show autocorrelations
        mask = (self.uvw[:, 0] != 0.) & (self.uvw[:, 1] != 0.)
        u = self.uvw[:, 0][mask]
        v = self.uvw[:, 1][mask]
        # u = np.hstack((-u, u))
        # v = np.hstack((-v, v))
        return u, v
# ============================================================= #

