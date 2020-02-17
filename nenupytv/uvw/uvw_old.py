#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    Test :
    from nenupytv.uvw import UVW;  from astropy.time import Time, TimeDelta
    u = UVW()
    u.track(freq=70, tmin=Time('2019-06-18 17:10:00'), tmax=Time('2019-06-18 17:40:00'), dt=TimeDelta(60, format='sec'), ra=0, dec=90)
    
    TO DO:
    - method to triangularize self.uvw ang get a shape (30, 16, 820, 3)

"""


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'UVW',
    'SphUVW'
    ]


import numpy as np
import warnings
from astropy.time import Time, TimeDelta

from nenupytv.instru import RadioArray, NenuFAR
from nenupytv.astro import (
    lha,
    eq_zenith,
    rotz,
    wavelength,
    ref_location,
    ho_zenith,
    to_gal,
    to_altaz,
    rephase
)
from nenupytv.read import Crosslets


# ============================================================= #
# ---------------------------- UVW ---------------------------- #
# ============================================================= #
class UVW(object):
    """
    """

    def __init__(self, radioarray=NenuFAR()):
        self.instru = radioarray
        
        self.bsl = self.instru.baselines
        self.positions = self.instru.ant_positions
        self.lat = np.radians(self.instru.array_position.lat.deg)

        self.uvw = None
        self.wavel = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def x_uvw(self):
        """ UVW (symmetrical) ready, without autocorrelations
        """
        x, y = (self.instru.tri_x, self.instru.tri_y)
        xx = np.hstack((x[x!=y], y[x!=y]))
        yy = np.hstack((y[x!=y], x[x!=y]))
        return np.squeeze(self.uvw[:, :, xx, yy, :])


    @property
    def instru(self):
        """ Instrument instance
        """
        return self._instru
    @instru.setter
    def instru(self, i):
        if not isinstance(i, RadioArray):
            raise TypeError(
                'RadioArray object required'
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

            Returns
            -------
            self.uvw : `np.ndarray`
                Shape (time, freq, nant*nant, 3)
        """
        if (ra is None) and (dec is None):
            ra, dec = eq_zenith(
                time=time,
                location=self.instru.array_position
                )

        self.wavel = wavelength(freq)

        uvw = np.zeros(
            (1, self.wavel.size, self.bsl.shape[0], 3)
            )
        
        # Prepare the transformation matrices
        rot_cel = self._celestial()
        rot_uvw = self._uvwplane(
            time=time,
            ra=ra,
            dec=dec
            )

        # Initialize the UVW array
        uvw = np.zeros(
            (
                1,
                self.instru.n_ant,
                self.instru.n_ant,
                3
            )
        )

        # Perform UVW computation for each baseline and frequency
        xi = self.instru.tri_x
        yi = self.instru.tri_y
        dpos = self.positions[xi] - self.positions[yi]
        xyz = rot_cel * np.matrix(dpos).T
        uvw_k = rot_uvw * xyz
        uvw[0, xi, yi, :] = uvw_k.T
        uvw[0, yi, xi, :] = -uvw_k.T
        # uvw = uvw[:, np.newaxis, ...] /\
        #     self.wavel[:, np.newaxis, np.newaxis, np.newaxis]
        uvw = uvw[:, np.newaxis, ...]
        # Stack the UVW coordinates by time
        if self.uvw is None:
            self.uvw = uvw
        else:
            self.uvw = np.vstack((self.uvw, uvw))

        return


    def scan_times(self, times=None, tmin=None, tmax=None, dt=None):
        """ Function to return a time list that could be ingested
            by `self.track()`. Either give the list of times or
            a start and stop times as well as a time step.

            Parameters
            ----------
            times : `astropy.Time`
                List of times or a particular time
            tmin : `astropy.Time`
                Start time
            tmax : `astropy.Time`
                Stop time
            dt : `astropy.TimeDelta`
                Time step

            Returns
            -------
            time_list : `astropy.Time`
                List of times
        """
        if times is not None:
            if not isinstance(times, Time):
                times_list = Time(times)
            else:
                times_list = times
            # if not isinstance(times, Time):
            #     raise TypeError(
            #         '`times` should be a Time object'
            #         )
            # if hasattr(times, '__len__'):
            #     times_list = times
            # else:
            #     times_list = Time([times])
        elif (tmin is not None) & (tmax is not None) & (dt is not None):
            if not isinstance(tmin, Time):
                raise TypeError(
                    'tmin is not a Time object'
                    )
            if not isinstance(tmax, Time):
                raise TypeError(
                    'tmax is not a Time object'
                    )
            if not isinstance(dt, TimeDelta):
                raise TypeError(
                    'dt is not a TimeDelta object'
                    )
            period = tmax - tmin
            ntimes = int(period.sec // dt.sec)
            times_list = tmin + np.arange(ntimes) * dt

        else:
            raise AttributeError(
                'Either fill `times` or `tmin`, `tmax` and `dt`.'
                )

        return times_list


    def track(self, freq, times, ra=None, dec=None):
        """ Compute UV coordinates for a tracking
        """
        for time in times:
            self.compute(
                freq=freq,
                time=time,
                ra=ra,
                dec=dec
                )
        return


    def from_crosslets(self, cross, fmean=True, tmean=True):
        """ Compute the UVW coordinates from a
            `~nenupytv.read.Crosslets` instanciated from a 
            NenuFAR-TV observation file.

            :param cross:
                Cross-correlations read from a NenuFAR-TV binary
            :type cross: `~nenupytv.read.Crosslets`

            :returns: None
            :rtype: None
        """
        if not isinstance(cross, Crosslets):
            raise TypeError(
                "Expected a Crosslet object"
            )
        times = self.scan_times(
            times=cross.time
        )
        self.track(
            freq=cross.meta['freq'],
            times=times,
            ra=None,
            dec=None
        )
        if fmean:
            self.uvw = np.expand_dims(
                np.mean(self.uvw, axis=1),
                axis=1
            )
        if tmean:
            self.uvw = np.expand_dims(
                np.mean(self.uvw, axis=0),
                axis=0
            )
        return


    def plot(self, freq=None):
        """ Plot the UV distribution
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() first.'
                )

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = plt.subplots(figsize=(7, 7))

        u, v = self._uvplot(freq=freq)

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

        lim = 1.1*np.max(
                (np.abs(ax.get_xlim()).max(),
                np.abs(ax.get_ylim()).max())
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.show()
        plt.close('all')

        return


    def plot_radial(self, freq=None):
        """ Plot the radial cut ont the UV distribution
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() first.'
                )

        import matplotlib.pyplot as plt
        from astropy.modeling import models, fitting

        u, v = self._uvplot(freq=freq)
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


    def plot_azimuthal(self, freq=None):
        """ Plot the azimuthal cut ont the UV distribution
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() first.'
                )

        import matplotlib.pyplot as plt

        u, v = self._uvplot(freq=freq)
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
            location=self.instru.array_position,
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


    def _uvplot(self, freq=None):
        """ Return UV for plotting purposes
        """
        if freq is None:
            fid = 0
        else:
            fid = np.argmin(np.abs(self.wavel - wavelength(freq)))
        uvw_f = self.uvw[:, fid, :, :]
        # dont show autocorrelations
        mask = (uvw_f[..., 0] != 0.) & (uvw_f[..., 1] != 0.)
        u = uvw_f[..., 0][mask]
        v = uvw_f[..., 1][mask]
        # u = np.hstack((-u, u))
        # v = np.hstack((-v, v))
        return u, v
# ============================================================= #



# ============================================================= #
# -------------------------- SphUVW --------------------------- #
# ============================================================= #
class SphUVW(object):
    """
    """
    def __init__(self, radioarray=NenuFAR()):
        self.instru = radioarray
        self.bsl = radioarray.baseline_xyz
        self.obsloc = radioarray.array_position
        self.obsref = ref_location()
        self._uvw = None
        self.wavel = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def uvw(self):
        """ UVW coordinates
        """
        return self._uvw


    @property
    def uvw_sph(self):
        """ UVW coordinates converted to spherical coordinates
        """
        if self._uvw is None:
            return
        # r is just calculated for 1st time stamp
        r = np.sqrt(np.sum(np.square(self._uvw[0]), axis=1))
        r = np.repeat(r[np.newaxis], self._uvw.shape[0], axis=0)
        # Standard spherical transform
        r[r==0] = 1 # not to divide by 0
        theta = np.arccos(self._uvw[..., 2]/r)
        phi = np.arctan2(self._uvw[..., 1], self._uvw[..., 0]) + np.pi
        # Default values for autocorrelations
        auto = np.where(r==0.)
        theta[auto] = np.pi / 2.
        phi[auto] = np.pi
        # Same r values, assuming far field observation
        r = r[0]
        return np.dstack((r, theta, phi))


    @property
    def u_max(self):
        """
        """
        return np.max(np.abs(self._uvw[..., 0]))


    @property
    def v_max(self):
        """
        """
        return np.max(np.abs(self._uvw[..., 1]))


    @property
    def uvdist(self):
        """ UV distance in lambdas
        """
        return np.sqrt(np.sum(self._uvw**2, axis=-1))
    


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def wave_vector(self, frequency):
        """ Compute the wave vector k = 2 pi / lambda
        """
        k = 2. * np.pi / wavelength(frequency)
        return k


    def compute(self, times, coord=None):
        """ Compute UVW coordinates

            :param times:
                List of times in ISO or ISOT format
            :type times: list
        """
        ntimes = len(times)
        uvw_shape = [ntimes] + list(self.bsl.shape)
        uvw = np.zeros(uvw_shape)
        self.zen_gal = [None]*ntimes
        self.obs_t = [None]*ntimes

        for i, time in enumerate(times):
            self.obs_t[i] = Time(
                time,
                scale='utc',
                location=self.obsref
            )
            obs_t_loc = Time(
                time,
                scale='utc'
            )
            sid_rad = self.obs_t[i].sidereal_time('mean').radian
            
            # if coord is None:
            #     altaz = ho_zenith(
            #         time=obs_t_loc,
            #         location=self.obsloc)
            # else:
            #     altaz = to_altaz(coord)

            # self.zen_gal[i] = to_gal(altaz)

            si = np.sin(sid_rad)
            co = np.cos(sid_rad)
            rot = np.array([
                [ si, co, 0.],
                [-co, si, 0.],
                [ 0., 0., 1.]    
            ])
            uvw[i, ...] = np.dot(rot, self.bsl.T).T[np.newaxis]

            if coord is not None:
                transfo = rephase(
                    ra=coord[0],
                    dec=coord[1],
                    time=time,
                    loc=self.obsloc
                )
                # ra, dec = eq_zenith(
                #     time=time,
                #     location=self.obsloc
                #     )
                # if coord == 'zenith':
                #     coord = (ra, dec)
                # def rotMatrix(r, d):
                #     """ r: ra in radians
                #         d: dec in radians
                #     """
                #     w = np.array([
                #         [  np.sin(r)*np.cos(d) ,  np.cos(r)*np.cos(d) , np.sin(d) ]
                #     ]).T
                #     v = np.array([
                #         [ -np.sin(r)*np.sin(d) , -np.cos(r)*np.sin(d) , np.cos(d) ]
                #     ]).T
                #     u = np.array([
                #         [  np.cos(r)           , -np.sin(r)           , 0.        ]
                #     ]).T
                #     rot_matrix = np.concatenate([u, v, w], axis=-1)
                #     return rot_matrix
                # final_trans = rotMatrix(
                #     r=np.radians(coord[0]),
                #     d=np.radians(coord[1])
                # )
                # original_trans = rotMatrix(
                #     r=np.radians(ra),
                #     d=np.radians(dec)
                # )
                # total_trans = np.dot(final_trans.T, original_trans)
                uvw[i, ...] = np.dot(uvw[i, ...], transfo.T)

        # Reshape to get cross-correlation matrices
        xtri, ytri = np.triu_indices(self.instru.n_ant)
        self._uvw = np.zeros(
            (
                ntimes,
                1, # frequencies not filled for now
                self.instru.n_ant,
                self.instru.n_ant,
                3 # (U, V, W)
            )
        )
        self._uvw[:, :, xtri, ytri, :] = uvw[:, np.newaxis, :, :]
        self._uvw[:, :, ytri, xtri, :] = - self._uvw[:, :, xtri, ytri, :]
        # self._uvw = uvw[:, np.newaxis, :, :]
        return


    def uvw_lambda(self, frequencies):
        """
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() before.'
            )
        if self._uvw.shape[1] != 1:
            warnings.warn(
                'uvw_lambda() has previously been computed.'
            )
            return
        if isinstance(frequencies, (int, np.integer))\
            or isinstance(frequencies, (float, np.inexact)):
            frequencies = [frequencies]
        if not isinstance(frequencies, np.ndarray):
            frequencies = np.array(frequencies)
        self.wavel = wavelength(frequencies)
        self._uvw = self._uvw / self.wavel[
            np.newaxis,
            :,
            np.newaxis,
            np.newaxis,
            np.newaxis
        ]
        return


    def average(self):
        """
        """
        avg_uvw = np.mean(self._uvw, axis=(0, 1))
        return np.reshape(avg_uvw, [1, 1] + list(avg_uvw.shape))


    def from_crosslets(self, crosslets):
        """
        """
        if not isinstance(crosslets, Crosslets):
            raise TypeError(
                "Expected a Crosslet object"
            )
        self.compute(times=crosslets.time)
        self.uvw_lambda(frequencies=crosslets.meta['freq'])
        return


    def plot(self, freq=None):
        """ Plot the UV distribution
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() first.'
                )

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = plt.subplots(figsize=(10, 10))

        u, v = self._uvplot(freq=freq)

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
        ax.set_xlabel('u ($\\lambda$)')
        ax.set_ylabel('v ($\\lambda$)')

        lim = 1.1*np.max(
                (np.abs(ax.get_xlim()).max(),
                np.abs(ax.get_ylim()).max())
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.show()
        plt.close('all')

        return


    def plot_radial(self, freq=None):
        """ Plot the radial cut ont the UV distribution
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() first.'
                )

        import matplotlib.pyplot as plt
        from astropy.modeling import models, fitting

        u, v = self._uvplot(freq=freq)
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


    def plot_azimuthal(self, freq=None):
        """ Plot the azimuthal cut ont the UV distribution
        """
        if self.uvw is None:
            raise Exception(
                'Run compute() first.'
                )

        import matplotlib.pyplot as plt

        u, v = self._uvplot(freq=freq)
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
    def _uvplot(self, freq=None):
        """ Return UV for plotting purposes
        """
        if freq is None:
            fid = 0
        else:
            fid = np.argmin(np.abs(self.wavel - wavelength(freq)))
        uvw_f = self.uvw[:, fid, ...]
        # dont show autocorrelations
        mask = (uvw_f[..., 0] != 0.) & (uvw_f[..., 1] != 0.)
        u = uvw_f[..., 0][mask]
        v = uvw_f[..., 1][mask]
        #u = np.hstack((-u, u))
        #v = np.hstack((-v, v))
        return u, v
# ============================================================= #

