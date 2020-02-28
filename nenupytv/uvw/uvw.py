#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupytv'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'UVW'
    ]


import numpy as np
from astropy.time import Time

from nenupytv.instru import RadioArray, NenuFAR
from nenupytv.astro import eq_zenith, lha, wavelength


# ============================================================= #
# ---------------------------- UVW ---------------------------- #
# ============================================================= #
class UVW(object):
    """
    """

    def __init__(self, array=NenuFAR(), freq=None):
        self.uvw = None
        self.array = array
        self.bsl = array.baseline_xyz
        self.loc = array.array_position
        self.freq = freq

        self._celrot = self._cel_rot()


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def array(self):
        """ Instrument instance
        """
        return self._array
    @array.setter
    def array(self, a):
        if not isinstance(a, RadioArray):
            raise TypeError(
                'RadioArray object required'
                )
        self._array = a
        return


    @property
    def uvw(self):
        """
        """
        return np.ma.masked_array(
            self._uvw,
            mask=self.flag | self.selection
            )
    @uvw.setter
    def uvw(self, u):
        self._uvw = u
        if u is None:
            pass
        else:
            self.flag = np.zeros(
                u.shape,
                dtype=bool
            )
            self.selection = np.zeros(
                u.shape,
                dtype=bool
            )
        return


    @property
    def freq(self):
        """ Frequency in MHz
        """
        return self._freq
    @freq.setter
    def freq(self, f):
        if f is not None:
            if isinstance(f, np.ndarray):
                pass
            elif isinstance(f, list):
                f = np.array(f)
            else:
                f = np.array([f])
            self._freq = f
            self.wavel = wavelength(f)
        else:
            self._freq = None
        return


    @property
    def uvw_wave(self):
        """ UVW expressed in wavelengths
        """
        if not hasattr(self, 'wavel'):
            raise AttributeError(
                'Set the frequency in MHz first.'
            )
        return self.uvw / self.wavel[
            np.newaxis,
            :,
            np.newaxis,
            np.newaxis,
            np.newaxis
        ]


    @property
    def u_max(self):
        """
        """
        return np.min(np.abs(self.uvw[..., 0]))


    @property
    def v_max(self):
        """
        """
        return np.min(np.abs(self.uvw[..., 1]))


    @property
    def u_max(self):
        """
        """
        return np.max(np.abs(self.uvw[..., 0]))


    @property
    def v_max(self):
        """
        """
        return np.max(np.abs(self.uvw[..., 1]))


    @property
    def uwave_min(self):
        """
        """
        return np.min(np.abs(self.uvw_wave[..., 0]))


    @property
    def vwave_min(self):
        """
        """
        return np.min(np.abs(self.uvw_wave[..., 1]))


    @property
    def uwave_max(self):
        """
        """
        return np.max(np.abs(self.uvw_wave[..., 0]))


    @property
    def vwave_max(self):
        """
        """
        return np.max(np.abs(self.uvw_wave[..., 1]))


    @property
    def uvwdist(self):
        """ UVW distance in meters
        """
        return np.sqrt(np.sum(self.uvw**2, axis=-1))


    @property
    def uvdist(self):
        """ UV distance in meters
        """
        return np.sqrt(np.sum(self.uvw[..., :-1]**2, axis=-1))


    @property
    def uvwdist_wave(self):
        """ UVW distance in wavelenghts
        """
        return np.sqrt(np.sum(self.uvw_wave**2, axis=-1))


    @property
    def uvdist_wave(self):
        """ UV distance in wavelengths
        """
        return np.sqrt(np.sum(self.uvw_wave[..., :-1]**2, axis=-1))


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute(self, time, ra=None, dec=None):
        """
        """
        time = self._get_time(time)

        # Prepare UVW array
        self.uvw = np.zeros(
            (
                time.size,
                self.freq.size,
                self.array.n_ant,
                self.array.n_ant,
                3
            )
        )

        #xyz = self._celrot * np.matrix(self.bsl).T
        xyz = np.matrix(self.bsl).T

        # Loop over time
        for i, ti in enumerate(time):
            # Check if a RA/Dec is specified,
            # otherwise compute at local zenith
            if (ra is None) and (dec is None):
                ra_p, dec_p = eq_zenith(
                    time=ti,
                    location=self.loc
                )
            else:
                ra_p = ra
                dec_p = dec

            # Computation of UVW
            ha = lha(
                time=ti,
                location=self.loc,
                ra=ra_p
            )
            ha = np.radians(ha)
            dec_p = np.radians(dec_p)
            sr = np.sin(ha)
            cr = np.cos(ha)
            sd = np.sin(dec_p)
            cd = np.cos(dec_p)
            rot_uvw = np.matrix([
                [    sr,     cr,  0],
                [-sd*cr,  sd*sr, cd],
                [ cd*cr, -cd*sr, sd]
            ])

            uvw_k = rot_uvw * xyz

            # Fill at the right time
            self.uvw[i, 0, self.array.triux, self.array.triuy, :] = -uvw_k.T
            self.uvw[i, 0, self.array.triuy, self.array.triux, :] = uvw_k.T

        # Fill in identical UVW (in m) for other frequencies
        for j in range(1, self.freq.size):
            self.uvw[:, j, ...] = self.uvw[:, 0, ...].copy()

        self._flag_auto()

        return


    def uvcoverage(self, wave=False, **kwargs):
        """ Plot the UV distribution
        """
        self._selection(**kwargs)

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = plt.subplots(figsize=(10, 10))

        hbins = ax.hexbin(
            x=self.uvw_wave[..., 0].ravel() if wave else self.uvw[..., 0].ravel(),
            y=self.uvw_wave[..., 1].ravel() if wave else self.uvw[..., 1].ravel(),
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
        ax.set_xlabel(
            'u ({})'.format('$\\lambda$' if wave else 'm')
        )
        ax.set_ylabel(
            'v ({})'.format('$\\lambda$' if wave else 'm')
        )

        lim = 1.1*np.max(
                (np.abs(ax.get_xlim()).max(),
                np.abs(ax.get_ylim()).max())
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.show()
        plt.close('all')

        return

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_time(self, time):
        """
        """
        if isinstance(time, str):
            time = Time([time])
        elif isinstance(time, list):
            time = Time(time)
        elif isinstance(time, Time):
            pass
        else:
            raise TypeError(
                'Wrong time format.'
            )
        return time


    def _cel_rot(self):
        """ Unused
        """
        lat = self.loc.lat.rad
        c = np.cos(lat)
        s = np.sin(lat)
        rot = np.matrix([   
            [0, -s, c],
            [1,  0, 0],
            [0,  c, s]
        ])
        return rot


    def _flag_auto(self):
        """
        """
        mask = self.uvdist == 0
        mask = np.tile(mask[..., np.newaxis], 3)
        self.flag[mask] = True
        return


    def _selection(self, **kwargs):
        """
        """
        # Reinitialization
        self.selection[...] = False

        for key, value in kwargs.items():
            if key == 'ant':
                if value is None:
                    continue
                mask = ~np.isin(
                    element=self.array.ant1,
                    test_elements=value)
                self.selection[:, :, mask, :] = True

                mask = ~np.isin(
                    element=self.array.ant2,
                    test_elements=value)
                self.selection[:, :, mask, :] = True

            if key == 'uvdist_min':
                if value is None:
                    continue
                if value <= self.uvdist.min():
                    raise ValueError(
                        'Need to select uvdist_min > {} m'.format(
                            self.uvdist.min()
                        )
                    )
                if value >= self.uvdist.max():
                    raise ValueError(
                        'Need to select uvdist_min < {} m'.format(
                            self.uvdist.max()
                        )
                    )
                mask = self.uvdist <= value
                self.selection[mask] = True

            if key == 'uvdist_max':
                if value is None:
                    continue
                if value <= self.uvdist.min():
                    raise ValueError(
                        'Need to select uvdist_max > {} m'.format(
                            self.uvdist.min()
                        )
                    )
                if value >= self.uvdist.max():
                    raise ValueError(
                        'Need to select uvdist_max < {} m'.format(
                            self.uvdist.max()
                        )
                    )
                mask = self.uvdist >= value
                self.selection[mask] = True

            if key == 'uvwave_min':
                if value is None:
                    continue
                if value <= self.uvdist_wave.min():
                    raise ValueError(
                        'Need to select uvwave_min > {} lambda'.format(
                            self.uvdist_wave.min()
                        )
                    )
                if value >= self.uvdist_wave.max():
                    raise ValueError(
                        'Need to select uvwave_min < {} lambda'.format(
                            self.uvdist_wave.max()
                        )
                    )
                mask = self.uvdist_wave <= value
                self.selection[mask] = True

            if key == 'uvwave_max':
                if value is None:
                    continue
                if value <= self.uvdist_wave.min():
                    raise ValueError(
                        'Need to select uvwave_max > {} lambda'.format(
                            self.uvdist_wave.min()
                        )
                    )
                if value >= self.uvdist_wave.max():
                    raise ValueError(
                        'Need to select uvwave_max < {} lambda'.format(
                            self.uvdist_wave.max()
                        )
                    )
                mask = self.uvdist_wave >= value
                self.selection[mask] = True

            if key == 'freq_min':
                if value is None:
                    continue
                if value <= self.freq.min():
                    raise ValueError(
                        'Need to select freq_min > {} MHz'.format(
                            self.freq.min()
                        )
                    )
                if value >= self.freq.max():
                    raise ValueError(
                        'Need to select freq_min < {} MHz'.format(
                            self.freq.max()
                        )
                    )
                mask = self.freq <= value
                self.selection[:, mask, ...] = True

            if key == 'freq_max':
                if value is None:
                    continue
                if value <= self.freq.min():
                    raise ValueError(
                        'Need to select freq_max > {} MHz'.format(
                            self.freq.min()
                        )
                    )
                if value >= self.freq.max():
                    raise ValueError(
                        'Need to select freq_max < {} MHz'.format(
                            self.freq.max()
                        )
                    )
                mask = self.freq >= value
                self.selection[:, mask, ...] = True
        return



# ============================================================= #

