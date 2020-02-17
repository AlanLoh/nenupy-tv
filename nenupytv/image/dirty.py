#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Dirty'
    ]


import numpy as np

from nenupytv.image import Grid, Grid_Simple
from nenupytv.read import Crosslets
from nenupytv.astro import astro_image, eq_zenith


# ============================================================= #
# -------------------------- Dirty ---------------------------- #
# ============================================================= #
class Dirty(object):
    """
    """

    def __init__(self, grid, crosslets=None):
        self.grid = grid
        self.crosslets = crosslets
        self.dirty = np.zeros(
            grid.measurement.shape,
            dtype=grid.measurement.dtype
        )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def grid(self):
        return self._grid
    @grid.setter
    def grid(self, g):
        if not isinstance(g, (Grid, Grid_Simple)):
            raise TypeError(
                'Grid object expected'
            )
        self._grid = g
        return


    @property
    def crosslets(self):
        return self._crosslets
    @crosslets.setter
    def crosslets(self, c):
        if not isinstance(c, Crosslets):
            raise TypeError(
                "Expected a Crosslet object"
            )    
        self._crosslets = c
        return


    @property
    def i(self):
        """ Stokes I
        """
        return np.real(self.dirty[0, :, :] + self.dirty[3, :, :]) * 0.5


    @property
    def q(self):
        """ Stokes Q
        """
        return np.real(self.dirty[0, :, :] - self.dirty[3, :, :]) * 0.5

    @property
    def u(self):
        """ Stokes U
        """
        return np.real(self.dirty[1, :, :] + self.dirty[2, :, :]) * 0.5

    @property
    def v(self):
        """ Stokes V
        """
        return np.abs(-.5j*(self.dirty[1, :, :] - self.dirty[2, :, :]))
    


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute(self):
        """
        """
        for p in range(self.grid.measurement.shape[0]):
            vis = np.fft.ifftshift(self.grid.measurement[p, ...] * self.grid.meas_weights)
            self.dirty[p, ...] = np.fft.fftshift(np.fft.ifft2(vis))
        return


    def plot(self, stokes='I', sources=False, cbar=False, center=None, **kwargs):
        """
        """
        start = self._crosslets.time[0]
        stop = self._crosslets.time[-1]
        if center is None:
            ra, dec = eq_zenith(
                time=start + (stop - start)/2,
            )
        else:
            ra, dec = center
        resol = np.degrees(self.grid.resol.value)
        npix = self.grid.nsize
        
        # Stokes (0: XX, 1: XY, 2: YX, 3: YY)
        if stokes.lower() == 'i':
            image = self.i
        elif stokes.lower() == 'q':
            image = self.q
        elif stokes.lower() == 'u':
            image = self.u
        elif stokes.lower() == 'v':
            image = self.v
        
        astro_image(
            image=image,
            center=(ra, dec),
            npix=npix,
            resol=resol,
            time=start + (stop - start)/2,
            show_sources=sources,
            colorbar=cbar,
            **kwargs
        )
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #



# ============================================================= #

