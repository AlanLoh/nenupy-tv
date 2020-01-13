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

from nenupytv.image import Grid


# ============================================================= #
# -------------------------- Dirty ---------------------------- #
# ============================================================= #
class Dirty(object):
    """
    """

    def __init__(self, grid):
        self.grid = grid
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
        if not isinstance(g, Grid):
            raise TypeError(
                'Grid object expected'
            )
        self._grid = g
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute(self):
        """
        """
        for p in range(self.grid.vis.shape[3]):
            vis = np.fft.ifftshift(self.grid.measurement[p, ...])
            self.dirty[p, ...] = np.fft.fftshift(np.fft.ifft2(vis))
        return


    def plot(self, **kwargs):
        """
        """
        import matplotlib.pyplot as plt
        image = np.abs(
            np.sqrt(self.dirty[0, :, :]**2. + self.dirty[3, :, :]**2)
        )
        im = plt.imshow(
            image,
            origin='lower',
            aspect='equal',
            cmap='YlGnBu_r',
            **kwargs
        )
        plt.colorbar(im)
        plt.xlabel('RA')
        plt.ylabel('Dec')
        return

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #



# ============================================================= #

