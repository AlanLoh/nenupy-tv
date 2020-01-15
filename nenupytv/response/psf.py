#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'PSF'
    ]


import numpy as np

from nenupytv.image import Grid


# ============================================================= #
# ------------------------ Crosslets -------------------------- #
# ============================================================= #
class PSF(object):
    """
    """

    def __init__(self, grid):
        self.grid = grid
        self.psf = np.zeros(
            grid.sampling.shape,
            dtype=grid.sampling.dtype
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
        # sampling = np.fft.ifftshift(self.grid.sampling)
        sampling = np.fft.ifftshift(self.grid.samp_weights)
        self.psf = np.fft.fftshift(np.fft.ifft2(sampling))
        self.psf /= self.psf.max()
        return


    def plot(self, **kwargs):
        """
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(7, 7))
        im = plt.imshow(
            np.abs(self.psf),
            origin='lower',
            aspect='equal',
            cmap='YlGnBu_r',
            **kwargs
        )
        plt.colorbar(im)
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #

