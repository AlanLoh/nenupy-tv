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
from astropy.modeling import models, fitting

from nenupytv.image import Grid, Grid_Simple


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
        self.clean_beam = None


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


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute(self):
        """
        """
        # sampling = np.fft.ifftshift(self.grid.sampling)
        sampling = np.fft.ifftshift(self.grid.samp_weights)
        self.psf = np.fft.fftshift(np.fft.ifft2(sampling))
        self.psf /= self.psf.max()

        self.clean_beam = self._clean_beam()
        return


    def plot(self, **kwargs):
        """
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))
        im = plt.imshow(
            np.real(self.psf),
            origin='lower',
            aspect='equal',
            cmap='YlGnBu_r',
            **kwargs
        )
        # plt.colorbar(im)
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _clean_beam(self):
        """
        """
        psf = self.psf.real.copy()

        # Put most of the PSF to 0 to help the fit
        psf[psf <= np.std(psf)] = 0 
        nsize = int(psf.shape[0])

        fit_init = models.Gaussian2D(
            amplitude=1,
            x_mean=nsize/2,
            y_mean=nsize/2,
            x_stddev=0.2,
            y_stddev=0.2
        )

        fit_algo = fitting.LevMarLSQFitter()
        yi, xi = np.indices(psf.shape)

        gaussian = fit_algo(fit_init, xi, yi, psf)

        clean_beam = gaussian(xi, yi)
        clean_beam /= clean_beam.max()
        
        return clean_beam[
            int(nsize/2/2):int(nsize/2*3/2),
            int(nsize/2/2):int(nsize/2*3/2)
        ]



# ============================================================= #

