#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Image'
    ]


import numpy as np
from numpy.fft import (
    fft2,
    ifft2,
    ifftshift,
    fftshift
)
from astropy.modeling import models, fitting

from nenupytv.image import Dirty
from nenupytv.response import PSF


# ============================================================= #
# --------------------------- Image --------------------------- #
# ============================================================= #
class Image(object):
    """
    """

    def __init__(self, dirty, psf):
        self.dirty = dirty
        self.psf = psf
        self.image = None

        self.residuals = self.dirty.copy()
        self.model = np.zeros(self.dirty.shape)


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def dirty(self):
        return self._dirty
    @dirty.setter
    def dirty(self, d):
        if not isinstance(d, Dirty):
            raise TypeError(
                'Dirty object expected'
            )
        if np.all(d.dirty == 0):
            d.compute()
        self._dirty = d.i
        return


    @property
    def psf(self):
        return self._psf
    @psf.setter
    def psf(self, p):
        if not isinstance(p, PSF):
            raise TypeError(
                'PSF object expected'
            )
        if np.all(p.psf == 0):
            p.compute() 
        self._psf = np.real(p.psf)
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def clean(self, gainfactor=0.05, niter=10000, threshold=None):
        """
        """
        nsize = int(self.residuals.shape[0])
        center_id = (
            int(nsize/2), int(nsize/2)
        )

        sig = []
        sigma = 10000 # huge value to start with
        n = 0
        while (n < niter) and (sigma > threshold):
            # find peak
            max_id = np.unravel_index(
                self.residuals.argmax(),
                self.residuals.shape
            )
            # flux to subtract
            fsub = self.residuals[max_id] * gainfactor
            # add a dirac with the flux to the model
            self.model[max_id] += fsub
            # shift the PSF
            psf_tmp = np.roll(
                self.psf, max_id[0] - center_id[0],
                axis=0
            )
            psf_tmp = np.roll(
                psf_tmp, max_id[1] - center_id[1],
                axis=1
            )
            psf_tmp = psf_tmp[
                int(nsize/2):int(nsize*3/2),
                int(nsize/2):int(nsize*3/2)
            ]
            # subtract the psf * flux to residuals
            self.residuals -= psf_tmp * fsub
            # compute the new std and keep track of it
            sigma = np.std(self.residuals)
            try:
                if sigma == sig[-1]:
                    break
            except IndexError:
                pass
            sig.append(sigma)
            n += 1

        # Convolve model and clean beam
        model_fft = ifftshift(fft2(fftshift(self.model)))
        clean_beam_fft = ifftshift(fft2(fftshift(self._clean_beam())))
        self.image = np.real(
            fftshift(ifft2(ifftshift(model_fft * clean_beam_fft)))
        ) + self.residuals

        return

    def plot(self, **kwargs):
        """
        """
        return

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _clean_beam(self):
        """
        """
        psf = self.psf.copy()

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
