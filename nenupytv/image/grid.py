#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    TO DO:
    map to healpix grid
"""


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Grid'
    ]


import healpy as hp
import numpy as np


# ============================================================= #
# --------------------------- Grid ---------------------------- #
# ============================================================= #
class Grid(object):
    """
    """

    def __init__(self, resolution):
        self.nside = None
        self.resol = resolution
        self.ra, self.dec = self._hpx_radec()


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def resol(self):
        """ Resolution of the cell in degrees
        """
        return self._resol
    @resol.setter
    def resol(self, r):
        nsides = 2**np.arange(1, 12)
        resol_rad = hp.nside2resol(
            nside=nsides,
            arcmin=False
            )
        resol_deg = np.degrees(resol_rad)
        idx = (np.abs(resol_deg - r)).argmin()
        self._resol = resol_deg[idx]
        self.nside = nsides[idx]
        return


    @property
    def nside(self):
        return self._nside
    @nside.setter
    def nside(self, n):
        if n is None:
            self._nside = None
        else:
            self.npix = hp.nside2npix(n)
            self._nside = n
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _hpx_radec(self):
        """
        """
        ra, dec = hp.pix2ang(
            nside=self.nside,
            ipix=np.arange(self.npix),
            lonlat=True,
            nest=False
            )
        return ra, dec

# ============================================================= #



