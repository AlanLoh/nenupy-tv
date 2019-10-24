#! /usr/bin/python3
# -*- coding: utf-8 -*-


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

from nenupytv.instru import NenuFAR
from nenupytv.astro import lha


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
    def compute(self, ra, dec, freq, time):
        """
        """
        uvw = np.zeros(
            (self.bsl.shape[0], 3, 1) # .., nfreq
            )

        for k, (i, j) in enumerate(self.bsl):
            dpos = self.positions[i] - self.positions[j]
            xyz = self._celestial() * np.matrix(dpos).T
            temp = self._uvwplane(ra, dec, time) * xyz
            uvw[k, ...] = temp[:, 0]# * freq.ravel() / speedOfLight
        return uvw


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


    def _uvwplane(self, ra, dec, time):
        """  Transformation to uvw plane

            Parameters
            ----------
            ra : float
                Right Ascension in radians
            dec : float
                Declination in radians
            time : `astropy.time.Time`
                Time at which observation happened
        """
        ha = lha(
            time=time,
            location=self.instru.coord,
            ra=ra
            )
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
# ============================================================= #

