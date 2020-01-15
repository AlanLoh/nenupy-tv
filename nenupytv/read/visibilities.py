#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
"""


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2020, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Visibilities'
    ]


import numpy as np
from astropy.time import Time

from nenupytv.read import Crosslets
from nenupytv.uvw import SphUVW
from nenupytv.astro import eq_zenith


# ============================================================= #
# ----------------------- Visibilities ------------------------ #
# ============================================================= #
class Visibilities(object):
    """
    """

    def __init__(self, crosslets):
        self.time = None
        self.freq = None
        self.vis = None
        self.uvw = None
        self.cross = crosslets


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def cross(self):
        return self._cross
    @cross.setter
    def cross(self, c):
        if not isinstance(c, Crosslets):
            raise TypeError(
                'Crosslets object expected'
            )
        self._cross = c
        self._get_vis()
        self._compute_uvw()
        return


    @property
    def time(self):
        return self._time
    @time.setter
    def time(self, t):
        if t is None:
            pass
        elif not isinstance(t, Time):
            raise TypeError(
                'Time object expected'
            )
        else:
            if t.shape[0] != self.vis.shape[0]:
                raise ValueError(
                    'Time shape mismatch'
                )
        self._time = t
        return


    @property
    def freq(self):
        return self._freq
    @freq.setter
    def freq(self, f):
        if f is None:
            pass
        elif not isinstance(f, np.ndarray):
            raise TypeError(
                'np.ndarray object expected'
            )
        else:
            if f.shape[0] != self.vis.shape[1]:
                raise ValueError(
                    'freq shape mismatch'
                )
        self._freq = f
        return    


    @property
    def vis(self):
        return self._vis
    @vis.setter
    def vis(self, v):
        if v is None:
            pass
        elif not isinstance(v, np.ndarray):
            raise TypeError(
                'np.ndarray expected'
            )
        self._vis = v
        return


    @property
    def uvw(self):
        return self._uvw
    @uvw.setter
    def uvw(self, u):
        if u is None:
            pass
        elif not isinstance(u, np.ndarray):
            raise TypeError(
                'np.ndarray expected'
            )
        else:
            if not self.vis.shape[:-1] == u.shape[:-1]:
                raise ValueError(
                    'vis and uvw have shape discrepancies'
                )
        self._uvw = u
        return


    @property
    def phase_center(self):
        """ Phase center (time, (RA, Dec)) in degrees
        """
        return np.array(list(map(eq_zenith, self.time)))


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def calibrate(self):
        """
        """
        return


    def average(self):
        """
        """
        return


    def make_dirty(self):
        """
        """
        dirty = 0
        return dirty


    def make_image(self):
        """
        """
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_vis(self):
        """
        """
        self.vis = self._cross.reshape(
            fmean=False,
            tmean=False
        )
        self.time = self._cross.time
        self.freq = self._cross.meta['freq']
        return


    def _compute_uvw(self):
        """
        """
        uvw = SphUVW()
        uvw.from_crosslets(self._cross)
        self.uvw = uvw._uvw
        return
# ============================================================= #

