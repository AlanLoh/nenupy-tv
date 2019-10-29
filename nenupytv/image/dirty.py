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


from nenupytv.image import Grid


# ============================================================= #
# -------------------------- Dirty ---------------------------- #
# ============================================================= #
class Dirty(object):
    """
    """

    def __init__(self, npix, csize):
        self.grid = None

        self.npix = npix
        self.csize = csize
        self.du = None
        self.dv = None
        self.umax = None
        self.umin = None


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def du(self):
        return self._du
    @du.setter
    def du(self, d):
        if d is None:
            if self.grid is None:
                self.grid = Grid(...)
        self._du = self.grid.du
        return


    @property
    def dv(self):
        return self._dv
    @dv.setter
    def dv(self, d):
        if d is None:
            if self.grid is None:
                self.grid = Grid(...)
        self._dv = self.grid.dv
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #



# ============================================================= #

