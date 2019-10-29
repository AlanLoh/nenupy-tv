#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'NenuFAR'
    ]


from astropy import units as u
from astropy.coordinates import EarthLocation 
from itertools import product
import numpy as np

from nenupytv.instru import ma_names, ma_positions, ma_indices


# ============================================================= #
# ------------------------ Crosslets -------------------------- #
# ============================================================= #
class NenuFAR(object):
    """ NenuFAR array object
    """

    def __init__(self, miniarrays=None):
        self.lon = 47.375944 * u.deg
        self.lat = 2.193361 * u.deg
        self.height = 136.195 * u.m

        self.ma = miniarrays


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def ma(self):
        """ Active mini-arrays indices
        """
        return self._ma
    @ma.setter
    def ma(self, m):
        if m is None:
            self._ma = ma_indices
        else:
            try:
                self._ma = ma_indices[m]
            except:
                raise Exception(
                    'Something went wrong during ma selection.'
                    )
        return


    @property
    def coord(self):
        """ Coordinate object of NenuFAR

            Returns
            -------
            coord : `astropy.coordinates.EarthLocation`
                Coordinates of the whole NenuFAR array
        """
        return EarthLocation(
            lat=self.lat,
            lon=self.lon,
            height=self.height
            )


    @property
    def pos(self):
        """ Mini-array positions
        """
        return ma_positions[self.ma]


    @property
    def names(self):
        """ Names of the mini-arrays
        """
        return ma_names[self._ma]


    @property
    def baselines(self):
        """ Baselines computed for all active Mini-Arrays of 
            NenuFAR.

            Returns
            -------
            baselines : `np.ndarray`
                Array of baseline (length-2 tuples of antennae)
        """
        bsl = list(product(self.ma, repeat=2))
        return np.array(bsl)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot_array(self):
        """
        """
        import matplotlib.pyplot as plt
        plt.plot(self.pos[:, 0], self.pos[:, 1], marker='o', linestyle='')
        plt.show()
        return

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #

# ============================================================= #