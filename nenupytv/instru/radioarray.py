#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'RadioArray'
    ]


from astropy.coordinates import EarthLocation 
from itertools import product
import numpy as np


# ============================================================= #
# ------------------------ RadioArray ------------------------- #
# ============================================================= #
class RadioArray(object):
    """
    """

    def __init__(
            self,
            array_name='',
            ant_names=None,
            ant_positions=None,
            array_position=None,
            antennas=None
        ):
        
        self.n_ant_total = 0
        self.n_ant = 0
        self.triux = None
        self.triuy = None
        self.tri_x = None
        self.tri_y = None
        
        self.array_name = array_name
        self.ant_positions = ant_positions
        self.ant_names = ant_names
        self.array_position = array_position
        self.antennas = antennas


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def array_position(self):
        return self._array_position
    @array_position.setter
    def array_position(self, p):
        if not isinstance(p, EarthLocation):
            raise TypeError(
                'EarthLocation object expected'
            )
        self._array_position = p


    @property
    def ant_positions(self):
        """ All antenna positions

            Parameters
            ----------
            positions : `np.ndarray`
                Array of shape (nant, 3), 3 being (x, y, z)

            Returns
            -------
            positions : `np.ndarray`
                Array of positions selected for active antennas
        """
        return self._ant_positions[self.antennas]
    @ant_positions.setter
    def ant_positions(self, a):
        if a is None:
            raise AttributeError(
                'ant_positions needs to be filled'
            )
        elif not isinstance(a, np.ndarray):
            raise TypeError(
                'ant_positions needs to be a numpy array'
            )
        elif len(a.shape) != 2:
            raise IndexError(
                'ant_positions shape legnth is greater than 2'
            )
        elif a.shape[1] != 3:
            raise IndexError(
                'ant_positions 2nd dimension should be 3 (x, y, z)'
            )
        else:
            pass

        self.n_ant_total = a.shape[0]
        self._ant_positions = a


    @property
    def ant_names(self):
        """ Antenna names
        """
        return self._ant_names[self.antennas]
    @ant_names.setter
    def ant_names(self, a):
        if a is None:
            a = np.arange(self.n_ant_total).astype(str)
        elif a.size != self.n_ant_total:
            raise IndexError(
                'ant_names and antenna number mismatch'
            )
        else:
            pass
        self._ant_names = a


    @property
    def antennas(self):
        """ Active antenna indices
        """
        return self._antennas
    @antennas.setter
    def antennas(self, a):
        if a is None:
            a = np.arange(self.n_ant_total)
        elif not hasattr(a, '__len__'):
            a = np.array([a])
        elif not isinstance(a, np.ndarray):
            a = np.array(a)
        else:
            pass

        if a.max() >= self.n_ant_total:
            raise IndexError(
                'Index greater than recorded antennas'
                )

        self._antennas = a
        self.n_ant = self._antennas.size
        self.tri_x, self.tri_y = np.tril_indices(
            self.n_ant,
            0
        )
        self.triux, self.triuy = np.triu_indices(
            self.n_ant
        )


    @property
    def baselines(self):
        """ Baselines computed for all antennas of the array.

            Returns
            -------
            baselines : `np.ndarray`
                Array of baseline (length-2 tuples of antennae)
        """
        bsl = list(product(self.antennas, repeat=2))
        return np.array(bsl)


    @property
    def ant1(self):
        """
        """
        ant1 = np.tile(self.antennas, (self.n_ant, 1))
        return ant1.astype(int)


    @property
    def ant2(self):
        """
        """
        ant2 = np.tile(self.antennas, (self.n_ant, 1)).T
        return ant2.astype(int)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot_array(self):
        """
        """
        import matplotlib.pyplot as plt
        plt.plot(
            self.ant_positions[:, 0],
            self.ant_positions[:, 1],
            marker='o',
            linestyle=''
        )
        plt.show()
        return

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #

