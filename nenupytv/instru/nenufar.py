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


import numpy as np

from nenupytv.instru import RadioArray
from nenupytv.instru import ma_names, ma_positions, nenufar_pos, ma_positions_enu


# ============================================================= #
# -------------------------- NenuFAR -------------------------- #
# ============================================================= #
class NenuFAR(RadioArray):
    """ NenuFAR array object
    """

    def __init__(self, miniarrays=None):
        super().__init__(
            array_name='NenuFAR',
            ant_names=ma_names,
            ant_positions=ma_positions,
            array_position=nenufar_pos,
            antennas=miniarrays
        )

    @property
    def ma_enu(self):
        """ENU coordinates
        """
        return ma_positions_enu[self.antennas]


    @property
    def baseline_xyz(self):
        """
        """
        xyz = self.ant_positions[..., np.newaxis]
        xyz = xyz[:, :, 0][:, np.newaxis]
        xyz = xyz - xyz.transpose(1, 0, 2)
        bsl_xyz = xyz[np.triu_indices(self.antennas.size)]
        return bsl_xyz
    


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


# ============================================================= #

