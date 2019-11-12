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


from nenupytv.instru import RadioArray
from nenupytv.instru import ma_names, ma_positions, nenufar_pos


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


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


# ============================================================= #

