#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'MiniArray'
    ]


from nenupytv.instru import RadioArray
from nenupytv.instru import ant_pos, nenufar_pos

import numpy as np


# ============================================================= #
# ------------------------- MiniArray ------------------------- #
# ============================================================= #
class MiniArray(RadioArray):
    """
    """

    def __init__(self, antennas=None):
        if antennas is None:
            antennas = np.arange(19)
        super().__init__(
            array_name='Mini-Array',
            ant_names=None,
            ant_positions=ant_pos,
            array_position=nenufar_pos,
            antennas=antennas
        )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #


# ============================================================= #
