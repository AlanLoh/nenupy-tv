#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'ma_names',
    'ma_positions',
    'ma_indices'
    ]


import numpy as np
from os.path import dirname, join

from nenupytv.astro import rotz


location_file  = join(
    dirname(__file__),
    'location_miniarrays.dat'
    )

def parse_file(lfile):
    """ Parse the location_miniarrays.dat file

        Returns
        -------
        ma : dict
            Dictionnary of mini-array indices and positions
    """
    ma_dict = {
        'DBCODE': [],
        'X': [],
        'Y': [],
        'Z': []
        }

    with open(lfile) as f:
        line = f.readline()
        while line:
            for key in ma_dict.keys():
                search = key + '='
                if search in line:
                    val = line.split(search)[1].split()[0]
                    ma_dict[key].append(val)
            line = f.readline()

    positions = np.zeros((len(ma_dict['DBCODE']), 3))
    positions[:, 0] = np.array(ma_dict['X'], dtype='float64')
    positions[:, 1] = np.array(ma_dict['Y'], dtype='float64')
    positions[:, 2] = np.array(ma_dict['Z'], dtype='float64')

    names = np.array(
        [name.strip("'") for name in ma_dict['DBCODE']],
        dtype='str'
        )

    indices = np.array(
        [ma.replace('MR', '') for ma in names],
        dtype='int'
        )

    return rotz(positions, 90), names, indices


ma_positions, ma_names, ma_indices = parse_file(location_file)

