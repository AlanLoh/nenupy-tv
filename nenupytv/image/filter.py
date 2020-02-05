#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    This class is from:
    https://github.com/ratt-ru/fundamentals_of_interferometry/blob/master/5_Imaging/
"""


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'AAFilter'
    ]


import numpy as np


class AAFilter(object):
    """ Anti-Aliasing filter
        
        Keyword arguments for __init__:
        filter_half_support --- Half support (N) of the filter; the filter has a full support of N*2 + 1 taps
        filter_oversampling_factor --- Number of spaces in-between grid-steps (improves gridding/degridding accuracy)
        filter_type --- box (nearest-neighbour), sinc or gaussian_sinc
    """
    
    half_sup = 0
    oversample = 0
    full_sup_wo_padding = 0
    full_sup = 0
    no_taps = 0
    filter_taps = None
    

    def __init__(self,
        half_support=3,
        oversampling_factor=63,
        ftype='sinc'#'''ones',#'sinc',#'sinc'
        ):
        self.half_sup = half_support
        self.oversample = oversampling_factor
        self.full_sup_wo_padding = (half_support * 2 + 1)
        self.full_sup = self.full_sup_wo_padding + 2 #+ padding
        self.no_taps = self.full_sup + (self.full_sup - 1) * (oversampling_factor - 1)
        taps = np.arange(self.no_taps)/float(oversampling_factor) - self.full_sup / 2
        if ftype == "box":
            self.filter_taps = np.where(
                (taps >= -0.5) & (taps <= 0.5),
                np.ones([len(taps)]),
                np.zeros([len(taps)])
            )
        elif ftype == "sinc":
            self.filter_taps = np.sinc(taps) 
            #self.filter_taps = np.sinc(np.radians(taps))
        elif ftype == 'ones':
            self.filter_taps = np.ones(taps.size)
        elif ftype == "gaussian_sinc":
            alpha_1=1.55
            alpha_2=2.52
            self.filter_taps = np.sin(
                np.pi/alpha_1*(taps+1e-11))/(np.pi*(taps+1e-11))*np.exp(-(taps/alpha_2)**2
            )
        else:
            raise ValueError("Expected one of 'box','sinc' or 'gausian_sinc'")