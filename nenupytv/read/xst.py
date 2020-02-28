#! /usr/bin/python3
# -*- coding: utf-8 -*-


""" 
"""


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'XST'
    ]


from astropy.io import fits
import numpy as np
from os.path import isfile, abspath
from astropy.time import Time


# ============================================================= #
# ------------------------ Crosslets -------------------------- #
# ============================================================= #
class XST(object):
    """
    """

    def __init__(self, xstfile):
        self.xstfile = xstfile


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def xstfile(self):
        """
        """
        return self._xstfile
    @xstfile.setter
    def xstfile(self, x):
        x = abspath(x)
        if not isfile(x):
            raise FileNotFoundError(
                'Unable to find {}'.format(x)
            )
        self.ma = fits.getdata(
            x,
            ext=1,
            memmap=True
        )['noMROn'][0]
        self._nma = self.ma.size
        self._ma_idx = np.arange(self._nma)
        self._xi, self._yi = np.tril_indices(self._nma * 2, 0)
        data_tmp = fits.getdata(
            x,
            ext=7,
            memmap=True
        )
        self._time = data_tmp['jd']
        self._freq = data_tmp['xstsubband']
        self._data = data_tmp['data']
        #self._data = self._reshape(data_tmp['data'])

        self.meta = {
            'ma': self.ma,
            'freq': np.unique(self._freq) * 0.1953125
        }
        return


    @property
    def time(self):
        return Time(self._time, format='jd')
    


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def get(self, ma1, ma2, pol='XX', tidx=None):
        """ Select data per baseline and polarization.
        """
        if tidx is not None:
            if not isinstance(tidx, np.ndarray):
                tidx = np.array([tidx])
        else:
            tidx = np.arange(self._time.shape[0])
        pdict = {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3}

        cross_matrix = np.zeros(
            (self._nma * 2, self._nma * 2),
            dtype='bool'
        )
        if ma1 < ma2:
            # Asking for conjugate data
            auto = False
            tmp = ma1
            ma1 = ma2
            ma2 = tmp
        elif ma1 > ma2:
            auto = False
        else:
            auto = True
        
        mask = (self._xi//2 == ma1) & (self._yi//2 == ma2)
        
        cross_matrix[self._xi[mask], self._yi[mask]] = True
        bl_sel = cross_matrix[self._xi, self._yi].ravel()
        bl_idx = np.arange(bl_sel.size)[bl_sel]

        if auto:
            if pol=='XX':
                return self._data[tidx, :, bl_idx[0]]
            elif pol=='XY':
                return self._data[tidx, :, bl_idx[1]].conj()
            elif pol=='YX':
                return self._data[tidx, :, bl_idx[1]]
            elif pol=='YY':
                return self._data[tidx, :, bl_idx[2]]
            else:
                pass
        else:
            #print(self._data.shape, tidx, bl_idx[ pdict[pol] ])
            return self._data[tidx, :, bl_idx[ pdict[pol] ]]


    def beamform(self, ma1, ma2, part='re'):
        """
        """
        xx = self.get(ma1=ma1, ma2=ma2, pol='XX')
        xy = self.get(ma1=ma1, ma2=ma2, pol='XY')
        yy = self.get(ma1=ma1, ma2=ma2, pol='YY')
        if part.lower() == 're':
            return np.real(xx) + 2*np.real(xy) + np.real(yy)
        elif part.lower() == 'im':
            return np.imag(xx) + 2*np.imag(xy) + np.imag(yy)
        elif part.lower() == 'x':
            auto_x1 = self.get(ma1=ma1, ma2=ma1, pol='XX')
            auto_x2 = self.get(ma1=ma2, ma2=ma2, pol='XX')
            return np.real(auto_x1) + 2*np.real(xx) + np.real(auto_x2)
        elif part.lower() == 'y':
            auto_y1 = self.get(ma1=ma1, ma2=ma1, pol='YY')
            auto_y2 = self.get(ma1=ma2, ma2=ma2, pol='YY')
            return np.real(auto_y1) + 2*np.real(yy) + np.real(auto_y2)
        else:
            pass


    def reshape(self, tidx=None, fmean=False, tmean=False):
        """
        """
        if tidx is not None:
            if not isinstance(tidx, np.ndarray):
                tidx = np.array([tidx])
        reshaped_matrix = np.zeros(
            (
                self._time.shape[0] if tidx is None else tidx.size,
                self._freq.shape[1],
                self._nma,
                self._nma,
                4
            ),
            dtype='complex64'
        )
        for ma_i in range(self._nma):
            for ma_j in range(ma_i, self._nma):
                for pol_i, pol in enumerate(['XX', 'XY', 'YX', 'YY']):
                    # reshaped_matrix[..., ma_i, ma_j, pol_i] = self.get(
                    #         ma1=ma_i,
                    #         ma2=ma_j,
                    #         pol=pol,
                    #         tidx=tidx
                    #     )
                    reshaped_matrix[..., ma_j, ma_i, pol_i] = self.get(
                            ma1=ma_i,
                            ma2=ma_j,
                            pol=pol,
                            tidx=tidx
                        )
        if fmean:
            reshaped_matrix = np.expand_dims(
                np.mean(reshaped_matrix, axis=1),
                axis=1
                )
        if tmean:
            reshaped_matrix = np.expand_dims(
                np.mean(reshaped_matrix, axis=0),
                axis=0
            )
        return reshaped_matrix


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _reshape(self, data):
        """ From (time, freq, visib)
            to (time, freq, nant, nant, corr)
        """
        tmp = np.zeros(
            (
                self._time.size,
                self._freq.shape[1],
                self._nma*2,
                self._nma*2
            ),
            dtype='complex64'
        )
        tmp[..., self._xi, self._yi] = data
        return tmp



# ============================================================= #








