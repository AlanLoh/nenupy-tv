#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Crosslets'
    ]


import numpy as np
from os.path import abspath, isfile
from itertools import islice
from astropy.time import Time


# ============================================================= #
# ------------------------ Crosslets -------------------------- #
# ============================================================= #
class Crosslets(object):
    """
    """

    def __init__(self, xst_bin):
        self.meta = {}
        self.data = None
        self.time = None
        self.xst_bin = xst_bin


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def xst_bin(self):
        """ NenuFAR TV XST snapshot binary file
        """
        return self._xst_bin
    @xst_bin.setter
    def xst_bin(self, x):
        if not isinstance(x, str):
            raise TypeError(
                'String expected.'
                )
        x = abspath(x)
        if not isfile(x):
            raise FileNotFoundError(
                'Unable to find {}'.format(x)
                )
        self._xst_bin = x
        self._load()
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def correl_matrix(self, freq=None):
        """ Generator of correlation matrices for each time
            at a particular frequency.
        """
        n_ma = self.meta['ma'].size
        mat = np.zeros(
            (n_ma*2, n_ma*2),
            dtype='complex64'
            )
        indices = np.tril_indices(n_ma*2, 0)

        if freq is None:
            freq = self.meta['freq'][0]
        sb_idx = np.argmin(np.abs(freq - self.meta['freq']))
        
        for it in range(self.time.size):
            mat[indices] = self.data[it, sb_idx, :]
            yield mat


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _load(self):
        """ Read the binary file
        """
        # Extract the ASCII header (5 first lines)
        with open(self._xst_bin, 'rb') as f:
            header = list(islice(f, 0, 5))
        assert header[0] == b'HeaderStart\n',\
            'Wrong header start'
        assert header[-1] == b'HeaderStop\n',\
            'Wrong header stop'
        header = [s.decode('utf-8') for s in header]
        hd_size = sum([len(s) for s in header])

        # Parse informations into a metadata dictionnary
        keys = ['freq', 'ma', 'accu']
        search = ['Freq.List', 'Mr.List', 'accumulation']
        types = ['float64', 'int', 'int']
        for key, word, typ in zip(keys, search, types):
            for h in header:
                if word in h:
                    self.meta[key] = np.array(
                        h.split('=')[1].split(','),
                        dtype=typ
                        )

        # Deduce the dtype for decoding
        n_ma = self.meta['ma'].size
        n_sb = self.meta['freq'].size
        dtype = np.dtype(
            [('jd', 'float64'),
            ('data', 'complex64', (n_sb, n_ma*n_ma*2 + n_ma))]
            )

        # Decoding the binary file
        tmp = np.memmap(
            filename=self._xst_bin,
            dtype='int8',
            mode='r',
            offset=hd_size
            )
        decoded = tmp.view(dtype)

        self.data = decoded['data'] / self.meta['accu']
        self.time = Time(decoded['jd'], format='jd', precision=0)

        return
# ============================================================= #

