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
    'Skymodel_old'
    ]


import warnings
from os.path import isfile
from urllib.request import urlopen
import numpy as np
try:
    from astroquery.ned import Ned
    NED_ENABLED = True
except:
    NED_ENABLED = False

from nenupytv.astro import ateam, radec, radec_hd


default = {
    'vir a': 3890.0,
    'cyg a': 22000.0,
    'cas a': 37200.0,
    'her a': 1840.0,
    'hyd a': 1802.1126213592233,
    'tau a': 2430.0
 }


# ============================================================= #
# ------------------------ Calibration ------------------------ #
# ============================================================= #
class Skymodel_old(object):
    """
    """

    def __init__(self, center, radius, freq, method='gsm', cutoff=100.):
        self.center = radec(ra=center[0], dec=center[1])
        self.radius = radius
        self.frequency = freq
        self.cutoff = cutoff # in Jy
        if method.lower() == 'manual':
            self.sources = self._ateam_in_fov()
            self.fluxes = self._ateam_flux()
            for key in self.sources['key']:
                if self.fluxes[key] < self.cutoff:
                    del self.sources[key]
                    del self.fluxes[key]
        elif method.lower() == 'gsm':
            self.gsmfile = ''
            self.sources, self.fluxes = self.from_gsm()


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def skymodel(self):
        """
        """
        skymodel = np.zeros(
            (len(self.sources), 3)
        )
        for i, key in enumerate(self.sources.keys()):
            skymodel[i, 0] = self.fluxes[key]
            skymodel[i, 1] = self.sources[key].ra.deg
            skymodel[i, 2] = self.sources[key].dec.deg
        return skymodel


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def save_from_gsm(self):
        """ Make a request and save the result in a file to
            speed up the reading time
        """


    def from_gsm(self):
        """ NEED TO EXTRAPOLATE AT THE RIGHT FREQUENCY
            CURRENTLY OK FOR 60 MHz
        """
        if not isfile(self.gsmfile):
            self.save_from_gsm()

        infield = {}
        command = [
            'https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?',
            'coord=' + str(self.center.ra.deg),
            ',' + str(self.center.dec.deg),
            '&radius=' + str(self.radius),
            '&cutoff=' + str(self.cutoff),
            '&unit=deg&deconv=y',
        ]
        url = ''.join(command)
        with urlopen(url) as response:
            response = response.read().decode('utf8')
            sources = response.split('\n')[3:]

        infield = {}
        fluxes = {}
        for source in sources:
            if source == '':
                continue
            src_desc = source.split(', ')
            src_name = src_desc[0]
            fluxes[src_name] = float(src_desc[4])
            src = radec_hd(
                src_desc[2] + ' ' + src_desc[3].replace('.', ':', 2)
            )
            infield[src_name] = radec(ra=src.ra.deg, dec=src.dec.deg)

            ref_freq = float(src_desc[8]) if src_desc[8] != '' else 60e6
            spec_index = float(src_desc[9].replace('[', '').replace(']', ''))
        
        return infield, fluxes


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _ateam_in_fov(self):
        """
        """
        at = ateam()
        infield = {}
        for key in at.keys():
            src_radec = radec(
                ra=at[key][0],
                dec=at[key][1]
            )
            sep = self.center.separation(src_radec).deg
            if sep <= self.radius:
                infield[key] = src_radec
        return infield


    def _ateam_flux(self):
        """
        """
        fluxes = {}
        for key in self.sources.keys():
            if not NED_ENABLED:
                warnings.warn(
                    (
                    'Astroquery not loaded. '
                    'Sky model for {} is taken by default '
                    'at 38 MHz'.format(key)
                    )
                )
                fluxes[key] = default[key]
                continue
            
            try:
                spectrum = Ned.get_table(
                    key if key != 'tau a' else 'crab',
                    table='photometry'
                )
            except:
                raise Exception(
                    'Unable to find {} on NED'.format(key)
                )
            spectrum = spectrum[spectrum['Frequency'] < 2.e9]
            from scipy.interpolate import interp1d
            interp_spec = interp1d(
                spectrum['Frequency']*1e-6,
                spectrum['Flux Density']
            )
            fluxes[key] = interp_spec(self.frequency) * 1
        return fluxes
# ============================================================= #
