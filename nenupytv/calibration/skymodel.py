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
    'Skymodel'
    ]


import warnings
import numpy as np
try:
    from astroquery.ned import Ned
    NED_ENABLED = True
except:
    NED_ENABLED = False

from nenupytv.astro import ateam, radec


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
class Skymodel(object):
    """
    """

    def __init__(self, center, radius, freq):
        self.center = radec(ra=center[0], dec=center[1])
        self.radius = radius
        self.frequency = freq
        self.sources = self._ateam_in_fov()
        self.fluxes = self._ateam_flux()


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
