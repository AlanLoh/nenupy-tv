#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'SphMap',
]


import numpy as np
from healpy.sphtfunc import (
    Alm,
    alm2map,
    map2alm
)
from healpy.rotator import Rotator
from healpy.pixelfunc import (
    nside2npix,
    pix2ang   
)


# ============================================================= #
# -------------------------- SphMap --------------------------- #
# ============================================================= #
class SphMap(object):
    """ A class to handle spherical harmonics on healpy map
        to perform radio imaging.
    """

    def __init__(self, lmax):
        self.lmax = lmax


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def lmax(self):
        """
        """
        return self._lmax
    @lmax.setter
    def lmax(self, l):
        self._almsize = Alm.getsize(l - 1)
        self._hpsize = max(128, 2 * l)#max(64, 2 * l)
        self._max = l
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def horizon_mask(self, zenith, radius=45):
        """ Mask the healpix map pixels that are below a given
            radius from the local zenith.

            :param zenith:
                Array of local zenith converted in Galactic
                coordinates. This might typically be
                :attr:`SphUVW.zen_gal`
            :type zenith: `np.ndarray`
            :param radius:
                Radius from zenith, below which mask pixels
            :type radius: float
        """
        if radius==0:
            return np.ones(nside2npix(self._hpsize), dtype='bool')
        # Theta coordinates of the healpix map
        theta, phi = pix2ang(
            self._hpsize,
            np.arange(nside2npix(self._hpsize)),
            lonlat=True
        )
        # Convert theta to longitudes
        t_mask = theta > np.pi
        theta[t_mask] =  (2. * np.pi - theta[t_mask]) * -1.
        pix_coord = np.deg2rad(
            np.vstack([theta, phi])[..., np.newaxis]
        )
        # Galactiv coordinates
        gal_l = np.array([z.l.deg for z in zenith])
        gal_b = np.array([z.b.deg for z in zenith])
        gal_coord= np.deg2rad(
            np.vstack([gal_l, gal_b])[:, np.newaxis]
        )
        # Difference between two sets of angles
        lon1, lon2 = gal_coord[0], pix_coord[0]
        lat1, lat2 = gal_coord[1], pix_coord[1]
        c1 = np.cos(lat1)
        c2 = np.cos(lat2)
        s1 = np.sin(lat1)
        s2 = np.sin(lat2)
        cd = np.cos(lon2 - lon1)
        sd = np.sin(lon2 - lon1)
        u = np.square(c2*sd) + np.square(c1*s2 - s1*c2*cd)
        d = s1*s2 + c1 + c2*cd
        dist = np.arctan2(
            u.flatten(),
            d.flatten()
        ).reshape(u.shape)

        # Mask evrything below horizon, i.e. keep if raidus < 45deg
        return ~np.any(np.rad2deg(dist)<radius, axis=1)

# ============================================================= #

