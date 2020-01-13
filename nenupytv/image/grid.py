#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    TO DO:
    map to healpix grid
"""


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Grid',
    'Grid_HPX'
    ]


# import healpy as hp
import numpy as np
import astropy.units as un

from nenupytv.image import AAFilter


# ============================================================= #
# ---------------------------- Grid --------------------------- #
# ============================================================= #
class Grid(object):
    """ Gridding class

        :param vis: Visibilities
        :type vis: `~np.ndarray`
        :param uvw: UVW coordinates
        :type uvw: `~np.ndarray`
        :param fov: Field of view radius in degrees
        :type fov: float
        :param freq: Observing frequency
        :type freq: float
        :param filter: Convolution filter to apply
        :type filter: `~nenupytv.image.AAFilter`
        :param cellsize: Cell size in degrees
        :type cellsize: float
    """

    def __init__(
        self,
        vis,
        uvw,
        freq,
        fov,
        conv_filter=AAFilter(),
        cellsize=None
        ):
        self.nsize = None
        self.fov = fov
        self.vis = vis
        self.uvw = uvw
        self.freq = freq
        self.filter = conv_filter
        self.cellsize = cellsize


    @property
    def _ovsmpl(self):
        """ Filter oversampling
        """
        return self.filter.oversample


    @property
    def _hsup(self):
        """ Filter Half_sup
        """
        return self.filter.half_sup


    @property
    def _filter_idx(self):
        """ Filter indices
        """
        return np.arange(-self._hsup, self._hsup + 1)


    @property
    def _ftaps(self):
        """
        """
        return self.filter.filter_taps


    @property
    def vis(self):
        return self._vis
    @vis.setter
    def vis(self, v):
        x, y = np.tril_indices(v.shape[2], 0)
        xx = np.hstack((x[x!=y], y[x!=y]))
        yy = np.hstack((y[x!=y], x[x!=y]))
        self._vis = v[:, :, xx, yy, ...] + v[:, :, yy, xx, ...].conj()
        return


    @property
    def uvw(self):
        return self._uvw
    @uvw.setter
    def uvw(self, u):
        x, y = np.tril_indices(u.shape[2], 0)
        xx = np.hstack((x[x!=y], y[x!=y]))
        yy = np.hstack((y[x!=y], x[x!=y]))
        self._uvw = u[:, :, xx, yy, ...]

        maxu = np.max(np.abs(self._uvw[..., 0]))
        maxv = np.max(np.abs(self._uvw[..., 1]))
        self.resol = 1. / (5 * 2 * np.max((maxu, maxv))) * un.rad
        resol = self.resol.to(un.deg).value
        self.nsize = int(np.round(self.fov / resol))
        return


    @property
    def nsize(self):
        return self._nsize
    @nsize.setter
    def nsize(self, n):
        if n is None:
            self._nsize = None
        else:
            self._nsize = n
            self.measurement = np.zeros(
                (self.vis.shape[3], self.nsize, self.nsize),
                dtype='complex64'
            )
            # for deconvolution the PSF should be 2x size of the image (see 
            # Hogbom CLEAN for details), one grid for the sampling function:
            self.sampling = np.zeros(
                (2*self.nsize, 2*self.nsize),
                dtype='complex64'
            )


    def populate(self):
        """
        """
        if self.uvw.shape[0] != 1:
            raise ValueError(
                'UVW need to be averaged in time'
            )
        if self.uvw.shape[1] != 1:
            raise ValueError(
                'UVW need to be averaged in frequency'
            )
        # iter over number of baselines in the triangular inferior
        # matrix (i.e. not autocorr), divided by 3 because (u, v, w)
        for vis_bl in range(int(self.uvw.size / 3)):
            u = self.uvw[..., vis_bl, 0] * np.radians(self.fov)
            v = self.uvw[..., vis_bl, 1] * np.radians(self.fov)

            du = int(np.round(u))
            dv = int(np.round(v))
            fu_offset = int(
                    (1 + self._hsup + (-u + du)) * self._ovsmpl
                )
            fv_offset = int(
                    (1 + self._hsup + (-v + dv)) * self._ovsmpl
                )

            du_psf = int(np.round(u*2))
            dv_psf = int(np.round(v*2))
            fu_offset_psf = int(
                    (1 + self._hsup + (-u*2 + du_psf)) * self._ovsmpl
                )
            fv_offset_psf = int(
                    (1 + self._hsup + (-v*2 + dv_psf)) * self._ovsmpl
            )

            if (dv + self.nsize // 2 + self._hsup >= self.nsize or
                du + self.nsize // 2 + self._hsup >= self.nsize or
                dv + self.nsize // 2 - self._hsup < 0 or
                du + self.nsize // 2 - self._hsup < 0):
                continue
            
            # yield (
            #     du,
            #     dv,
            #     fu_offset,
            #     fv_offset,
            #     du_psf,
            #     dv_psf,
            #     fu_offset_psf,
            #     fv_offset_psf
            # )

            for conv_v in self._filter_idx:
                v_tap = self._ftaps[conv_v * self._ovsmpl + fv_offset]
                v_tap_psf = self._ftaps[conv_v * self._ovsmpl + fv_offset_psf]

                grid_v = dv + conv_v + self.nsize // 2
                grid_v_psf = dv_psf + conv_v + self.nsize
                
                for conv_u in self._filter_idx:
                    u_tap = self._ftaps[conv_u * self._ovsmpl + fu_offset]
                    u_tap_psf = self._ftaps[conv_u * self._ovsmpl + fu_offset_psf]
                    
                    grid_u = du + conv_u + self.nsize // 2
                    grid_u_psf = du_psf + conv_u + self.nsize

                    conv_weight = v_tap * u_tap
                    conv_weight_psf = v_tap_psf * u_tap_psf
                    
                    for p in range(self.vis.shape[3]):
                        self.measurement[p, grid_v, grid_u] += self.vis[0, 0, vis_bl, p] * conv_weight
                    # assuming the PSF is the same for different correlations:
                    self.sampling[grid_v_psf, grid_u_psf] += (1+0.0j) * conv_weight_psf
# ============================================================= #




# ============================================================= #
# -------------------------- Grid_HPX ------------------------- #
# ============================================================= #
class Grid_HPX(object):
    """
    """

    def __init__(self, resolution):
        self.nside = None
        self.resol = resolution
        self.ra, self.dec = self._hpx_radec()


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def resol(self):
        """ Resolution of the cell in degrees
        """
        return self._resol
    @resol.setter
    def resol(self, r):
        nsides = 2**np.arange(1, 12)
        resol_rad = hp.nside2resol(
            nside=nsides,
            arcmin=False
            )
        resol_deg = np.degrees(resol_rad)
        idx = (np.abs(resol_deg - r)).argmin()
        self._resol = resol_deg[idx]
        self.nside = nsides[idx]
        return


    @property
    def nside(self):
        return self._nside
    @nside.setter
    def nside(self, n):
        if n is None:
            self._nside = None
        else:
            self.npix = hp.nside2npix(n)
            self._nside = n
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _hpx_radec(self):
        """
        """
        ra, dec = hp.pix2ang(
            nside=self.nside,
            ipix=np.arange(self.npix),
            lonlat=True,
            nest=False
            )
        return ra, dec

# ============================================================= #



