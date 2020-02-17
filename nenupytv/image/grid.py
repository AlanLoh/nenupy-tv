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
    'Grid_Simple',
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
        cellsize=None,
        robust=0.
        ):
        self.nsize = None
        self.fov = fov
        self.vis = vis
        self.uvw = uvw
        self.freq = freq
        self.filter = conv_filter
        self.cellsize = cellsize
        self.robust = robust


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
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
        x, y = np.tril_indices(v.shape[0], 0)
        xx = np.hstack((x[x!=y], y[x!=y]))
        yy = np.hstack((y[x!=y], x[x!=y]))
        self._vis = v[xx, yy, ...] + v[yy, xx, ...].conj()
        return


    @property
    def uvw(self):
        return self._uvw
    @uvw.setter
    def uvw(self, u):
        x, y = np.tril_indices(u.shape[0], 0)
        xx = np.hstack((x[x!=y], y[x!=y]))
        yy = np.hstack((y[x!=y], x[x!=y]))
        self._uvw = u[xx, yy, ...]

        maxu = np.max(np.abs(self._uvw[..., 0]))
        maxv = np.max(np.abs(self._uvw[..., 1]))
        self.resol = 1. / (5 * 2 * np.max((maxu, maxv))) * un.rad
        # self.resol = 1. / (10 * 2 * np.max((maxu, maxv))) * un.rad
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
                (self.vis.shape[1], self.nsize, self.nsize),
                dtype='complex64'
            )
            self._meas_w = np.zeros(
                (self.nsize, self.nsize),
                dtype='float'
            )
            # for deconvolution the PSF should be 2x size of the image (see 
            # Hogbom CLEAN for details), one grid for the sampling function:
            self.sampling = np.zeros(
                (2*self.nsize, 2*self.nsize),
                dtype='complex64'
            )
            self._samp_w = np.zeros(
                (2*self.nsize, 2*self.nsize),
                dtype='float'
            )


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def populate(self):
        """
        """
        # if self.uvw.shape[0] != 1:
        #     raise ValueError(
        #         'UVW need to be averaged in time'
        #     )
        # if self.uvw.shape[1] != 1:
        #     raise ValueError(
        #         'UVW need to be averaged in frequency'
        #     )
        # iter over number of baselines in the triangular inferior
        # matrix (i.e. not autocorr), divided by 3 because (u, v, w)
        for vis_bl in range(int(self.uvw.size / 3)):
            u = self.uvw[vis_bl, 0] * np.radians(self.fov)
            v = self.uvw[vis_bl, 1] * np.radians(self.fov)

            if np.ma.is_masked(u) or np.ma.is_masked(v):
                continue

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
                    
                    # for p in range(self.vis.shape[3]):
                    #     self.measurement[p, grid_v, grid_u] += self.vis[0, 0, vis_bl, p] * conv_weight
                    for p in range(self.vis.shape[1]):
                        self.measurement[p, grid_v, grid_u] += self.vis[vis_bl, p] * conv_weight
                        # self._meas_w[grid_v, grid_u] += 1
                        self._meas_w[grid_v, grid_u] += 1. * conv_weight
                    # assuming the PSF is the same for different correlations:
                    self.sampling[grid_v_psf, grid_u_psf] += (1+0.0j) * conv_weight_psf
                    # self._samp_w[grid_v_psf, grid_u_psf] += 1
                    self._samp_w[grid_v_psf, grid_u_psf] += 1. * conv_weight_psf

        self._compute_weights()
        return

    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _compute_weights(self):
        """
        """
        # nbsl = self.uvw.size / 3
        # bsl_time = (2.*nbsl*1) # one time step
        # num = (5.*10.**(-self.robust))**2.*bsl_time
        # meas_f = num/np.sum(self._meas_w**2.)
        # samp_f = num/np.sum(self._samp_w**2.)

        # self.meas_weighted = self._meas_w / (1 + self._meas_w * meas_f)
        # self.samp_weighted = self._samp_w / (1 + self._samp_w * samp_f)

        factor = (5. * 10.**(-self.robust) )**2
        f = factor / (np.sum(self._meas_w**2.) / np.sum(self._meas_w))
        self.meas_weights = self._meas_w / (1 + self._meas_w * f)
        self.meas_weights /= self.meas_weights.max()

        f = factor / (np.sum(self._samp_w**2.) / np.sum(self._samp_w))
        self.samp_weights = self._samp_w / (1 + self._samp_w * f)
        self.samp_weights /= self.samp_weights.max()
        return
# ============================================================= #



# ============================================================= #
# ---------------------------- Grid --------------------------- #
# ============================================================= #
class Grid_Simple(object):
    """
    """

    def __init__(
        self,
        vis,
        uvw,
        freq,
        fov,
        cellsize=None,
        robust=0.,
        convolution=None
        ):
        self.nsize = None
        self.fov = fov
        self.vis = vis
        self.uvw = uvw
        self.freq = freq
        self.cellsize = cellsize
        self.robust = robust
        self.convolution = convolution


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def vis(self):
        return self._vis
    @vis.setter
    def vis(self, v):
        x, y = np.tril_indices(v.shape[0], 0)
        xx = np.hstack((x[x!=y], y[x!=y]))
        yy = np.hstack((y[x!=y], x[x!=y]))
        self._vis = v[xx, yy, ...] + v[yy, xx, ...].conj()
        return


    @property
    def uvw(self):
        return self._uvw
    @uvw.setter
    def uvw(self, u):
        x, y = np.tril_indices(u.shape[0], 0)
        xx = np.hstack((x[x!=y], y[x!=y]))
        yy = np.hstack((y[x!=y], x[x!=y]))
        self._uvw = u[xx, yy, ...]

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
                (self.vis.shape[1], self.nsize, self.nsize),
                dtype='complex64'
            )
            self._meas_w = np.zeros(
                (self.nsize, self.nsize),
                dtype='float'
            )
            # for deconvolution the PSF should be 2x size of the image (see 
            # Hogbom CLEAN for details), one grid for the sampling function:
            self.sampling = np.zeros(
                (2*self.nsize, 2*self.nsize),
                dtype='complex64'
            )
            self._samp_w = np.zeros(
                (2*self.nsize, 2*self.nsize),
                dtype='float'
            )


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def populate(self):
        """
        """
        for vis_bl in range(int(self.uvw.size / 3)):
            u = self.uvw[vis_bl, 0] * np.radians(self.fov)
            v = self.uvw[vis_bl, 1] * np.radians(self.fov)

            if np.ma.is_masked(u) or np.ma.is_masked(v):
                continue

            du = int(np.round(u)) + self.nsize // 2
            dv = int(np.round(v)) + self.nsize // 2
            du_psf = int(np.round(u*2)) + self.nsize
            dv_psf = int(np.round(v*2)) + self.nsize
            
            for p in range(self.vis.shape[1]):
                self.measurement[p, dv, du] += self.vis[vis_bl, p]
            self._meas_w[dv, du] += 1.
            # assuming the PSF is the same for different correlations:
            self.sampling[dv_psf, du_psf] += (1+0.0j)
            self._samp_w[dv_psf, du_psf] += 1.

        self._convolve()

        self._compute_weights()
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _compute_weights(self):
        """
        """
        factor = (5. * 10.**(-self.robust) )**2
        f = factor / (np.sum(self._meas_w**2.) / np.sum(self._meas_w))
        self.meas_weights = self._meas_w / (1 + self._meas_w * f)
        self.meas_weights /= self.meas_weights.max()

        f = factor / (np.sum(self._samp_w**2.) / np.sum(self._samp_w))
        self.samp_weights = self._samp_w / (1 + self._samp_w * f)
        self.samp_weights /= self.samp_weights.max()
        return


    def _convolve(self):
        """
        """
        if self.convolution is None:
            return
        elif self.convolution.lower() == 'gaussian':
            from scipy.ndimage import gaussian_filter
            def _scipy_gauss(im, sig=0.25):
                if im.dtype == np.complex64:
                    im_re = gaussian_filter(
                        input=im.real,
                        sigma=sig,
                        order=0,
                        output=None,
                        mode='reflect',
                        cval=0.0,
                        truncate=4.0
                    )
                    im_im = gaussian_filter(
                        input=im.imag,
                        sigma=sig,
                        order=0,
                        output=None,
                        mode='reflect',
                        cval=0.0,
                        truncate=4.0
                    )
                    return im_re + 1.j*im_im
                else:
                    return gaussian_filter(
                        input=im,
                        sigma=sig,
                        order=0,
                        output=None,
                        mode='reflect',
                        cval=0.0,
                        truncate=4.0
                    )
            for p in range(self.measurement.shape[0]):
                self.measurement[p, ...] = _scipy_gauss(self.measurement[p, ...])
            self.sampling = _scipy_gauss(self.sampling)
            self._meas_w = _scipy_gauss(self._meas_w)
            self._samp_w = _scipy_gauss(self._samp_w)
        else:
            raise Exception(
                'Convolution kernel {} not implemented.'.format(self.convolution)
            )
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




# class ImgWPlus(aipy.img.ImgW):
#     """
#     Sub-class of the aipy.img.ImgW class that adds support for different 
#     visibility weighting scheme and uv plane tapering.  This class also
#     adds in a couple of additional methods that help determine the size of
#     the field of view and the pixels near the phase center.
#     """
#     def __init__(self, size=100, res=1, wres=.5, mf_order=0):
#         """size = number of wavelengths which the UV matrix spans (this 
#         determines the image resolution).
#         res = resolution of the UV matrix (determines image field of view).
#         wres: the gridding resolution of sqrt(w) when projecting to w=0."""
#         self.res = float(res)
#         self.size = float(size)
#         ## Small change needed to work with Numpy 1.12+
#         dim = numpy.int64(numpy.round(self.size / self.res))
#         self.shape = (dim,dim)
#         self.uv = numpy.zeros(shape=self.shape, dtype=numpy.complex64)
#         self.bm = []
#         for i in range(mf_order+1):
#             self.bm.append(numpy.zeros(shape=self.shape, dtype=numpy.complex64))
#         self.wres = float(wres)
#         self.wcache = {}
#     def put(self, uvw, data, wgts=None, invker2=None):
#         """Same as Img.put, only now the w component is projected to the w=0
#         plane before applying the data to the UV matrix."""
#         u, v, w = uvw
#         if len(u) == 0: return
#         if wgts is None:
#             wgts = []
#             for i in range(len(self.bm)):
#                 if i == 0:
#                     wgts.append(numpy.ones_like(data))
#                 else:
#                     wgts.append(numpy.zeros_like(data))
#         if len(self.bm) == 1 and len(wgts) != 1:
#             wgts = [wgts]
#         assert(len(wgts) == len(self.bm))
#         # Sort uvw in order of w
#         order = numpy.argsort(w)
#         u = u.take(order)
#         v = v.take(order)
#         w = w.take(order)
#         data = data.take(order)
#         wgts = [wgt.take(order) for wgt in wgts]
#         sqrt_w = numpy.sqrt(numpy.abs(w)) * numpy.sign(w)
#         i = 0
#         while True:
#             # Grab a chunk of uvw's that grid w to same point.
#             j = sqrt_w.searchsorted(sqrt_w[i]+self.wres)
#             print('%d/%d datums' % (j, len(w)))
#             avg_w = numpy.average(w[i:j])
#             # Put all uv's down on plane for this gridded w point
#             wgtsij = [wgt[i:j] for wgt in wgts]
#             uv,bm = aipy.img.Img.put(self, (u[i:j],v[i:j],w[i:j]),
#                 data[i:j], wgtsij, apply=False)
#             # Convolve with the W projection kernel
#             invker = numpy.fromfunction(lambda u,v: self.conv_invker(u,v,avg_w),
#                 uv.shape)
#             if not invker2 is None:
#                 invker *= invker2
#             self.uv += ifft2Function(fft2Function(uv) * invker)
#             #self.uv += uv
#             for b in range(len(self.bm)):
#                 self.bm[b] += ifft2Function(fft2Function(bm[b]) * invker)
#                 #self.bm[b] += numpy.array(bm)[0,:,:]
#             if j >= len(w):
#                 break
#             i = j
