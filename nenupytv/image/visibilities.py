#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupytv'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'Visibilities'
    ]


import numpy as np

import nenupytv
from nenupytv.read import Crosslets, XST
from nenupytv.uvw import UVW
from nenupytv.instru import NenuFAR
from nenupytv.image import Grid_Simple, Dirty
from nenupytv.response import PSF
from nenupytv.astro import eq_zenith, to_lmn, astro_image, radio_sources
from nenupytv.calibration import Skymodel


# ============================================================= #
# ----------------------- Visibilities ------------------------ #
# ============================================================= #
class Visibilities(UVW):
    """
    """

    def __init__(self, crosslets, tidx=None):
        self.skymodel = None
        self.srcnames = None

        self.cross = crosslets
        self.vis = self.cross.reshape(
            tidx=tidx,
            fmean=False,
            tmean=False
        )

        start = self.cross.time[0]
        stop = self.cross.time[-1]
        self.phase_center = eq_zenith(
                time=start + (stop - start)/2,
            )
        self.fov = None
        
        # Initialize the UVW Class
        super().__init__(
            array=NenuFAR(
                miniarrays=self.cross.meta['ma']
            ),
            freq=self.cross.meta['freq']
        )
        # Compute the UVW coordinates at the zenith
        self.compute(
            time=self.cross.time if tidx is None else self.cross.time[tidx],
            ra=None,
            dec=None,
        )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def cross(self):
        return self._cross
    @cross.setter
    def cross(self, c):
        if isinstance(c, str):
            if c.endswith('.dat'):
                c = Crosslets(c)
            elif c.endswith('.fits'):
                c = XST(c)
            else:
                raise ValueError(
                    'Unrecognized format'
                )
        if not isinstance(c, (Crosslets, XST)):
            raise TypeError(
                'Crosslets or XST object expected'
            )
        self._cross = c
        return


    @property
    def vis(self):
        total_mask = self.flag | self.selection
        total_mask = np.tile(
            np.expand_dims(total_mask[..., 0], axis=4),
            4
        )
        return np.ma.masked_array(
            self._vis,
            mask=total_mask
            )
    @vis.setter
    def vis(self, v):
        xx, yy = np.tril_indices(v.shape[2])
        v[..., yy, xx, :] = v[..., xx, yy, :].conj()
        self._vis = v
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def flagdata(self, nsig=5):
        """
        """
        for i in range(self.vis.shape[-1]):
            mean = np.mean(self.vis[..., i].real)
            std = np.std(self.vis[..., i].real)
            self.flag += (np.abs(self.vis[..., i].real) > mean + nsig*std)[..., np.newaxis]
        return


    def predict(self, skymodel=None, names=None, fov=None, cutoff=None):
        """ Predict visibilities given a sky model

            Skymodel : `np.ndarray`
                [(name, flux, ra, dec)]
        """
        self.fov = fov
        if skymodel is None:
            self.skymodel, self.srcnames = self._get_skymodel(
                cutoff=cutoff
            )
        else:
            self.skymodel = skymodel
            self.srcnames = names 

        vis_model = np.zeros(
            self.vis.shape,
            dtype='complex'
        )

        # Compute the zenith coordinates for every time step
        zen = np.array(list(map(eq_zenith, self.cross.time)))
        ra_0 = zen[:, 0]
        dec_0 = zen[:, 1]

        # Pointers to u, v, w coordinates
        u = self.uvw_wave[..., 0]
        v = self.uvw_wave[..., 1]
        w = self.uvw_wave[..., 2]

        # Loop over skymodel sources
        na = np.newaxis
        for k in range(self.skymodel.shape[0]):
            flux = self.skymodel[k, 0] # Jy
            ra, dec = self.skymodel[k, 1], self.skymodel[k, 2]
            l, m, n = to_lmn(ra, dec, ra_0, dec_0)
            ul = u*l[:, na, na, na]
            vm = v*m[:, na, na, na]
            nw = (n[:, na, na, na] - 1)*w
            # phase = np.exp(-2.j*np.pi*(ul + vm))# + nw))
            phase = np.exp(-2.j*np.pi*(ul + vm + nw))
            # adding the w component mess up with subsequent plots
            vis_model += flux * phase[..., na]
        
        total_mask = self.flag | self.selection
        total_mask = np.tile(
            np.expand_dims(total_mask[..., 0], axis=4),
            4
        )
        return np.ma.masked_array(
            vis_model,
            mask=total_mask
            )


    def calibrate(self, vis_model):
        """
        """
        from scipy.optimize import leastsq

        def err_func(g, d, m, residual):
            Nm = d.size//2
            N = g.size//2
            G = np.diag(g[0:N] + 1j*g[N:])
            D = np.reshape(
                    d[0:Nm],
                    (N, N)
                ) + np.reshape(
                    d[Nm:],
                    (N, N)
                )*1j #matrization
            M = np.reshape(
                    m[0:Nm],
                    (N, N)
                ) + np.reshape(
                    m[Nm:],
                    (N, N)
                )*1j
            T = np.dot(G, M)
            T = np.dot(T, G.conj())
            R = D - T
            r_r = np.ravel(R.real) #vectorization
            r_i = np.ravel(R.imag)
            #r = np.hstack([r_r, r_i])
            residual[:r_r.size] = r_r
            residual[r_i.size:] = r_i
            r = residual.copy()
            return r

        def reorder(array):
            array = np.moveaxis(
                np.mean(array, axis=1)[..., 0],
                [0, 1, 2],
                [2, 0, 1]
            )
            return array

        vis = reorder(self.vis)
        mod = reorder(vis_model)

        nant = vis.shape[0] #number of antennas
        ant_shape = (nant, nant)
        ntime = vis.shape[2]

        temp = np.ones(ant_shape ,dtype=complex)
        G = np.zeros(vis.shape, dtype=complex)
        # g = np.zeros((vis.shape[0],vis.shape[2]), dtype=complex)
       
        data = np.zeros(nant**2 * 2)
        model = np.zeros(nant**2 * 2)
        residual = np.zeros(nant**2 * 2)
        
        for t in range(vis.shape[2]): #perform calibration per time-slot
            g_0 = np.ones((2*nant,)) # first antenna gain guess 
            g_0[nant:] = 0
            data[:data.size//2] = vis[..., t].real.ravel()
            data[data.size//2:] = vis[..., t].imag.ravel()
            model[:model.size//2] = mod[:, :, t].real.ravel()
            model[model.size//2:] = mod[:, :, t].imag.ravel()
            m = model.copy()
            g_lstsqr_temp = leastsq(
                err_func,
                g_0,
                args=(data, model, residual)
            )
            g_lstsqr = g_lstsqr_temp[0]          
               
            G_m = np.dot(
                np.diag(g_lstsqr[0:nant] + 1j*g_lstsqr[nant:]),
                temp
            )
            G_m = np.dot(
                G_m,
                np.diag((g_lstsqr[0:nant] + 1j*g_lstsqr[nant:]).conj())
            )           

            # g[:, t] = g_lstsqr[0:nant] + 1j*g_lstsqr[nant:] #creating antenna gain vector       
            G[..., t] = G_m
             
        cal_vis = G**(-1) * vis
        cal_vis = np.tile(
            np.moveaxis(
                cal_vis,
                [0, 1, 2],
                [1, 2, 0]
            )[:, np.newaxis, ..., np.newaxis],
            4
        )
        cal_vis = np.repeat(
            cal_vis,
            self.freq.size,
            axis=1
        )

        return cal_vis


    def make_nearfield_image(self):
        """
        """
        extent = [-100, 100, -100, 100]
        npix_p = 100
        npix_q = 100
        z = 10
        x = np.linspace(extent[0], extent[1], npix_p)
        y = np.linspace(extent[2], extent[3], npix_q)
        posx, posy = np.meshgrid(x, y)
        posxyz = np.transpose(np.array([posx, posy, z * np.ones_like(posx)]), [1, 2, 0])

        station_pqr = self.bsl.copy()
        diff_vectors = (station_pqr[:, None, None, :] - posxyz[None, :, :, :])
        distances = np.linalg.norm(diff_vectors, axis=3)
        return


    def make_dirty(self, vis=None, fov=60, robust=-2, **kwargs):
        """
        """
        self._selection(**kwargs)
        self.fov = fov

        if vis is None:
            vis = self.vis
        avg_vis = np.mean(vis, axis=(0, 1))
        avg_uvw = np.mean(self.uvw_wave, axis=(0, 1))

        self.grid = Grid_Simple(
            vis=avg_vis,
            uvw=avg_uvw,
            freq=np.mean(self.freq),
            fov=fov,
            robust=robust
        )
        self.grid.populate()

        dirty = Dirty(self.grid, self.cross)
        dirty.compute()

        psf = PSF(self.grid)
        psf.compute()
        self.psf = psf.psf
        self.clean_beam = psf.clean_beam

        return dirty


    def clean(self,
            dirty,
            pngfile=None,
            fitsfile=None,
            gainfactor=0.05,
            niter=10000,
            threshold=None,
            **kwargs):
        """
        """
        if not isinstance(dirty, Dirty):
            raise TypeError(
                'Give a Dirty object'
            )
        self._residuals = dirty.i.copy()
        self._model = np.zeros(dirty.i.shape)
        nsize = int(self._residuals.shape[0])
        center_id = (
            int(nsize/2), int(nsize/2)
        )

        sig = []
        sigma = 10000 # huge value to start with
        n = 0
        while (n < niter) and (sigma > threshold):
            # find peak
            max_id = np.unravel_index(
                self._residuals.argmax(),
                self._residuals.shape
            )
            # flux to subtract
            fsub = self._residuals[max_id] * gainfactor
            # add a dirac with the flux to the model
            self._model[max_id] += fsub
            # shift the PSF
            psf_tmp = np.roll(
                self.psf.real, max_id[0] - center_id[0],
                axis=0
            )
            psf_tmp = np.roll(
                psf_tmp, max_id[1] - center_id[1],
                axis=1
            )
            psf_tmp = psf_tmp[
                int(nsize/2):int(nsize*3/2),
                int(nsize/2):int(nsize*3/2)
            ]
            # subtract the psf * flux to residuals
            self._residuals -= psf_tmp * fsub
            # compute the new std and keep track of it
            sigma = np.std(self._residuals)
            try:
                if sigma == sig[-1]:
                    break
            except IndexError:
                pass
            sig.append(sigma)
            n += 1

        # Convolve model and clean beam
        model_fft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(self._model)))
        clean_beam_fft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(self.clean_beam)))
        image = np.abs(
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(model_fft * clean_beam_fft)))
        ) + self._residuals

        # Plot
        start = self.cross.time[0]
        stop = self.cross.time[-1]
        astro_image(
            image=image,
            center=self.phase_center,
            npix=self.grid.nsize,
            resol=np.degrees(self.grid.resol.value),
            time=start + (stop - start)/2,
            freq=np.mean(self.freq),
            show_sources=True,
            colorbar=False,
            pngfile=pngfile,
            fitsfile=fitsfile,
            **kwargs
        )
        return


    def plot_crosscorr(self, pngfile=None, fitsfile=None):
        """
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.ticker import LinearLocator
        from matplotlib.colors import Normalize
        from os import path

        if fitsfile is not None:
            from astropy.io import fits
            
            prihdr = fits.Header()
            prihdr.set('INSTRU', 'NenuFAR')
            prihdr.set('SOFTWARE', 'nenupytv')
            prihdr.set('VERSION', nenupytv.__version__)
            prihdr.set('OBSTART', self.cross.time[0].isot)
            prihdr.set('OBSTOP', self.cross.time[-1].isot)
            prihdr.set('FREQ', np.mean(self.freq))
            prihdr.set('CONTACT', 'alan.loh@obspm.fr')

            primhdu = fits.PrimaryHDU(data=None, header=prihdr)
            hdus = [primhdu]

            c_names = ['XX', 'XY', 'YX', 'YY']
            c_indices = [0, 1, 2, 3]
            for ci, ni in zip(c_indices, c_names):
                di = np.absolute(
                    np.mean(
                        self.vis[..., ci].data,
                        axis=(0, 1)
                    )
                )
                hdus.append(
                    fits.ImageHDU(
                        data=di,
                        header=None,
                        name=ni
                    )
                )
            hdulist = fits.HDUList(hdus)
            hdulist.writeto(
                path.join(
                    fitsfile,
                    'nenufartv_crossmat_{}.fits'.format(
                        self.cross.time[0].isot.split('.')[0].replace(':', '-')
                    )
                ),
                overwrite=True
            )

        cross_matrix = np.absolute(
            np.mean(
                self.vis[..., 0].data,
                axis=(0, 1)
            )
        )
        cross_matrix[np.arange(cross_matrix.shape[0]), np.arange(cross_matrix.shape[1])] = 0.
        cross_matrix = 10*np.log10(
            cross_matrix
        )

        vmin = np.percentile(cross_matrix, 5.)
        vmax = np.percentile(cross_matrix, 99)

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(
            cross_matrix,
            origin='lower',
            cmap='YlGnBu_r',
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax
        )
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size=0.3, pad=0.2)
        cb = ColorbarBase(
            cax,
            cmap='YlGnBu_r',
            orientation='vertical',
            norm=Normalize(vmin=vmin, vmax=vmax),
            ticks=LinearLocator()
        )
        cb.solids.set_edgecolor('face')
        cb.set_label('Amplitude XX (dB)')
        cb.formatter.set_powerlimits((0, 0))
        ax.set_xlabel('MA')
        ax.set_ylabel('MA')
        ax.set_title('{}'.format(
                self.cross.time[self.cross.time.size//2].iso.split('.')[0]
            )
        )
        ax.set_xticks(np.arange(0, self.array.ant1[0, :].size, 1));
        ax.set_yticks(np.arange(0, self.array.ant2[:, 0].size, 1));
        ax.set_xticklabels(self.array.ant1[0, :], fontsize=7, rotation=45);
        ax.set_yticklabels(self.array.ant2[:, 0], fontsize=7);
        ax.grid(color='black', linestyle='-', linewidth=1, alpha=0.1)

        plt.tight_layout()

        if pngfile is None:
            plt.show()
        else:
            plt.savefig(pngfile, dpi=300)
        
        return


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _get_skymodel(self, cutoff):
        """
        """
        sky = Skymodel(
            ra=self.phase_center[0],
            dec=self.phase_center[1],
            radius=self.fov/2,
            cutoff=cutoff
        )
        skymodel, names = sky.get_skymodel(
            freq=np.mean(self.freq)
        )
        return skymodel, names
# ============================================================= #



# def stefcal(data, model):
#     """
#     """
#     def reorder(array):
#         # Separate different cross-correlations
#         array = array.reshape(
#             list(array.shape[:-1]) + [2, 2]
#         )
#         # Reshape
#         array = np.moveaxis(
#             array,
#             [0, 1, 2, 3, 4, 5],
#             [3, 2, 0, 5, 1, 4]
#         )
#         return array

#     data = reorder(data)
#     model = reorder(model)
    
#     oldshape = data.shape
#     newshape = (oldshape[0]*oldshape[1], oldshape[2], oldshape[3]*oldshape[4]*oldshape[5])
#     V = data.reshape(newshape)
#     M = model.reshape(newshape)

#     G=np.ones(newshape[0])*np.sqrt(np.linalg.norm(V)/np.linalg.norm(M))

#     nSt = newshape[0]
#     nCh = newshape[1]

#     itera=0
#     while itera<20:
#         itera+=1
#         Gold = G
#         for st1 in range(nSt):
#             w=0.
#             t=0.
#             for ch in range(nCh):
#                 z = np.conj(Gold) *  M[:, ch, st1] # element-wise
#                 w += np.dot(np.conj(z), z).real
#                 t += np.dot(np.conj(z), V[:, ch, st1]).real
#             G[st1] = t/w;
#         if itera % 2 == 0:
#             dG= np.linalg.norm(G - Gold) / np.linalg.norm(G);
#             G = 0.5 * G + 0.5 * Gold
#             if dG<1.e-5:
#                 break 

#     return G

# G = stefcal(visib, model)