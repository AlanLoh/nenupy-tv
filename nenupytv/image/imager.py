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
    'Imager'
    ]


import numpy as np
import numexpr as ne
from os import remove
from os.path import dirname, isdir, abspath, basename, join
import multiprocessing as mp
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes

import nenupytv
from nenupytv.read import Crosslets, XST
from nenupytv.uvw import UVW
from nenupytv.instru import NenuFAR
from nenupytv.astro import (
    eq_zenith,
    to_lmn,
    astro_image,
    AstroPlot,
    radio_sources,
    wavelength,
    to_radec
)
from nenupytv.calibration import Skymodel


# ============================================================= #
# ------------------- Multiprocessing Image ------------------- #
# ============================================================= #
def _init(arr_to_populate1, arr_to_populate2, lg, mg, u, v):
    """ Each pool process calls this initializer. Load the array to be populated into that process's global namespace """
    global arr1
    global arr2
    global larr
    global marr
    global uarr
    global varr
    arr1 = arr_to_populate1
    arr2 = arr_to_populate2
    larr = lg
    marr = mg
    uarr = u
    varr = v

def fill_per_block(args):
    x0, x1, y0, y1 = args.astype(int)
    tmp_r = np.ctypeslib.as_array(arr1)
    tmp_i = np.ctypeslib.as_array(arr2)
    na = np.newaxis

    lg = larr[na, x0:x1, y0:y1]
    mg = marr[na, x0:x1, y0:y1]
    pi = np.pi
    expo = ne.evaluate('exp(2j*pi*(uarr*lg+varr*mg))')

    tmp_r[:, x0:x1, y0:y1] = expo.real
    tmp_i[:, x0:x1, y0:y1] = expo.imag

def mp_expo(npix, ncpus, lg, mg, u, v):
    block_size = int(npix/np.sqrt(ncpus))
    result_r = np.ctypeslib.as_ctypes(
        np.zeros((u.shape[0], npix, npix))
    )
    result_i = np.ctypeslib.as_ctypes(
        np.zeros_like(result_r)
    )
    shared_array_r = sharedctypes.RawArray(
        result_r._type_,
        result_r
    )
    shared_array_i = sharedctypes.RawArray(
        result_i._type_,
        result_i
    )
    n_windows = int(np.sqrt(ncpus))
    block_idxs = np.array([
        (i, i+1, j, j+1) for i in range(n_windows) for j in range(n_windows)
    ])*block_size
    # pool = Pool(ncpus)
    # res = pool.map(
    #     fill_per_block,
    #     (block_idxs, shared_array_r, shared_array_i, block_size, lg, mg)
    # )
    pool = Pool(
        processes=ncpus,
        initializer=_init,
        initargs=(shared_array_r, shared_array_i, lg, mg, u, v)
    )
    res = pool.map(fill_per_block, (block_idxs))
    pool.close()
    result_r = np.ctypeslib.as_array(shared_array_r)
    result_i = np.ctypeslib.as_array(shared_array_i)
    del shared_array_r, shared_array_i
    return result_r + 1j * result_i
# ============================================================= #


# ============================================================= #
# ----------------- Multiprocessing Nearfield ----------------- #
# ============================================================= #
def _init_nf(arr_to_populate1, arr_to_populate2, delay, wavel):
    """ Each pool process calls this initializer. Load the array to be populated into that process's global namespace """
    global arr1b
    global arr2b
    global dela
    global lamb
    arr1b = arr_to_populate1
    arr2b = arr_to_populate2
    dela = delay
    lamb = wavel

def fill_per_block_nf(args):
    x0, x1, y0, y1 = args.astype(int)
    tmp_r = np.ctypeslib.as_array(arr1b)
    tmp_i = np.ctypeslib.as_array(arr2b)

    dd = dela[..., x0:x1, y0:y1]
    pi = np.pi
    expo = ne.evaluate('exp(2j*pi*dd/lamb)')

    tmp_r[..., x0:x1, y0:y1] = expo.real
    tmp_i[..., x0:x1, y0:y1] = expo.imag

def mp_expo_nf(npix, ncpus, dd, wavel):
    block_size = int(npix/np.sqrt(ncpus))
    result_r = np.ctypeslib.as_ctypes(
        np.zeros((1, dd.shape[1], npix, npix))
    )
    result_i = np.ctypeslib.as_ctypes(
        np.zeros_like(result_r)
    )
    shared_array_r = sharedctypes.RawArray(
        result_r._type_,
        result_r
    )
    shared_array_i = sharedctypes.RawArray(
        result_i._type_,
        result_i
    )
    n_windows = int(np.sqrt(ncpus))
    block_idxs = np.array([
        (i, i+1, j, j+1) for i in range(n_windows) for j in range(n_windows)
    ])*block_size
    pool = Pool(
        processes=ncpus,
        initializer=_init_nf,
        initargs=(shared_array_r, shared_array_i, dd, wavel)
    )
    res = pool.map(fill_per_block_nf, (block_idxs))
    pool.close()
    result_r = np.ctypeslib.as_array(shared_array_r)
    result_i = np.ctypeslib.as_array(shared_array_i)
    return result_r + 1j * result_i
# ============================================================= #


# ============================================================= #
# -------------------------- Imager --------------------------- #
# ============================================================= #
class Imager(UVW):
    """
    """

    def __init__(self, crosslets, fov=60, tidx=None, ncpus=1):
        self.skymodel = None
        self.srcnames = None
        self.fov = fov
        self.ncpus = ncpus

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


    @property
    def fov(self):
        return self._fov
    @fov.setter
    def fov(self, f):
        #lmmax = np.cos(np.radians(f))
        lmmax = np.cos(np.radians(90 - f/2))
        self.lmax = lmmax
        self.mmax = lmmax
        self._fov = f
        return


    @property
    def ncpus(self):
        return self._ncpus
    @ncpus.setter
    def ncpus(self, n):
        if not np.sqrt(n).is_integer():
            raise ValueError(
                'Number of CPUs must be a sqrtable'
            )
        if not ( n <= mp.cpu_count()):
            raise Exception(
                'Number of CPUs should be <={}'.format(n)
            )
        self._ncpus = n
        return


    @property
    def npix(self):
        return self._npix
    @npix.setter
    def npix(self, n):
        if not np.log2(n).is_integer():
            raise ValueError(
                'Number of pixels must be a power of 2'
            )
        self._npix = n
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


    def predict(self, skymodel=None, names=None, cutoff=100):
        """ Predict visibilities given a sky model

            Skymodel : `np.ndarray`
                [(name, flux, ra, dec)]
        """
        print('\tPredicting visibilities')
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
        self.predicted = np.ma.masked_array(
            vis_model,
            mask=total_mask
            )
        return

    def calibrate(self, fi=0, vis_model=None):
        """
        """
        if vis_model is None:
            vis_model = self.predict()

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
            # array = np.moveaxis(
            #     np.mean(array, axis=1)[..., 0],
            #     [0, 1, 2],
            #     [2, 0, 1]
            # )
            array = np.moveaxis(
                array[..., 0],
                [0, 1, 2],
                [2, 0, 1]
            )
            return array
        vis = reorder(self.vis[:, fi, ...])
        mod = reorder(self.predicted[:, fi, ...])

        nant = vis.shape[0] #number of antennas
        ant_shape = (nant, nant)
        ntime = vis.shape[2]

        temp = np.ones(ant_shape, dtype=complex)
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

        self.corected = cal_vis
        return


    def make_dirty(self, npix=256, fi=0, data='data', **kwargs):
        """ Make a dirty image from the visibilities

            :param npix:
                Size in pixels of the image
            :type npix: int, optional
            :param fi:
                Frequency index
            :type fi: int, optional
            :param data:
                Data type to be imaged, choose between `'data'`,
                `'corrected'` and `'predicted'`.
            :type data: str, optional
            :param output:
                Type of output `'plot'`,
                `'array'` and `'savefile'` (in .png).
            :type output: str, optional
        """
        self._selection(**kwargs)

        if fi >= self.freq.size:
            raise IndexError(
                'Frequency index exceeds {}'.format(
                    self.freq.size
                )
            )
        self.npix = npix
        self._fi = fi

        na = np.newaxis

        # Average over time
        if data.lower == 'corrected':
            if not hasattr(self, 'corrected'):
                self.calibrate(fi=fi) 
            vis = np.mean(self.corrected, axis=0)
        elif data.lower() == 'predicted':
            if not hasattr(self, 'predicted'):
                self.predict()
            vis = np.mean(self.predicted, axis=0)
        else:
            vis = np.mean(self.vis, axis=0)
        uvw = np.mean(self.uvw, axis=0)
        # Flatten the arrays
        vis = vis[:, self.array.tri_y, self.array.tri_x, :]
        uvw = uvw[:, self.array.tri_y, self.array.tri_x, :]
        # Get Stokes I
        vis = 0.5*(vis[..., 0] + vis[..., 3])
        # # Transform UVW in lambdas units, take fi frequency
        # uvw = uvw[fi, ...] / wavelength(self.freq[fi, na, na])

        # # Prepare l, m grid
        # l = np.linspace(-self.lmax, self.lmax, npix)
        # m = np.linspace(self.mmax, -self.mmax, npix)
        # lg, mg = np.meshgrid(l, m)
        
        # # Make the TF of Visbilities
        # print('\tMaking dirty image')
        # u = uvw[:, 0][:, na, na]
        # v = uvw[:, 1][:, na, na]
        # # Slow:
        # # self.dirty = np.zeros([npix, npix], dtype=np.complex64)
        # # expo = np.exp(
        # #     2j * np.pi * (u * lg[na, ...] + v * mg[na, ...])
        # # )
        # # Faster :
        # # self.dirty = np.zeros([npix, npix], dtype=np.complex64)
        # # lg = lg[na, ...]
        # # mg = mg[na, ...]
        # # pi = np.pi
        # # expo = ne.evaluate('exp(2j*pi*(u*lg+v*mg))')
        # # self.dirty += np.mean(
        # #     vis[fi, :, na, na] * expo,
        # #     axis=0
        # # )
        # # Mutliprocessing
        # expo = mp_expo(self.npix, self.ncpus, lg, mg, u, v)
        # self.dirty = np.mean(
        #     vis[fi, :, na, na] * expo,
        #     axis=0
        # )

        # Transform UVW in lambdas units, take fi frequency
        uvw = uvw / wavelength(self.freq[:, na, na])

        # Prepare l, m grid
        l = np.linspace(-self.lmax, self.lmax, npix)
        m = np.linspace(self.mmax, -self.mmax, npix)
        lg, mg = np.meshgrid(l, m)
        
        # Make the TF of Visbilities
        print('\tMaking dirty image')
        self.dirty = 0
        for fi in range(self.freq.size):
            u = uvw[fi, :, 0][:, na, na]
            v = uvw[fi, :, 1][:, na, na]
            # Slow:
            # self.dirty = np.zeros([npix, npix], dtype=np.complex64)
            # expo = np.exp(
            #     2j * np.pi * (u * lg[na, ...] + v * mg[na, ...])
            # )
            # Faster :
            # self.dirty = np.zeros([npix, npix], dtype=np.complex64)
            # lg = lg[na, ...]
            # mg = mg[na, ...]
            # pi = np.pi
            # expo = ne.evaluate('exp(2j*pi*(u*lg+v*mg))')
            # self.dirty += np.mean(
            #     vis[fi, :, na, na] * expo,
            #     axis=0
            # )
            # Mutliprocessing
            expo = mp_expo(self.npix, self.ncpus, lg, mg, u, v)
            self.dirty += np.mean(
                vis[fi, :, na, na] * expo,
                axis=0
            ) / self.freq.size
            break # only compute image on first frequency 
        return

    def plot_dirty(self, output='plot', altazdir=None, **kwargs):
        """
        """
        obstime = self.cross.time[0] + (self.cross.time[-1] - self.cross.time[0])/2.

        if altazdir is not None:
            # Find pointing information to plot the beam
            from glob import glob
            from astropy.time import Time
            # Get hold of the pointing files
            pointings = sorted(
                glob(
                    join(
                        altazdir,
                        '*.altazA'
                    )
                )
            )
            # Parse the pointings
            time = []
            azimuth = []
            elevation = []
            for pointing in pointings:
                try:
                    t, b, az, el, az1, el1, f, el2 = np.loadtxt(
                        fname=pointing,
                        dtype=str,
                        comments=';',
                        unpack=True
                    )
                except:
                    # TESTANT-type files...
                    t, b, az, el, f, el2 = np.loadtxt(
                        fname=pointing,
                        dtype=str,
                        comments=';',
                        unpack=True
                    )
                time += t.tolist()
                azimuth += az.tolist()
                elevation += el.tolist()
            time = Time(np.array(time))
            azimuth = np.array(azimuth).astype(float)
            elevation = np.array(elevation).astype(float)
            # Remove unused files and keed the last one
            for pointing in sorted(pointings)[:-1]:
                remove(pointing)
            # Mini-Array beam width
            primarybeam = np.degrees(wavelength(np.mean(self.freq)) / 25)
            # Where NenuFAR was aiming
            ti = np.max(np.where((time - obstime).sec <= 0)[0])
            ra_point, dec_point = to_radec(
                alt=elevation[ti],
                az=azimuth[ti],
                time=obstime
            )
            circle = (ra_point, dec_point, primarybeam)
        else:
            circle = None

        astroplt = AstroPlot(
            image=np.real(np.flipud(self.dirty)),
            center=self.phase_center,
            resol=self.fov/self.npix
        )
        
        if output.lower() == 'plot':
            astroplt.plot(
                title='{}'.format(obstime.isot),
                cblabel='Amplitude',
                sources=True,
                time=obstime,
                circle=circle,
                **kwargs
            )
        elif output.endswith('.png'):
            output = abspath(output)
            if isdir(dirname(output)):
                astroplt.plot(
                    title='{}'.format(obstime.isot),
                    cblabel='Amplitude',
                    sources=True,
                    time=obstime,
                    filename=output,
                    circle=circle,
                    **kwargs
                )
        else:
            pass
        return
    
    def savefits_dirty(self, output):
        """
        """
        astroplt = AstroPlot(
            image=np.real(np.flipud(self.dirty)),
            center=self.phase_center,
            resol=self.fov/self.npix
        )
        obstime = self.cross.time[0] + (self.cross.time[-1] - self.cross.time[0])/2.

        if output.endswith('.fits'):
            output = abspath(output)
            # Add time in the filename
            pathname = dirname(output)
            filename = basename(output)
            filename = '_'.join([
                filename.split('.fits')[0],
                obstime.isot.replace(':','-'),
                '.fits'
            ])
            output = join(pathname, filename)
            if isdir(dirname(output)):
                astroplt.savefits(
                    fitsname=output,
                    time=obstime,
                    freq=self.freq[self._fi]
                )
        else:
            pass
        return


    def make_psf(self, npix=None, fi=None):
        """ Make the PSF regarding the UV distribution

            :param npix:
                Size in pixels of the image
            :type npix: int, optional
            :param fi:
                Frequency index
            :type fi: int, optional
        """
        from astropy.modeling.models import Gaussian2D
        from astropy.modeling.fitting import LevMarLSQFitter

        print('\tMaking PSF')
        if fi is None:
            fi = self._fi
        if npix is None:
            npix = self.npix * 2
        na = np.newaxis

        # Average over time
        uvw = np.mean(self.uvw, axis=0)
        # Flatten the array
        uvw = uvw[:, self.array.tri_y, self.array.tri_x, :]
        # Transform UVW in lambdas units, take fi frequency
        uvw = uvw[fi, ...] / wavelength(self.freq[fi, na, na])

        # Prepare l, m grid
        lm_psf = np.cos(np.radians(90 - self.fov))
        l = np.linspace(-lm_psf, lm_psf, npix)
        m = np.linspace(lm_psf, -lm_psf, npix)
        lg, mg = np.meshgrid(l, m)

        # Make the TF of UV distribution
        u = uvw[..., 0][:, na, na]
        v = uvw[..., 1][:, na, na]

        # Slow
        # expo = np.exp(
        #     2j * np.pi * (u * lg[na, ...] + v * mg[na, ...])
        # )
        # Faster
        # lg = lg[na, ...]
        # mg = mg[na, ...]
        # pi = np.pi
        # expo = ne.evaluate('exp(2j*pi*(u*lg+v*mg))')
        # psf = np.real(
        #     np.mean(
        #         expo,
        #         axis=0
        #     )
        # )
        # Mutliprocessing
        expo = mp_expo(npix, self.ncpus, lg, mg, u, v)
        psf = np.real(np.mean(
            expo,
            axis=0
        ))
        self.psf = psf / psf.max()

        # Get clean beam
        print('\tComputing clean beam')
        # Put most of the PSF to 0 to help the fit
        simple_psf = self.psf.copy()
        simple_psf[self.psf <= np.std(self.psf)] = 0 
        nsize = int(simple_psf.shape[0])
        fit_init = Gaussian2D(
            amplitude=1,
            x_mean=npix/2,
            y_mean=npix/2,
            x_stddev=0.2,
            y_stddev=0.2
        )
        fit_algo = LevMarLSQFitter()
        yi, xi = np.indices(simple_psf.shape)
        gaussian = fit_algo(fit_init, xi, yi, simple_psf)
        clean_beam = gaussian(xi, yi)
        clean_beam /= clean_beam.max()
        self.clean_beam = clean_beam[
            int(npix/2/2):int(npix/2*3/2),
            int(npix/2/2):int(npix/2*3/2)
        ]
        return


    def clean(self, maxiter=200, gain=0.1, output='plot', **kwargs):
        from scipy.signal import fftconvolve as convolve

        residual = np.real(self.dirty).copy()
        model = np.zeros(self.dirty.shape)
        nsize = int(residual.shape[0])
        center_id = (
            int(nsize/2), int(nsize/2)
        )
        sig = []
        sigma = 1e10 # huge value to start with
        n = 0
        while (n < maxiter):
            # find peak
            max_id = np.unravel_index(
                residual.argmax(),
                residual.shape
            )
            # flux to subtract
            fsub = residual[max_id] * gain
            
            # shift the PSF
            psf_tmp = np.roll(
                self.psf, max_id[0] - center_id[0],
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
            
            # add a dirac with the flux to the model
            model[max_id] += fsub
            
            # subtract the psf * flux to residuals
            residual -= psf_tmp * fsub
            # compute the new std and keep track of it
            sigma = np.std(residual)
            try:
                if sigma >= sig[-1]:
                #if sigma == sig[-1]:
                    # Undo last iteration
                    model[max_id] -= fsub
                    residual += psf_tmp * fsub
                    break
            except IndexError:
                pass
            sig.append(sigma)
            n += 1
        
        conv = convolve(model, self.clean_beam, mode='same')
        conv *= ((np.real(self.dirty)-residual).max() / conv.max())
        self.restored = conv + residual
        
        astroplt = AstroPlot(
            image=np.real(np.flipud(self.restored)),
            center=self.phase_center,
            resol=self.fov/nsize
        ) 

        if output.lower() == 'plot':
            obstime = self.cross.time[0] + (self.cross.time[-1] - self.cross.time[0])/2.
            astroplt.plot(
                title='{}'.format(obstime.isot),
                cblabel='Amplitude',
                **kwargs
            )
        elif output.lower() == 'array':
            return self.restored
        elif output.endswith('.png'):
            output = abspath(output)
            if isdir(dirname(output)):
                astroplt.savepng(
                    filename=output
                )
            return
        elif output.endswith('.fits'):
            output = abspath(output)
            if isdir(dirname(output)):
                astroplt.savefits(
                    filename=output
                )
            return
        else:
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


    def make_nearfield(self, filename='', radius=300, npix=128):
        """
            :param radius:
                Radius of nearfield in meters
            :type radius: float, optional
            :param npix:
                Size in pixels of the image
            :type npix: int, optional
        """
        import matplotlib.pyplot as plt
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.ticker import LinearLocator
        from matplotlib.colors import Normalize
        from matplotlib.cm import get_cmap
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        antpos = self.array.ma_enu.copy()
        
        # Compute the delays at the ground
        z = np.average(antpos[:, 2]) + 1
        x = np.linspace(-radius, radius, npix)
        y = np.linspace(-radius, radius, npix)
        posx, posy = np.meshgrid(x, y)
        delay = (antpos[:, 0][:, None, None] - posx[None]) ** 2 \
            + (antpos[:, 1][:, None, None] - posy[None]) ** 2 \
            + (antpos[:, 2][:, None, None] - z) ** 2
        delay = np.sqrt(delay)

        # Compute delays between antenna pairs
        k = 0
        nant = self.array.antennas.size
        dd = np.zeros(( int(nant*(nant-1)/2+nant), npix, npix))
        for i in range(nant):
            for j in range(i, nant):
                dd[k] = np.array([delay[j] - delay[i]])
                k += 1
        
        # Multi-frequencies
        vis = np.mean(self.vis, axis=0) # mean in time
        vis_i = 0.5*(vis[..., 0] + vis[..., 3])
        j2pi = 2.j * np.pi
        nearfield = 0
        n = vis_i.shape[0]
        for j in range(n):
            v = vis_i[j, self.array.tri_x, self.array.tri_y][:, None, None]
            d = dd[None, ...]
            lamb = wavelength(self.freq[j])
            # Multiproc
            h = v * mp_expo_nf(npix, self.ncpus, d, lamb)
            # Slow
            # h = ne.evaluate('v*exp(j2pi*d/lamb)')
            nearfield += h
        nearfield = np.sum(np.abs(nearfield[0] / len(self.freq)), axis=0) # /mean

        # Plot
        loc_buildings_enu = np.array([
            [27.75451691, -51.40993459, 7.99973228],
            [20.5648047, -59.79299576, 7.99968629],
            [167.86485612, 177.89170175, 7.99531119]
        ])
        fig, ax = plt.subplots(figsize=(10, 10))
        nearfield_db = 10*np.log10(nearfield)
        ax.imshow(
            np.fliplr(nearfield_db),
            cmap='YlGnBu_r',
            extent=[-radius, radius, -radius, radius]
        )
        # Colorbar
        cax = inset_axes(ax,
           width='5%',
           height='100%',
           loc='lower left',
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax.transAxes,
           borderpad=0,
           )
        cb = ColorbarBase(
            cax,
            cmap=get_cmap(name='YlGnBu_r'),
            orientation='vertical',
            norm=Normalize(
                vmin=np.min(nearfield_db),
                vmax=np.max(nearfield_db)
            ),
            ticks=LinearLocator()
        )
        cb.solids.set_edgecolor('face')
        cb.set_label('dB')
        cb.formatter.set_powerlimits((0, 0))
        # Array info
        ax.scatter(
            self.array.ma_enu[:, 0],
            self.array.ma_enu[:, 1],
            color='tab:red'
        )
        ax.scatter(
            loc_buildings_enu[:, 0],
            loc_buildings_enu[:, 1],
            color='tab:orange'
        )
        for i in range(self.array.ant_names.size):
            ax.text(
                self.array.ma_enu[i, 0],
                self.array.ma_enu[i, 1],
                ' ' + self.array.ant_names[i].replace('MR', ''))
        ax.set_xlabel(r'$\Delta x$ (m)')
        ax.set_ylabel(r'$\Delta y$ (m)')
        obstime = self.cross.time[0] + (self.cross.time[-1] - self.cross.time[0])/2.
        ax.set_title('{}'.format(obstime.isot))
        plt.savefig(filename, dpi=300, bbox_inches='tight')
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



