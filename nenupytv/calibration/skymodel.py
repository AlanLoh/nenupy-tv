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
    'Skymodel'
    ]


from os.path import dirname, realpath, join, isfile
from astropy.table import Table, Column
import astropy.units as u
from urllib.request import urlopen
import numpy as np

from nenupytv.astro import radec, radec_hd
from nenupytv.instru import nenufar_pos

# ============================================================= #
# ------------------------- Skymodel -------------------------- #
# ============================================================= #
class Skymodel(object):
    """
    """

    def __init__(self, ra, dec, radius, cutoff=100.):
        self.ra = ra
        self.dec = dec
        self.radius = radius
        self.cutoff = cutoff

        self.skymodel_file = join(
            dirname(realpath(__file__)),
            'lfsky.fits'
        )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    

    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def update(self, override=False):
        """ This takes a looong time.
        """
        from tqdm import tqdm

        if isfile(self.skymodel_file):
            t = Table.read(self.skymodel_file)
            if override:
                pass
            elif t.meta['CUTOFF'] <= self.cutoff:
                # Dont recompute it, it already have the sources
                # up to the required flux cutoff.
                return
            del t

        model_dict = {}

        ra, dec, res = self._hpx_coord(resolution=4)
        for r, d in tqdm(zip(ra, dec), total=ra.size):
            # Check if RA/Dec will ever be visible
            if d < nenufar_pos.lat.deg - 90:
                continue

            sources = self._from_gsm(
                ra=r,
                dec=d,
                radius=res*2,
                cutoff=self.cutoff
            )
            for source in sources.keys():
                if 'name' not in model_dict.keys():
                    model_dict['name'] = []
                elif source in model_dict['name']:
                    # Source is already registered
                    continue
                else:
                    pass
                model_dict['name'].append(source)
                for key in sources[source].keys():
                    if key not in model_dict.keys():
                        model_dict[key] = []
                    model_dict[key].append(sources[source][key])

        # Convert in numpy arrays
        for key in model_dict.keys():
            model_dict[key] = np.array(model_dict[key])

        t = Table(
            [
                Column(
                    model_dict[k],
                    name=k,
                    dtype=model_dict[k].dtype.str
                ) for k in model_dict.keys()
            ],
            meta={'cutoff': self.cutoff}
        )
        t.write(self.skymodel_file, overwrite=True)
        return


    def get_skymodel(self, freq=None):
        """
        """
        sky = Table.read(self.skymodel_file)

        all_coords = radec(sky['ra'], sky['dec'])
        phase_center = radec(self.ra, self.dec)
        sep = all_coords.separation(phase_center).deg
        infield_mask = sep <= self.radius

        infield_sky = sky[infield_mask]
        flux = self._extrapol_spec(
            freq=freq,
            rflux=infield_sky['flux'],
            rfreq=infield_sky['rfreq'],
            index=infield_sky['index']
        )
        fluxmask = flux >= self.cutoff
        infield_sky = infield_sky[fluxmask]
        flux = flux[fluxmask]

        names = np.zeros(len(infield_sky), dtype=str)
        skymodel = np.zeros(
            (len(infield_sky), 3)
        )
        for i in range(len(infield_sky)):
            names[i] = infield_sky['name'][i]
            skymodel[i, 0] = flux[i]
            skymodel[i, 1] = infield_sky['ra'][i]
            skymodel[i, 2] = infield_sky['dec'][i]

        return skymodel, names


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _hpx_coord(resolution):
        """ Returns all healpix coordinates of a sphere at a
            resolution close to `resolution` in degrees.
        """
        from healpy import (
            nside2resol,
            nside2npix,
            pix2ang
        )
        nsides = np.array([2**i for i in range(1, 12)])
        resol_rad = nside2resol(
            nside=nsides,
            arcmin=False
        )
        resol_deg = np.degrees(resol_rad)
        idx = (np.abs(resol_deg - resolution)).argmin()
        resol = resol_deg[idx]
        nside = nsides[idx]
        pixels = nside2npix(nside=nside)
        ra, dec = pix2ang(
            nside=nside,
            ipix=np.arange(pixels),
            lonlat=True
        )
        return ra, dec, resol


    @staticmethod
    def _from_gsm(ra, dec, radius, cutoff):
        """ Send requests to LCS165, with a given RA/Dec in
            degrees, a radius and a cutoff in Jy.
        """
        command = [
            'https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?',
            'coord=' + str(ra),
            ',' + str(dec),
            '&radius=' + str(radius),
            '&cutoff=' + str(cutoff),
            '&unit=deg&deconv=y',
        ]
        url = ''.join(command)
        with urlopen(url) as response:
            response = response.read().decode('utf8')
            sources = response.split('\n')[3:]

        infield = {}
        for source in sources:
            if source == '':
                continue
            src_desc = source.split(', ')
            src_name = src_desc[0]
            infield[src_name] = {}
            infield[src_name]['flux'] = float(src_desc[4])
            src = radec_hd(
                src_desc[2] + ' ' + src_desc[3].replace(
                    '.',
                    ':',
                    2
                )
            )
            infield[src_name]['ra'] = src.ra.deg
            infield[src_name]['dec'] = src.dec.deg

            ref_freq = float(src_desc[8]) if src_desc[8] != '' else 60e6
            infield[src_name]['rfreq'] = ref_freq
            spec_index = float(
                src_desc[9].split(',')[0].replace('[', '').replace(']', '')
            )
            infield[src_name]['index'] = spec_index
        
        return infield


    @staticmethod
    def _extrapol_spec(freq, rflux, rfreq, index):
        """ Given the `rflux` in Jy at the `rfreq` in MHz,
            and the spectral index, extrapolate the `flux`
            at `freq` MHz
        """
        if freq is None:
            return rflux
        freq *= u.MHz
        rfreq *= u.Hz
        return (rflux * (freq / rfreq.to(u.MHz))**index).value

