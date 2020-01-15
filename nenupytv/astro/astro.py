#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'nenufar_loc',
    'lst',
    'lha',
    'ho_zenith',
    'eq_zenith',
    'radec',
    'eq_coord',
    'to_radec',
    'to_altaz',
    'to_gal',
    'rotz',
    'wavelength',
    'ref_location',
    'radio_sources',
    'ateam',
    'astro_image'
    ]


import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (
    EarthLocation,
    Angle,
    SkyCoord,
    AltAz,
    Galactic,
    ICRS,
    get_body
)
from astropy.constants import c as lspeed
from astropy.wcs import WCS


# ============================================================= #
# ------------------------ nenufar_loc ------------------------ #
# ============================================================= #
def nenufar_loc():
    """
    """
    return EarthLocation(
        lat=47.375944 * u.deg,
        lon=2.193361 * u.deg,
        height=136.195 * u.m
    )
# ============================================================= #


# ============================================================= #
# ---------------------------- lst ---------------------------- #
# ============================================================= #
def lst(time, location=None):
    """ Local sidereal time

        Parameters
        ----------
        time : `astropy.time.Time`
            UTC time
        location : `astropy.coord.EarthLocation`
            Location of the instrument

        Returns
        -------
        lst : float
            Local sidereal time in degrees
    """
    if not isinstance(time, Time):
        raise TypeError(
            'time is not an astropy Time.'
            )
    if location is None:
        location = nenufar_loc()
    if not isinstance(location, EarthLocation):
        raise TypeError(
            'time is not an astropy EarthLocation.'
            )
    lon = location.to_geodetic().lon
    lst = time.sidereal_time('apparent', lon)
    return lst.deg#.hourangle
# ============================================================= #


# ============================================================= #
# ---------------------------- lha ---------------------------- #
# ============================================================= #
def lha(time, ra, location=None):
    """ Local hour angle of an object in the observer's sky

        Parameters
        ----------
        time : `astropy.time.Time`
            UTC time
        ra : float
            Right Ascension in degrees
        location : `astropy.coord.EarthLocation`, optional
            Location of the instrument

        Returns
        -------
        lha : float
            Local hour angle in degrees
    """
    ra = Angle(ra * u.deg).deg#.hourangle
    ha = lst(time, location) - ra
    if ha < 0:
        ha += 360.
    elif ha > 360:
        ha -= 360.
    return ha
# ============================================================= #


# ============================================================= #
# ------------------------- ho_zenith ------------------------- #
# ============================================================= #
def ho_zenith(time, location=None):
    """ Horizontal coordinates of zenith
    """
    if location is None:
        location = nenufar_loc()
    if not isinstance(location, EarthLocation):
        raise TypeError(
            'time is not an astropy EarthLocation.'
            )
    altaz = AltAz(
        az=0.*u.deg,
        alt=90.*u.deg,
        location=location,
        obstime=time
    )
    return altaz
# ============================================================= #


# ============================================================= #
# ------------------------- eq_zenith ------------------------- #
# ============================================================= #
def eq_zenith(time, location=None):
    """ Get the ra dec coordinates of the zenith
        
        Parameters
        ----------
        time : `astropy.time.Time`
            UTC time
        location : `astropy.coord.EarthLocation`
            Location of the instrument

        Returns
        -------
        ra : float
            Right Ascension in degrees
        dec : float
            Declination in degrees
    """
    if location is None:
        location = nenufar_loc()
    if not isinstance(location, EarthLocation):
        raise TypeError(
            'time is not an astropy EarthLocation.'
            )

    zen_alt = 90*u.deg
    zen_az = 0*u.deg
    azel = SkyCoord(
        alt=zen_alt,
        az=zen_az,
        obstime=time,
        location=location,
        frame='altaz'
        )
    eq = azel.icrs
    return eq.ra.deg, eq.dec.deg
# ============================================================= #


# ============================================================= #
# ------------------------- eq_coord -------------------------- #
# ============================================================= #
def radec(ra, dec):
    """ Equatorial coordinates
        
        :param ra:
            Right ascension in degrees
        :type ra: float
        :param dec:
            Declination in degrees
        :type dec: float

        :returns: :class:`astropy.coordinates.ICRS` object
        :rtype: :class:`astropy.coordinates.ICRS`

        :Example:
        
        >>> from nenupysim.astro import eq_coord
        >>> radec = eq_coord(
                ra=51,
                dec=39,
            )
    """
    return ICRS(
        ra=ra*u.deg,
        dec=dec*u.deg
    )
# ============================================================= #


# ============================================================= #
# ------------------------- eq_coord -------------------------- #
# ============================================================= #
def eq_coord(ra, dec):
    """ Equatorial coordinates
        
        :param ra:
            Right ascension in degrees
        :type ra: float
        :param dec:
            Declination in degrees
        :type dec: float

        :returns: :class:`astropy.coordinates.ICRS` object
        :rtype: :class:`astropy.coordinates.ICRS`

        :Example:
        
        >>> from nenupysim.astro import eq_coord
        >>> radec = eq_coord(
                ra=51,
                dec=39,
            )
    """
    eq = radec(
        ra=ra*u.deg,
        dec=dec*u.deg
    )
    return eq.ra.deg, eq.dec.deg
# ============================================================= #


# ============================================================= #
# ------------------------- eq_zenith ------------------------- #
# ============================================================= #
def to_radec(alt, az, time, location=None):
    """ Get the ra dec coordinates of the a altaz pointing
        
        Parameters
        ----------
        alt : float
            Elevation in degrees
        az : float
            Azimuth in degrees
        time : `astropy.time.Time`
            UTC time
        location : `astropy.coord.EarthLocation`
            Location of the instrument

        Returns
        -------
        ra : float
            Right Ascension in degrees
        dec : float
            Declination in degrees
    """
    if location is None:
        location = nenufar_loc()
    if not isinstance(location, EarthLocation):
        raise TypeError(
            'time is not an astropy EarthLocation.'
            )

    zen_alt = alt*u.deg
    zen_az = az*u.deg
    azel = SkyCoord(
        alt=zen_alt,
        az=zen_az,
        obstime=time,
        location=location,
        frame='altaz'
        )
    eq = azel.icrs
    return eq.ra.deg, eq.dec.deg
# ============================================================= #


# ============================================================= #
# ------------------------- to_altaz -------------------------- #
# ============================================================= #
def to_altaz(ra, dec, time, location=None):
    """ Transform altaz coordinates to ICRS equatorial system
        
        :param radec:
            Equatorial coordinates
        :type altaz: :class:`astropy.coordinates.ICRS`
        :param time:
            Time at which the local coordinates should be 
            computed. It can either be provided as an 
            :class:`astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: str, :class:`astropy.time.Time`

        :returns: :class:`astropy.coordinates.AltAz` object
        :rtype: :class:`astropy.coordinates.AltAz`

        :Example:
        
        >>> from nenupysim.astro import eq_coord
        >>> radec = eq_coord(
                ra=51,
                dec=39,
            )
    """
    if location is None:
        location = nenufar_loc()
    if not isinstance(location, EarthLocation):
        raise TypeError(
            'time is not an astropy EarthLocation.'
            )

    altaz_frame = AltAz(
        obstime=time,
        location=location
    )
    eq = ICRS(
        ra=ra*u.deg,
        dec=dec*u.deg
    )
    altaz = eq.transform_to(altaz_frame)
    return altaz
# ============================================================= #


# ============================================================= #
# -------------------------- to_gal --------------------------- #
# ============================================================= #
def to_gal(src):
    """ Convert an astorpy source to galactic cooridnates
    """
    return src.transform_to(Galactic)
# ============================================================= #


# ============================================================= #
# --------------------------- rotz ---------------------------- #
# ============================================================= #
def rotz(array, angle):
    """ Rotate the 3D array by an angle along z-axis
    """
    ang = np.radians(angle)
    cosa = np.cos(ang)
    sina = np.sin(ang)
    rot = np.array([
            [cosa, -sina, 0],
            [sina,  cosa, 0],
            [   0,     0, 1]
        ])
    return np.dot(array, rot)
# ============================================================= #


# ============================================================= #
# ------------------------ wavelength ------------------------- #
# ============================================================= #
def wavelength(freq):
    """ Convert between MHz and wavelength in meters

        Returns
        -------
        wavelength : `np.ndarray`
            Wavelength in meters
    """
    if not hasattr(freq, '__len__'):
        freq = [freq]
    if not isinstance(freq, np.ndarray):
        freq = np.array(freq)
    freq *= u.MHz
    freq = freq.to(u.Hz)
    wavel = lspeed.value / freq.value
    return wavel
# ============================================================= #


# ============================================================= #
# ------------------------ ref_location ----------------------- #
# ============================================================= #
def ref_location():
    """
    """
    return EarthLocation(
        lat=0*u.deg,
        lon=-90*u.deg,
        height=0*u.m
    )
# ============================================================= #


# ============================================================= #
# ----------------------- radio_sources ----------------------- #
# ============================================================= #
def radio_sources(time):
    """
    """
    if not isinstance(time, Time):
        time = Time(time)

    def solarsyst_eq(src, time):
        src = get_body(
            src,
            time,
            nenufar_loc()
        )
        return src.ra.deg, src.dec.deg

    with solar_system_ephemeris.set('builtin'):
        src_radec = {
            'vira': (187.70593075, +12.39112331),
            'cyga': (299.86815263, +40.73391583),
            'casa': (350.850000, +58.815000),
            'hera': (252.783433, +04.993031),
            'hyda': (139.523546, -12.095553),
            'taua': (83.63308, +22.01450),
            'sun': solarsyst_eq('sun', time),
            'moon': solarsyst_eq('moon', time),
            'jupiter': solarsyst_eq('jupiter', time),
        }
    return src_radec
# ============================================================= #


# ============================================================= #
# --------------------------- ateam --------------------------- #
# ============================================================= #
def ateam():
    """
    """
    src_radec = {
        'vir a': (187.70593075, +12.39112331),
        'cyg a': (299.86815263, +40.73391583),
        'cas a': (350.850000, +58.815000),
        'her a': (252.783433, +04.993031),
        'hyd a': (139.523546, -12.095553),
        'tau a': (83.63308, +22.01450)
    }
    return src_radec
# ============================================================= #


# ============================================================= #
# ------------------------ astro_image ------------------------ #
# ============================================================= #
def astro_image(image, center, npix, resol, savefile=None, **kwargs):
    """
    """
    import matplotlib.pyplot as plt
    # if 'vmin' not in kwargs.keys():
    #     kwargs['vmin'] = np.percentile(specdata.amp, 5)
    # if 'vmax' not in kwargs.keys():
    #     kwargs['vmax'] = np.percentile(specdata.amp, 95)
    if 'cmap' not in kwargs.keys():
        kwargs['cmap'] = 'YlGnBu_r'

    w = WCS(naxis=2)
    w.wcs.crpix = [npix/2, npix/2]
    w.wcs.cdelt = np.array([resol, resol])
    w.wcs.crval = [center[0], center[1]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=w)
    ax.imshow(
        image,
        origin='lower',
        aspect='equal',
        **kwargs
    )

    ax.coords.grid(True, color='white', ls='solid', alpha=0.5)
    axra = ax.coords[0]
    axdec = ax.coords[1]
    axra.set_axislabel('RA')
    axra.set_major_formatter('d')
    axra.set_ticks(number=10)
    axdec.set_axislabel('Dec')
    axdec.set_major_formatter('d')
    axdec.set_ticks(number=10)

    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile, **kwargs)
# ============================================================= #

