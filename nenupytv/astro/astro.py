#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'lst',
    'lha',
    'ho_zenith',
    'eq_zenith',
    'eq_coord',
    'to_radec',
    'to_altaz',
    'to_gal',
    'rotz',
    'wavelength',
    'ref_location'
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
    ICRS
)
from astropy.constants import c as lspeed


# ============================================================= #
# ---------------------------- lst ---------------------------- #
# ============================================================= #
def lst(time, location):
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
def lha(time, location, ra):
    """ Local hour angle of an object in the observer's sky

        Parameters
        ----------
        time : `astropy.time.Time`
            UTC time
        location : `astropy.coord.EarthLocation`
            Location of the instrument
        ra : float
            Right Ascension in degrees

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
def ho_zenith(time, location):
    """ Horizontal coordinates of zenith
    """
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
def eq_zenith(time, location):
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
    eq = ICRS(
        ra=ra*u.deg,
        dec=dec*u.deg
    )
    return eq.ra.deg, eq.dec.deg
# ============================================================= #


# ============================================================= #
# ------------------------- eq_zenith ------------------------- #
# ============================================================= #
def to_radec(alt, az, time, location):
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
def to_altaz(ra, dec, time, location):
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
