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
    'lha'
    ]


from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, Angle


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
            Local sidereal time in hour angle
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
    return lst.hourangle


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
            Local hour angle
    """
    ra = Angle(ra * u.deg).hourangle
    ha = lst(time, location) - ra
    if ha < 0:
        ha += 360.
    elif ha > 360:
        ha -= 360.
    return ha

