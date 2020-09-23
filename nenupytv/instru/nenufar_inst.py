#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2019, nenupytv'
__credits__ = ['Alan Loh', 'Julien Girard']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'ma_names',
    'ma_positions',
    'ma_indices',
    'nenufar_pos',
    'ant_pos',
    'ma_positions_enu'
    ]


import numpy as np
from os.path import dirname, join
from astropy import units as u
from astropy.coordinates import EarthLocation 

from nenupytv.astro import rotz


location_file  = join(
    dirname(__file__),
    'location_miniarrays.dat'
    )

def parse_file(lfile):
    """ Parse the location_miniarrays.dat file

        Returns
        -------
        ma : dict
            Dictionnary of mini-array indices and positions
    """
    ma_dict = {
        'DBCODE': [],
        'X': [],
        'Y': [],
        'Z': []
        }

    with open(lfile) as f:
        line = f.readline()
        while line:
            for key in ma_dict.keys():
                search = key + '='
                if search in line:
                    val = line.split(search)[1].split()[0]
                    ma_dict[key].append(val)
            line = f.readline()

    positions = np.zeros((len(ma_dict['DBCODE']), 3))
    positions[:, 0] = np.array(ma_dict['X'], dtype='float64')
    positions[:, 1] = np.array(ma_dict['Y'], dtype='float64')
    positions[:, 2] = np.array(ma_dict['Z'], dtype='float64')

    names = np.array(
        [name.strip("'") for name in ma_dict['DBCODE']],
        dtype='str'
        )

    indices = np.array(
        [ma.replace('MR', '') for ma in names],
        dtype='int'
        )

#    return rotz(positions, 90), names, indices
    return positions, names, indices

gps_b = 6356752.31424518
gps_a = 6378137
e_squared = 6.69437999014e-3

def XYZ_from_LatLonAlt(latitude, longitude, altitude):
    """
    from pyuvdata
    Calculate ECEF x,y,z from lat/lon/alt values.

    Parameters
    ----------
    latitude :  ndarray or float
        latitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
    longitude :  ndarray or float
        longitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
    altitude :  ndarray or float
        altitude, numpy array (if Npts > 1) or value (if Npts = 1) in meters

    Returns
    -------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.

    """
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    altitude = np.array(altitude)
    n_pts = latitude.size
    if longitude.size != n_pts:
        raise ValueError(
            "latitude, longitude and altitude must all have the same length"
        )
    if altitude.size != n_pts:
        raise ValueError(
            "latitude, longitude and altitude must all have the same length"
        )

    # see wikipedia geodetic_datum and Datum transformations of
    # GPS positions PDF in docs/references folder
    gps_n = gps_a / np.sqrt(1 - e_squared * np.sin(latitude) ** 2)
    xyz = np.zeros((n_pts, 3))
    xyz[:, 0] = (gps_n + altitude) * np.cos(latitude) * np.cos(longitude)
    xyz[:, 1] = (gps_n + altitude) * np.cos(latitude) * np.sin(longitude)
    xyz[:, 2] = (gps_b ** 2 / gps_a ** 2 * gps_n + altitude) * np.sin(latitude)

    xyz = np.squeeze(xyz)
    return xyz


def ENU_from_ECEF(xyz, latitude, longitude, altitude):
    """
    from pyuvdata
    Calculate local ENU (east, north, up) coordinates from ECEF coordinates.

    Parameters
    ----------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.
    latitude : float
        Latitude of center of ENU coordinates in radians.
    longitude : float
        Longitude of center of ENU coordinates in radians.
    altitude : float
        Altitude of center of ENU coordinates in radians.

    Returns
    -------
    ndarray of float
        numpy array, shape (Npts, 3), with local ENU coordinates

    """
    xyz = np.array(xyz)
    if xyz.ndim > 1 and xyz.shape[1] != 3:
        raise ValueError("The expected shape of ECEF xyz array is (Npts, 3).")

    xyz_in = xyz

    if xyz_in.ndim == 1:
        xyz_in = xyz_in[np.newaxis, :]

    # check that these are sensible ECEF values -- their magnitudes need to be
    # on the order of Earth's radius
    ecef_magnitudes = np.linalg.norm(xyz_in, axis=1)
    sensible_radius_range = (6.35e6, 6.39e6)
    if np.any(ecef_magnitudes <= sensible_radius_range[0]) or np.any(
        ecef_magnitudes >= sensible_radius_range[1]
    ):
        raise ValueError(
            "ECEF vector magnitudes must be on the order of the radius of the earth"
        )

    xyz_center = XYZ_from_LatLonAlt(latitude, longitude, altitude)

    xyz_use = np.zeros_like(xyz_in)
    xyz_use[:, 0] = xyz_in[:, 0] - xyz_center[0]
    xyz_use[:, 1] = xyz_in[:, 1] - xyz_center[1]
    xyz_use[:, 2] = xyz_in[:, 2] - xyz_center[2]

    enu = np.zeros_like(xyz_use)
    enu[:, 0] = -np.sin(longitude) * xyz_use[:, 0] + np.cos(longitude) * xyz_use[:, 1]
    enu[:, 1] = (
        -np.sin(latitude) * np.cos(longitude) * xyz_use[:, 0]
        - np.sin(latitude) * np.sin(longitude) * xyz_use[:, 1]
        + np.cos(latitude) * xyz_use[:, 2]
    )
    enu[:, 2] = (
        np.cos(latitude) * np.cos(longitude) * xyz_use[:, 0]
        + np.cos(latitude) * np.sin(longitude) * xyz_use[:, 1]
        + np.sin(latitude) * xyz_use[:, 2]
    )
    if len(xyz.shape) == 1:
        enu = np.squeeze(enu)

    return enu


ma_positions, ma_names, ma_indices = parse_file(location_file)


nenufar_pos = EarthLocation(
            lat=47.375944 * u.deg,
            lon=2.193361 * u.deg,
            height=136.195 * u.m
            )

# Convert to ENU
# _mapos = EarthLocation.from_geocentric(
#     ma_positions[:, 0],
#     ma_positions[:, 1],
#     ma_positions[:, 2],
#     unit=u.m
# )
# _mapos = np.array(_mapos.value.tolist())
ma_positions_enu = ENU_from_ECEF(
    xyz=ma_positions,#_mapos,
    latitude=np.radians(47.376511),#nenufar_pos.lat.rad,
    longitude=np.radians(2.192400),#nenufar_pos.lon.rad,
    altitude=150.00#nenufar_pos.height.value
)



ant_pos = np.array([
    -5.50000000e+00, -9.52627850e+00,  0.00000000e+00,
     0.00000000e+00, -9.52627850e+00,  0.00000000e+00,
     5.50000000e+00, -9.52627850e+00,  0.00000000e+00,
    -8.25000000e+00, -4.76313877e+00,  0.00000000e+00,
    -2.75000000e+00, -4.76313877e+00,  0.00000000e+00,
     2.75000000e+00, -4.76313877e+00,  0.00000000e+00,
     8.25000000e+00, -4.76313877e+00,  0.00000000e+00,
    -1.10000000e+01,  9.53674316e-07,  0.00000000e+00,
    -5.50000000e+00,  9.53674316e-07,  0.00000000e+00,
     0.00000000e+00,  9.53674316e-07,  0.00000000e+00,
     5.50000000e+00,  9.53674316e-07,  0.00000000e+00,
     1.10000000e+01,  9.53674316e-07,  0.00000000e+00,
    -8.25000000e+00,  4.76314068e+00,  0.00000000e+00,
    -2.75000000e+00,  4.76314068e+00,  0.00000000e+00,
     2.75000000e+00,  4.76314068e+00,  0.00000000e+00,
     8.25000000e+00,  4.76314068e+00,  0.00000000e+00,
    -5.50000000e+00,  9.52628040e+00,  0.00000000e+00,
     0.00000000e+00,  9.52628040e+00,  0.00000000e+00,
     5.50000000e+00,  9.52628040e+00,  0.00000000e+00
    ]).reshape(19, 3)

