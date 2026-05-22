"""
Author: Dr. Christian Kehl
Date: 06-07-2025
"""
# from argparse import ArgumentParser
from glob import glob
import math
import datetime
import numpy as np

import xarray as xr
# import dask.array as da
from netCDF4 import Dataset
import h5py

import gc
import os
import scipy.spatial.qhull as qhull
from scipy.interpolate import interpn, griddata

DBG_MSG = False

def interp_weights(xyz, uvw):
    d = 3
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

coords_points = [(13.175806, -20.196043),
                 (13.265418, -20.489281),
                 (13.445512, -20.911225),
                 (13.622055, -21.191146),
                 (13.795100, -21.415476),
                 (14.252816, -22.096660),
                 (14.519789, -22.690266),
                 (14.526929, -22.825006),
                 (14.477003, -23.247734),
                 (14.469232, -23.356596),
                 (15.136016, -26.610156),
                 (16.453795, -28.641185),
                 (16.721975, -28.982585),
                 (16.855950, -29.259383),
                 (17.249728, -30.263043),
                 (17.262927, -30.318351),
                 (17.351370, -30.472325),
                 (17.569327, -30.848735),
                 (18.182380, -31.707337),
                 (18.228367, -31.812545),
                 (18.293050, -32.093162),
                 (18.323014, -32.304017),
                 (18.113211, -32.751982),
                 (17.918734, -33.078900),
                 (18.373475, -33.858867),
                 (18.304258, -34.090465),
                 (18.340500, -34.263150),
                 (18.476748, -34.376803),
                 (18.857755, -34.416841),
                 (19.281130, -34.640196),
                 (20.042563, -34.860687),
                 (20.901432, -34.406152),

                 (20.913995,-34.414499),
                 (20.918974,-34.425296),
                 (20.923609,-34.431137),
                 (20.923394,-34.453894),
                 (20.919446,-34.476857),
                 (20.906056,-34.501406),
                 (20.874127,-34.523755),
                 (20.805978,-34.544542),
                 (20.762547,-34.552460),
                 (20.656289,-34.546522),
                 (20.578355,-34.562356),
                 (20.537156,-34.621991),
                 (20.488404,-34.684970),
                 (20.385236,-34.729140),
                 (20.370988,-34.757493),
                 (20.255288,-34.850521),
                 (20.193146,-34.914312),
                 (20.121392,-34.961314),
                 (20.000371,-34.969754),
                 (19.889992,-34.939364),
                 (19.781846,-34.876015),
                 (19.647092,-34.905303),
                 (19.507703,-34.861509),
                 (19.382733,-34.768633),
                 (19.263257,-34.761723),
                 (19.128159,-34.718700),
                 (19.076146,-34.600798),
                 (19.143437,-34.494190),
                 (19.058808,-34.501264),
                 (18.950661,-34.467305),
                 (18.874615,-34.486550),
                 (18.745354,-34.479475),
                 (18.617295,-34.391837),
                 (18.584679,-34.278721),
                 (18.597039,-34.298861),
                 (18.641156,-34.449187),
                 (18.250455,-34.454850),
                 (18.171834,-34.332747),
                 (18.118790,-34.186471),
                 (18.082570,-34.048903),
                 (18.164452,-33.907407),
                 (18.156384,-33.794212),
                 (18.137501,-33.656293),
                 (18.073815,-33.510425),
                 (17.867650,-33.359006),
                 (17.806710,-33.221972),
                 (17.784566,-33.071202),
                 (17.677964,-32.926657),
                 (17.666806,-32.802944),
                 (17.738045,-32.668510),
                 (17.978200,-32.561512),
                 (18.090638,-32.572073),
                 (18.118962,-32.459600),
                 (18.088406,-32.314786),
                 (18.013734,-32.033050),
                 (17.934083,-31.739787),
                 (17.771519,-31.553611),
                 (17.448624,-31.211713),
                 (17.151478,-30.645236),
                 (17.017239,-30.363928),
                 (16.769360,-29.804393),
                 (16.658685,-29.432156),
                 (16.178891,-29.290774),
                 (14.985501,-28.550936),
                 (14.180066,-27.600381),
                 (13.935620,-26.646419),
                 (13.671949,-25.775968),
                 (13.296354,-23.923695),
                 (12.726438,-21.600177),
                 (12.317197,-20.947529)] # ,
                 # (11.090162,-19.031811),
                 # (10.646589,-17.915570),
                 # (10.577237,-16.646920),
                 # (10.766065,-15.696628),
                 # (11.390912,-13.101085),
                 # (10.808637,-12.180513),
                 # (9.710004 , -5.464472)]

def interpolate_bathymetry(array_coord_tuples, bathytopo_path):
    coord_array = np.array(array_coord_tuples)
    lons = coord_array[:, 0]
    lats = coord_array[:, 1]
    bottom_coords = None
    halfway_coords = None

    target_coords = (np.squeeze(lats), np.squeeze(lons))

    # bathytopo_path = os.path.join("/media","christian", "DATA", "Documents", "LifeStages", "RUG", "Research", "SouthernOcean", "gebco_2026_n-20.0_s-65.0_w-15.0_e45.0.nc")
    btopo_file = xr.open_dataset(bathytopo_path, decode_cf=True, engine='netcdf4')
    fX = btopo_file.variables["lon"]
    fY = btopo_file.variables["lat"]
    fBTOPO = btopo_file.variables["elevation"]
    # mgrid = np.array([fY, fX], dtype=fX.dtype).T
    # if DBG_MSG:
    #     print("Compute triangulation P0 ...")
    # tri0 = Delaunay(mgrid0)
    # if DBG_MSG:
    #     print("Interpolating U0 ...")
    # F0_interp = LinearNDInterpolator(tri0, fF0t)
    # Fs_local_0 = F0_interp(gcenters_arr)
    with np.errstate(invalid='ignore'):
        fBTOPO = np.nan_to_num(fBTOPO, nan=0.0)
    mgrid0 = (fY, fX)
    if DBG_MSG:
        print("mgrid0 dims = ({}, {})".format(mgrid0[0].shape, mgrid0[1].shape))
    bottom_coords = -interpn(mgrid0, fBTOPO.squeeze(), target_coords, method='linear', fill_value=.0)
    if DBG_MSG:
        print("mgrid0: min_0 - max_0: {} - {}; min_1 - max_1: {} - {};".format(np.min(np.array(fY)), np.max(np.array(fY)), np.min(np.array(fX)), np.max(np.array(fX))))
        print("gcenters: min_0 - max_0: {} - {}; min_1 - max_1: {} - {};".format(np.min(target_coords[0]), np.max(target_coords[0]), np.min(target_coords[1]), np.max(target_coords[1])))
        print("bottom elevations: min - max = {} - {}".format(np.nanmin(bottom_coords), np.nanmax(bottom_coords)))
    del mgrid0
    # halfway_coords = (0.0 - bottom_coords) / 2.0
    halfway_coords = np.maximum(bottom_coords / 2.0, 0.001)
    return (bottom_coords, halfway_coords)

def interpolate_bathymetry_via_lonlat(lons, lats, bathytopo_path):
    bottom_coords = None
    halfway_coords = None

    target_coords = (np.squeeze(lats), np.squeeze(lons))

    # bathytopo_path = os.path.join("/media","christian", "DATA", "Documents", "LifeStages", "RUG", "Research", "SouthernOcean", "gebco_2026_n-20.0_s-65.0_w-15.0_e45.0.nc")
    btopo_file = xr.open_dataset(bathytopo_path, decode_cf=True, engine='netcdf4')
    fX = btopo_file.variables["lon"]
    fY = btopo_file.variables["lat"]
    fBTOPO = btopo_file.variables["elevation"]
    # mgrid = np.array([fY, fX], dtype=fX.dtype).T
    # if DBG_MSG:
    #     print("Compute triangulation P0 ...")
    # tri0 = Delaunay(mgrid0)
    # if DBG_MSG:
    #     print("Interpolating U0 ...")
    # F0_interp = LinearNDInterpolator(tri0, fF0t)
    # Fs_local_0 = F0_interp(gcenters_arr)
    with np.errstate(invalid='ignore'):
        fBTOPO = np.nan_to_num(fBTOPO, nan=0.0)
    mgrid0 = (fY, fX)
    if DBG_MSG:
        print("mgrid0 dims = ({}, {})".format(mgrid0[0].shape, mgrid0[1].shape))
    bottom_coords = -interpn(mgrid0, fBTOPO.squeeze(), target_coords, method='linear', fill_value=.0)
    if DBG_MSG:
        print("mgrid0: min_0 - max_0: {} - {}; min_1 - max_1: {} - {};".format(np.min(np.array(fY)), np.max(np.array(fY)), np.min(np.array(fX)), np.max(np.array(fX))))
        print("gcenters: min_0 - max_0: {} - {}; min_1 - max_1: {} - {};".format(np.min(target_coords[0]), np.max(target_coords[0]), np.min(target_coords[1]), np.max(target_coords[1])))
        print("bottom elevations: min - max = {} - {}".format(np.nanmin(bottom_coords), np.nanmax(bottom_coords)))
    del mgrid0
    # halfway_coords = (0.0 - bottom_coords) / 2.0
    halfway_coords = np.maximum(bottom_coords / 2.0, 0.001)
    return (bottom_coords, halfway_coords)


if __name__  == "__main__":
    coord_array = np.array(coords_points)
    lons = coord_array[:, 0]
    lats = coord_array[:, 1]
    bottom_coords = None
    halfway_coords = None

    target_coords = (np.squeeze(lats), np.squeeze(lons))

    bathytopo_path = os.path.join("/media","christian", "DATA", "Documents", "LifeStages", "RUG", "Research", "SouthernOcean", "gebco_2026_n-20.0_s-65.0_w-15.0_e45.0.nc")
    btopo_file = xr.open_dataset(bathytopo_path, decode_cf=True, engine='netcdf4')
    fX = btopo_file.variables["lon"]
    fY = btopo_file.variables["lat"]
    fBTOPO = btopo_file.variables["elevation"]
    # mgrid = np.array([fY, fX], dtype=fX.dtype).T
    # if DBG_MSG:
    #     print("Compute triangulation P0 ...")
    # tri0 = Delaunay(mgrid0)
    # if DBG_MSG:
    #     print("Interpolating U0 ...")
    # F0_interp = LinearNDInterpolator(tri0, fF0t)
    # Fs_local_0 = F0_interp(gcenters_arr)
    with np.errstate(invalid='ignore'):
        fBTOPO = np.nan_to_num(fBTOPO, nan=0.0)
    mgrid0 = (fY, fX)
    if DBG_MSG:
        print("mgrid0 dims = ({}, {})".format(mgrid0[0].shape, mgrid0[1].shape))
    bottom_coords = -interpn(mgrid0, fBTOPO.squeeze(), target_coords, method='linear', fill_value=.0)
    if DBG_MSG:
        print("mgrid0: min_0 - max_0: {} - {}; min_1 - max_1: {} - {};".format(np.min(np.array(fY)), np.max(np.array(fY)), np.min(np.array(fX)), np.max(np.array(fX))))
        print("gcenters: min_0 - max_0: {} - {}; min_1 - max_1: {} - {};".format(np.min(target_coords[0]), np.max(target_coords[0]), np.min(target_coords[1]), np.max(target_coords[1])))
        print("bottom elevations: min - max = {} - {}".format(np.nanmin(bottom_coords), np.nanmax(bottom_coords)))
    del mgrid0
    # halfway_coords = (0.0 - bottom_coords) / 2.0
    halfway_coords = bottom_coords / 2.0
    print(halfway_coords)


