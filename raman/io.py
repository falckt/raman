# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import xarray as xr
from scipy import io as sio
import collections

from . import  _renishaw

def read_matlab(paths_or_files, data_var, dim_vars, long_names={}, units={}):
    
    if not isinstance(paths_or_files, collections.Iterable) or isinstance(paths_or_files, str):
        paths_or_files = [paths_or_files]
    
    data = {}
    for p in paths_or_files:
        sio.loadmat(p, data)

    da = xr.DataArray(
        data[data_var],
        dims=dim_vars.keys(),
        coords={k: data[v] for k, v in dim_vars}
    )

    for d in da.dims:
        if d in long_names:
            da[d].attrs['long_name'] = long_names[d]

        if d in units:
            da[d].attrs['units'] = units[d]

    if '__data__' in units:
        da.attrs['units'] = units['__data__']

    return da

def read_wdf(path):
    rawdata = _renishaw.parse_wdf(path)
    
    spectra = rawdata['DATA']
    f = rawdata['XLST']['data']
    x = rawdata['ORGN']['X']['data']
    y = rawdata['ORGN']['Y']['data']
    # W = rawdata['WMAP']['width']
    # H = rawdata['WMAP']['height']

    da = xr.DataArray(
        spectra,
        dims=('pixel', 'f'),
        coords={
            'f': f,
            'x': ('pixel', x),
            'y': ('pixel', y),
        }
    )
    da = da.assign_coords(pixel=da.get_index('pixel'))

    da.f.attrs['long_name'] = rawdata['XLST']['type']
    da.f.attrs['units'] = rawdata['XLST']['units']

    da.x.attrs['long_name'] = rawdata['ORGN']['X']['name']
    da.x.attrs['units'] = rawdata['ORGN']['X']['units']

    da.y.attrs['long_name'] = rawdata['ORGN']['Y']['name']
    da.y.attrs['units'] = rawdata['ORGN']['Y']['units']

    da.attrs['long_name'] = 'Value'
    da.attrs['units'] = rawdata['WDF1']['spectral_units']

    da = da.sortby(da.f)

    return da