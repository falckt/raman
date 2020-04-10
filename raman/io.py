# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import BinaryIO, Hashable, Mapping, Optional, Sequence, Union

import collections
import pathlib

import xarray as xr
import numpy as np
from scipy import io as sio

from . import  _renishaw

PathOrFile = Union[str, pathlib.Path, BinaryIO]

def read_matlab(
        paths_or_files: Union[PathOrFile, Sequence[PathOrFile]],
        data_var: str,
        dim_vars: Mapping[str, str],
        long_names: Optional[Mapping[Hashable, str]] = None,
        units: Optional[Mapping[Hashable, str]] = None
        ) -> xr.DataArray:
    if not isinstance(paths_or_files, collections.Iterable) or isinstance(paths_or_files, str):
        paths_or_files = [paths_or_files]
    if not long_names:
        long_names = {}
    if not units:
        units = {}

    data: Mapping[str, np.ndarray] = {}
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

def read_wdf(path: Union[pathlib.Path, str]) -> xr.DataArray:
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
