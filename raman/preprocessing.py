# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Hashable, Optional, Sequence

import xarray as xr
import numpy as np
from scipy import interpolate


def mask_saturated_pixels(arr: xr.DataArray, saturation_value: float = 0) -> xr.DataArray:
    return arr.where(arr != saturation_value)

def _interpolate_masked_pixels(
        y: np.ndarray,
        free_dims: int,
        indexes: Sequence[Any],
        method: str
        ) -> xr.DataArray:
    D = len(indexes)
    ys = y.shape

    if y.ndim == free_dims + 1:
        x = np.array(indexes)
    elif y.ndim == free_dims + D:
        x = np.array([
            c.reshape(-1) for c in np.meshgrid(*indexes)
        ])

        y = y.reshape(ys[:free_dims] + (-1,), order='F')
    else:
        raise ValueError('Indexing error')

    invalid = np.isnan(y)
    fill_value = np.nanmax(y)

    x = x.T

    for idx in zip(*np.where(invalid.any(axis=-1))):
        yy = y[idx + np.index_exp[:]].ravel()
        ii = invalid[idx + np.index_exp[:]].ravel()

        if D == 1:
            x = x.ravel()
            f = interpolate.interp1d(x[~ii], yy[~ii], kind=method, fill_value='extrapolate')
            y[idx + np.index_exp[ii]] = f(x[ii])

        elif method == 'nearest':
            f = interpolate.NearestNDInterpolator(x[~ii, :], yy[~ii])
            yh = f(x[ii, :])

            nan_idx = np.isnan(yh)
            if np.any(nan_idx):
                yh[nan_idx] = fill_value

            y[idx + np.index_exp[ii]] = yh

        else:
            f = interpolate.LinearNDInterpolator(x[~ii, :], yy[~ii], fill_value=fill_value)
            y[idx + np.index_exp[ii]] = f(x[ii, :])

    return y.reshape(ys, order='F')

def interpolate_masked_pixels(
        arr: xr.DataArray,
        method: str = 'linear',
        interpolation_dims: Optional[Sequence[Hashable]] = None,
        dim: Hashable = 'f'
        ) -> xr.DataArray:
    D0 = len(arr.dims)

    if interpolation_dims is None:
        interpolation_dims = list(arr.dims)
        interpolation_dims.remove(dim)

    D = len(interpolation_dims)

    indexes = [arr.get_index(d) for d in interpolation_dims]

    return xr.apply_ufunc(
        _interpolate_masked_pixels,
        arr.copy(),
        D0 - D,
        indexes,
        kwargs={'method': method},
        input_core_dims=[interpolation_dims, [], []],
        output_core_dims=[interpolation_dims]
    )

def delete_invalid_pixels(
        arr: xr.DataArray,
        thres: int = 10,
        drop_old_index: bool = False,
        dim: Hashable = 'f'
        ) -> xr.DataArray:
    idx = (arr.isnull().sum(dim) > thres)

    if arr.ndim == 2:
        drop_dim = [d for d in arr.dims if d != dim][0]
        arr = (arr # type: ignore[assignment]
               .isel({drop_dim: ~idx.data})
               .reset_index(drop_dim, drop=drop_old_index)
              )
        arr = arr.assign_coords({drop_dim: arr.get_index(drop_dim)})
    else:
        arr = arr.where(idx)

    return arr

def normalize(
        arr: xr.DataArray,
        dim: Hashable = 'f',
        method: str = 'root_mean_square'
        ) -> xr.DataArray:
    if method == 'root_mean_square':
        ss = np.sqrt((arr*arr).mean(dim=dim))
        res = arr / ss

    elif method == 'snv':
        std = arr.std(dim=dim)
        mean = arr.mean(dim=dim)
        res = (arr - mean) / std

    elif method == 'unit_variance':
        std = arr.std(dim=dim)
        res = mean / std

    return res.assign_attrs(arr.attrs)
