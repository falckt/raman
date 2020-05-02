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
    """Mask saturated pixels

    Args:
        arr: input data
        saturation_value: pixel value that indicates a saturated pixel, defaults to 0

    Returns:
        dataarray with saturated pixels replaced by NaNs
    """

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
    """Interpolate masked pixels with neighborhood information

    Impute NaN values in data array using values from neighboring elements along
    selectable dimensions. Different interpolation methods are supported depending
    on the number of dimensions used for interpolation. If no neighboring information
    is available, the maximum valid value within that spectrum is used.

    Args:
        arr: input data
        method: {'linear', 'nearest', ... }, delaults to 'linear'
            Valid choices depends on `interpolation_dims`. If len(interpolation_dims) = 1,
            then any method supported by `scipy.interpolate.interp1d` can be used. Otherwise
            just 'linear' and 'nearest' are supported.
        interpolation_dims: defaults to all dimensions but 'dim'
            the array dimensions which are used to fill in missing values
        dim: defaults to 'f'
            used to infer `interpolation_dims` if they are not explicitly specified

    Returns:
        dataarray with NaN values imputed with values from neighboring pixels.

    See also:
        scipy.interpolate.interp1d: function used for 1-d interpolation
        scipy.interpolate.NearestNDInterpolator: function used for nearest neighbor interpolation
            in n-dimensions
        scipy.interpolate.LinearNDInterpolator: function used for linear interpolation in
            n-dimensions
    """

    D0 = len(arr.dims)

    if interpolation_dims is None:
        interpolation_dims = [d for d in arr.dims if d != dim]

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
    """Delete pixels (spectra) that contain too many invalid values

    Counts the number of invalid values in each spectrum. If the number of invalid
    values exceeds a threshold, the corresponding pixel is dropped from the dataarray.

    Args:
        arr: input datarray with shape (n_frequencies, n_pixels)
        thres: number of invalid values to tolerate in a single pixel (spectrum)
        drop_old_index: if deleting pixels, also delete the original pixel index
            if the input array has pixel indices [0, *1, 2, *3, *4, 5] where '*'
            indicates invalid values in that spectrum, the returned array will
            have a new pixel index [0, 1, 2]. If `drop_old_index` is False, the
            returned array will have a coordinate `dim_old` = [0, 2, 5] which
            contains the original index values of the retained pixels.
        dim: array dimension that contains spectra, defaults to 'f'

    Returns:
        dataarray with shape (n_frequencies, n_valid_pixels) with n_valid_pixels <= n_pixels
    """

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
    """Normalize spectra

    Normalized every spectrum contained in the dataarray.

    Args:
        arr: input array
        dim: array dimension that contains spectra, defaults to 'f'
        method: {'root_mean_square', 'snv', 'unit_variance'}

    Returns:
        array of same shape as input array but with normalized spectra
    """

    if method == 'root_mean_square':
        ss = np.sqrt((arr*arr).mean(dim=dim))
        res = arr / ss

    elif method == 'snv':
        std = arr.std(dim=dim)
        mean = arr.mean(dim=dim)
        res = (arr - mean) / std

    elif method == 'unit_variance':
        std = arr.std(dim=dim)
        res = arr / std

    return res.assign_attrs(arr.attrs)
