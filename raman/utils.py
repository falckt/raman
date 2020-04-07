# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

import xarray as xr
import numpy as np
from scipy import interpolate

def stack_xy(arr):
    return arr.stack(sample=('x', 'y')).reset_index('sample')

def mask_saturated_pixes(arr, dim='f', saturation_count_threshold=10, saturation_value=0):
    sat_pixels = (arr == saturation_value)
    sat_pixels_per_spectrum = sat_pixels.sum(dim=dim)

    idx_spectra_w_saturations = sat_pixels_per_spectrum >= saturation_count_threshold
    idx_sat_pixels = sat_pixels & idx_spectra_w_saturations

    return arr.where(~idx_sat_pixels)

def _interpolate_masked_pixels(y, free_dims, indexes, method):
    D = len(indexes)
    ys = y.shape
    
    if y.ndim == free_dims + 1:
        x = np.array(indexes)
    elif y.ndim == free_dims + D:
        x = np.array([
            c.reshape(-1) for c in np.meshgrid(*indexes)
        ])

        y = y.reshape(ys[:free_dims] + (-1,))
    else:
        raise ValueError('Indexing error')

    invalid = np.isnan(y)
    x = x.T

    for idx in np.ndindex(*ys[:free_dims]):
        yy = y[idx + np.index_exp[:]].ravel()
        ii = invalid[idx + np.index_exp[:]].ravel()

        if ii.sum() == 0:
            continue

        if D == 1:
            x = x.ravel()
            f = interpolate.interp1d(x[~ii], yy[~ii], kind=method, fill_value='extrapolate')
            y[idx + np.index_exp[ii]] = f(x[ii])

        elif method == 'nearest':
            f = interpolate.NearestNDInterpolator(x[~ii, :], yy[~ii])
            y[idx + np.index_exp[ii]] = f(x[ii, :])

        else:
            f = interpolate.LinearNDInterpolator(x[~ii, :], yy[~ii])
            y[idx + np.index_exp[ii]] = f(x[ii, :])

    return y.reshape(ys)

def interpolate_masked_pixels(arr, method='linear', interpolation_dims=None, dim='f'):
    D0 = len(arr.dims)

    if interpolation_dims is None:
        interpolation_dims = list(arr.dims)
        interpolation_dims.remove(dim)
    
    D = len(interpolation_dims)

    indexes = [arr.get_index(d) for d in interpolation_dims]
    
    return xr.apply_ufunc(
        _interpolate_masked_pixels,
        arr,
        D0 - D,
        indexes,
        kwargs={'method': method},
        input_core_dims=[interpolation_dims, [], []],
        output_core_dims=[interpolation_dims]
    )
    
def unsaturate(arr, dim='f', saturation_count_threshold=10, saturation_value=0):
    masked = mask_saturated_pixes(arr, dim, saturation_count_threshold, saturation_value)

    return interpolate_masked_pixels(masked)
