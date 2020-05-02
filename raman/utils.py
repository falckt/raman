# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence

from collections import defaultdict
import xarray as xr

import raman

def cubify(
        arr: xr.DataArray,
        *spatial_dims: str,
        pixel_dim: Hashable = 'pixel'
        ):
    """Convert dataarray into a datacube

    Convert a dataarray with a 'pixel' dimension into a cube with spatial dimensions.

    Args:
        arr: input datarray with dimension `pixel_dim` and linked coordinates given
            by `spatial_dims`
        *spatial_dims: coordinate variables corresponding to spatial dimensions,
            default is 'x', 'y'
        pixel_dim: name of the existing dimension describing the pixels, defaults to 'pixel'

    Returns:
        reshaped dataarray into a datacube
    """
    if not spatial_dims:
        spatial_dims = ('x', 'y')
    cube = arr.set_index({pixel_dim: spatial_dims}).unstack(pixel_dim)  # type: ignore[union-attr]
    for d in spatial_dims:
        cube.coords[d].attrs = arr.coords[d].attrs
    return cube

def concat(
        arrays: Iterable[xr.DataArray],
        align_frequencies: bool = True,
        align_spatially: bool = False,
        *,  # make the following arguments keyword only
        concat_dim: Hashable = 'file_idx',
        frequency_dim: Hashable = 'f',
        spatial_dims: Iterable[Hashable] = ('x', 'y'),
        frequency_interp_method: str = 'linear',
        frequency_interp_kwargs: Mapping[str, Any] = {},
        spatial_origin: str = 'center',
        spatial_align_tolerance: Optional[float] = 1,
        spatial_align_method: Optional[str] = 'nearest'
        ) -> xr.DataArray:
    if align_spatially:
        arrays = (
            raman.alignment.normalize_spatial_dimensions(
                arr, origin=spatial_origin, dims=spatial_dims)
            for arr in arrays
        )

        arrays = raman.alignment.align_spatial_dimensions(
            arrays,
            indexes=spatial_dims,
            tolerance=spatial_align_tolerance,
            method=spatial_align_method
        )

    if align_frequencies:
        arrays = raman.alignment.align_frequency_dimension(
            arrays,
            dim=frequency_dim,
            method=frequency_interp_method,
            interp_kwargs=frequency_interp_kwargs
        )

    return xr.concat(list(arrays), dim=concat_dim)

def ensure_dims(array: xr.DataArray, *dimensions: Hashable) -> xr.DataArray:
    """Ensure that the given variables are dimensions

    Ensure that the requested variables are dimensions of the returned array. This
    is useful for follow up processing and plotting.

    Args:
        array: input dataarray
        *dimensions: variable names that should be dimensions in the returned array

    Returns:
        dataarray in which all variables given as arguments are dimensions. The shape
        of the output array will likely be different.

    See also:
        stack_dims: in some contexts the inverse operation

    Example:
        a : <xarray.DataArray (f: 3, pixel: 4)>
            array([[0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.]])
            Coordinates:
              * f        (f) int64 600 620 640
                x        (pixel) int64 0 0 2 2
                y        (pixel) int64 1 3 1 3
            Dimensions without coordinates: pixel

        >>> ensure_dims(a, ['f', 'x', 'y'])
        <xarray.DataArray (f: 3, y: 2, x: 2)>
        array([[[0., 0.],
                [0., 0.]],

               [[0., 0.],
                [0., 0.]],

               [[0., 0.],
                [0., 0.]]])
        Coordinates:
          * f        (f) int64 600 620 640
          * y        (y) int64 1 3
          * x        (x) int64 0 2
    """
    missing_dims = set(dimensions) - set(array.dims)

    new_dims = defaultdict(list)
    for coord in missing_dims:
        cdim_tuple = array.coords[coord].dims

        if len(cdim_tuple) > 1:
            raise ValueError('Multi dimensional coordinates are not supported')

        cdim = cdim_tuple[0]

        new_dims[cdim].append(coord)

    for dim, coords in new_dims.items():
        array = array.set_index({cdim: tuple(coords)})  # type: ignore[assignment]

        if len(coords) > 1:
            array = array.unstack(dim)

    return array.drop_vars(array.coords.keys() - set(array.dims))

def stack_dims(
        array: xr.DataArray,
        **dimensions: Sequence[Hashable]
        ) -> xr.DataArray:
    """Stack values along several dimensions into a new dimension

    Args:
        array: input array
        **dimensions: new_dimension_name=(old_dim_1, old_dim_2, ...)

    Returns:
        array with selected dimensions flattened into a new dimension.
        The original index values are retained in linked coordinate
        variables.

    Example:
        a : <xarray.DataArray (f: 3, y: 2, x: 2)>
            array([[[0., 0.],
                    [0., 0.]],

                   [[0., 0.],
                    [0., 0.]],

                   [[0., 0.],
                    [0., 0.]]])
            Coordinates:
              * f        (f) int64 600 620 640
              * y        (y) int64 1 3
              * x        (x) int64 0 2

        >>> stack_dims(a, linear_index=('x', 'y'))
        <xarray.DataArray (f: 3, linear_index: 4)>
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        Coordinates:
          * f             (f) int64 600 620 640
            x             (linear_index) int64 0 0 2 2
            y             (linear_index) int64 1 3 1 3
          * linear_index  (linear_index) int64 0 1 2 3
    """

    new_array = array.stack(dimensions).reset_index(tuple(dimensions.keys()))  # type: ignore[arg-type]

    return new_array.assign_coords({dim: new_array.get_index(dim) for dim in dimensions})  # type: ignore[union-attr]
