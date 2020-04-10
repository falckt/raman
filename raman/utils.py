# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, cast

from collections import defaultdict
import xarray as xr

import raman

def cubify(
        arr: xr.DataArray,
        *spatial_dims: str,
        pixel_dim: Hashable = 'pixel'
        ):
    if not spatial_dims:
        spatial_dims = ('x', 'y')
    cube = cast(xr.DataArray, arr.set_index({pixel_dim: spatial_dims})).unstack(pixel_dim)
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

    arr_list = list(arrays)
    con_arr = xr.concat(arr_list, dim=concat_dim)

    con_coords = {}
    for attr in arr_list[0].attrs:
        values = list(arr.attrs[attr] for arr in arr_list)

        if len(set(values)) == 1:
            # value is constant --> not a coordinate
            continue

        con_coords[attr] = xr.DataArray(values, dims=concat_dim)

    con_arr = con_arr.assign_coords(con_coords)
    for key in con_coords:
        del con_arr.attrs[key]

    return con_arr

def ensure_dims(array: xr.DataArray, *dimensions: Hashable) -> xr.DataArray:
    missing_dims = set(dimensions) - set(array.dims)

    new_dims = defaultdict(list)
    for coord in missing_dims:
        cdim_tuple = array.coords[coord].dims

        if len(cdim_tuple) > 1:
            raise ValueError('Multi dimensional coordinates are not supported')

        cdim = cdim_tuple[0]

        new_dims[cdim].append(coord)

    for dim, coords in new_dims.items():
        array = cast(xr.DataArray, array.set_index({cdim: tuple(coords)}))

        if len(coords) > 1:
            array = array.unstack(dim)

    return array.drop_vars(array.coords.keys() - set(array.dims))

def stack_dims(
        array: xr.DataArray,
        **dimensions: Sequence[Hashable]
        ) -> xr.DataArray:

    # to satisfy static type check with mypy
    dim = cast(Mapping[Hashable, Sequence[Hashable]], dimensions)

    new_array = array.stack(dim).reset_index(tuple(dimensions.keys()))

    # mypy: reset_index can happen inplace in which case it returns none
    new_array = cast(xr.DataArray, new_array)

    return new_array.assign_coords({dim: new_array.get_index(dim) for dim in dimensions})
