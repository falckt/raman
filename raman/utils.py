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

    new_array = array.stack(dimensions).reset_index(tuple(dimensions.keys()))  # type: ignore[arg-type]

    return new_array.assign_coords({dim: new_array.get_index(dim) for dim in dimensions})  # type: ignore[union-attr]
