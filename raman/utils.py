# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Hashable, Iterable, Mapping, Optional

import xarray as xr

from . import alignment

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
            alignment.normalize_spatial_dimensions(arr, origin=spatial_origin, dims=spatial_dims)
            for arr in arrays
        )

        arrays = alignment.align_spatial_dimensions(
            arrays,
            dims=spatial_dims,
            tolerance=spatial_align_tolerance,
            method=spatial_align_method
        )

    if align_frequencies:
        arrays = alignment.align_frequency_dimension(
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
