# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Hashable, Iterable, List, Mapping, Optional

import xarray as xr

def normalize_spatial_dimensions(
        arr: xr.DataArray, origin: str = 'center', dims: Iterable[Hashable] = ('x', 'y')
        ) -> xr.DataArray:
    orig_dims = set(arr.dims)
    dims = set(dims)

    arr = arr.rename({k: f'{k}_old' for k in dims})

    new_coords = {}
    for dim in dims:
        coord = arr.coords[f'{dim}_old']

        if origin == 'center':
            c0 = (coord.max() + coord.min()) / 2
        elif origin == 'min':
            c0 = coord.min()
        else:
            raise ValueError('Coordinate origin `{origin}` is not supported')

        new_coords[dim] = coord - c0

    arr = arr.assign_coords(new_coords)
    arr = arr.swap_dims({f'{k}_old': k for k in dims & orig_dims})

    return arr

def align_frequency_dimension(
        arrays: Iterable[xr.DataArray],
        target_index: Optional[Any] = None,
        dim: Hashable = 'f',
        method: str = 'linear',
        interp_kwargs: Optional[Mapping[str, Any]] = None
        ) -> List[xr.DataArray]:
    if not target_index:
        arrays_iter = iter(arrays)

        proto = next(arrays_iter)
        proto_coord = proto.coords[dim]

        fs = xr.DataArray(
            [proto_coord] + [arr.coords[dim] for arr in arrays_iter],
            dims=('_array', dim)
        )
        target_index = fs.mean('_array').assign_attrs(proto_coord.attrs)


    kwargs = {'fill_value': 'extrapolate'}
    if interp_kwargs:
        kwargs.update(interp_kwargs)

    return [
        arr.interp({dim: target_index}, method=method, kwargs=kwargs).assign_attrs(arr.attrs)
        for arr in arrays
    ]

def align_spatial_dimensions(
        arrays: Iterable[xr.DataArray],
        dims: Iterable[Hashable] = ('x', 'y'),
        tolerance: Optional[float] = 1,
        method: Optional[str] = 'nearest'
        ) -> List[xr.DataArray]:
    arr_iter = iter(arrays)

    prototype = next(arr_iter)
    indexes = {dim: prototype.get_index(dim) for dim in dims}

    return [prototype] + [
        arr.reindex(indexes, tolerance=tolerance, method=method) for arr in arr_iter
    ]
