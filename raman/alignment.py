# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Hashable, Iterable, List, Mapping, Optional, Union

import xarray as xr

def normalize_spatial_dimensions(
        arr: xr.DataArray, origin: str = 'center', dims: Iterable[Hashable] = ('x', 'y')
        ) -> xr.DataArray:
    """Normalize spatial coordinates

    For a dataarray with arbitrary spatial positioning, introduce new spatial indexing that
    puts (0, 0) either in the center of the scan or in the bottom left corner. This is a
    necessary preprocessing if multiple scans of the same size should be aligned.

    Args:
        arr: input dataarray
        origin: {'center' (default), 'min'}
            - 'center': new spatial coordinates will have (0, 0) at the center of the image
            - 'min': new spatial coordinates will have (0, 0) at the bottom left corner
                of the image
        dims: dimensions that should be normalized, defaults to ('x', 'y')

    Returns:
        dataarray with normalized spatial coordinates, the orginal coordinates are retained
        as 'orginal_name'_old.

    Examples:
        a : <xarray.DataArray (pixel: 4)>
            array([0., 0., 0., 0.])
            Coordinates:
                x        (pixel) int64 6 6 9 9
                y        (pixel) int64 1 3 1 3
            Dimensions without coordinates: pixel

        >>> normalize_spatial_dimensions(a, 'center')
        <xarray.DataArray (pixel: 4)>
        array([0., 0., 0., 0.])
        Coordinates:
            x_old    (pixel) int64 6 6 9 9
            y_old    (pixel) int64 1 3 1 3
            y        (pixel) float64 -1.0 1.0 -1.0 1.0
            x        (pixel) float64 -1.5 -1.5 1.5 1.5
        Dimensions without coordinates: pixel

        >>> normalize_spatial_dimensions(a, 'min')
        <xarray.DataArray (pixel: 4)>
        array([0., 0., 0., 0.])
        Coordinates:
            x_old    (pixel) int64 6 6 9 9
            y_old    (pixel) int64 1 3 1 3
            y        (pixel) int64 0 2 0 2
            x        (pixel) int64 0 0 3 3
        Dimensions without coordinates: pixel
    """

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
    """Align frequency dimension of several data arrays

    Align all given data arrays onto a shared frequency axis. The target frequency
    axis can either be provided as an input, or it will be computed as the mean
    of the frequency axes of all inputs. The alignment is carried out by interpolating
    the original data onto the new frequency axis.
    This is useful in different scenarios. Even for measurements with the same settings,
    the actual measured frequencies will deviate slightly between measurements. Without
    alignment combining such meausrements in a single data array, would create lots of 
    gaps for those frequencies that do not match exactly.
    In cases of measurements acquired with different settings this is even more important.
    For this case one should provide the target axis as input, as the assumptions used
    while averaging will likely not be satisfied.

    Args:
        arrays: Iterable (list, tuple, ...) of dataarrays to be aligned
        target_index: target frequency axis for data, if None (default)
            the target will be computed as the average of the frequency
            axis of all input arrays
        dim: array dimension corresponding to the frequency axis, defaults
            to 'f'
        method: {'liner' (default), 'quadratic', ...} interpolation method
            refer to xarray.DataArray.interp for all possible values
        interp_kwargs: keyword arguments passed onto
            xarray.DataArray.interp

    Returns:
        list of dataarrays with identical frequency axes

    See also:
        xarray.DataArray.interp: underlying interpolation function
        align_spatial_dimensions: similar functionality for spatial alignment

    Examples:
        Two data arrays with slightly different frequencies

        >>> a = xr.DataArray([1, 2, 3, 4], dims=['f'], coords={'f': [600, 610, 620, 630]})
        >>> b = xr.DataArray([3, 4, 5, 6], dims=['f'], coords={'f': [599, 611, 620.2, 629.7]})

        Direct concatenation results in gaps

        >>> xr.concat([a, b], dim='samples')
        <xarray.DataArray (samples: 2, f: 8)>
        array([[nan,  1.,  2., nan,  3., nan, nan,  4.],
               [ 3., nan, nan,  4., nan,  5.,  6., nan]])
        Coordinates:
          * f        (f) float64 599.0 600.0 610.0 611.0 620.0 620.2 629.7 630.0
        Dimensions without coordinates: samples

        Frequency alignment avoids the gaps

        >>> ab = align_frequency_dimension([a, b])
        >>> xr.concat(ab, dim='samples')
        xarray.DataArray (samples: 2, f: 4)>
        array([[0.95      , 2.05      , 3.01      , 3.985     ],
               [3.04166667, 3.95833333, 4.98913043, 6.01578947]])
        Coordinates:
          * f        (f) float64 599.5 610.5 620.1 629.9
        Dimensions without coordinates: samples

        Alignment onto a specified frequency axis

        >>> ab = align_frequency_dimension([a, b], np.linspace(600, 620, 5))
        >>> xr.concat(ab, dim='samples')
        <xarray.DataArray (sample: 2, f: 5)>
        array([[1.        , 1.5       , 2.        , 2.5       , 3.        ],
               [3.08333333, 3.5       , 3.91666667, 4.43478261, 4.97826087]])
        Coordinates:
          * f        (f) float64 600.0 605.0 610.0 615.0 620.0
        Dimensions without coordinates: sample
    """

    if target_index is None:
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
        indexes: Union[Iterable[Hashable], Mapping[Hashable, Any]] = ('x', 'y'),
        tolerance: Optional[float] = 1,
        method: Optional[str] = 'nearest'
        ) -> List[xr.DataArray]:
    """Align spatial dimensions of several data arrays

    Align all given data arrays onto a shared spatial grid. Grid values can either be
    given as an input, or the spatial grid of the the first input array will be used
    as a target. This just casts the data onto a common grid, the data is not 
    interpolated.
    Even for measurements with the same settings, the exact spatial grid spacing will 
    deviate slightly between measurements. Without alignment combining such meausrements 
    in a single data array, will create lots of gaps for those grid cells that do not 
    match exactly.

    Args:
        arrays: Iterable (list, tuple, ...) of dataarrays to be aligned
        indexes: either list of dimension names (defaults to ('x', 'y'))
            in which case the index of these dimensions of the array will
            be used as target grid.
            Otherwise can be a dict-like structure with dimension name as
            key and the desired index values as value.
        tolerance: maximum distance from a grid point to look for candidate
            values, refer to xarray.DataArray.reindex for detailed information
        method: {'nearest' (default), ...} reindexing method
            refer to xarray.DataArray.reindex for all possible values

    Returns:
        list of dataarrays with identical spatial dimensions

    Note:
        Most likely you will run `normalize_spatial_dimensions` first to
        establish a consistent coordinate origin accross all measurements.
        Consider using `raman.utils.concat` which does this automatically.

    See also:
        normalize_spatial_dimensions: normalize spatial dimensions
        align_frequency_dimension: similar functionality for frequncy alignment
        raman.utils.contact: function that combines spatial normalization,
            spatial and frequency alignment, and concatenation
        xarray.DataArray.reindex: underlying function

    Examples:
        Two data arrays with slightly different frequencies

        >>> a = xr.DataArray([[1, 2], [3, 4]], dims=['x', 'y'], coords={'x': [0, 5], 'y': [0, 5]})
        >>> b = xr.DataArray([[3, 4], [5, 6]], dims=['x', 'y'], coords={'x': [0, 5.5], 'y': [0, 7]})

        Direct concatenation results in a diffrent shaped array with gaps

        >>> xr.concat([a, b], dim='samples')
        <xarray.DataArray (samples: 2, x: 3, y: 3)>
        array([[[ 1.,  2., nan],
                [ 3.,  4., nan],
                [nan, nan, nan]],

               [[ 3., nan,  4.],
                [nan, nan, nan],
                [ 5., nan,  6.]]])
        Coordinates:
          * x        (x) float64 0.0 5.0 5.5
          * y        (y) int64 0 5 7
        Dimensions without coordinates: samples

        Spatial alignment retains shape and avoids gaps where possible

        >>> ab = align_spatial_dimensions([a, b])
        >>> xr.concat(ab, dim='samples')
        <xarray.DataArray (samples: 2, x: 2, y: 2)>
        array([[[ 1.,  2.],
                [ 3.,  4.]],

               [[ 3., nan],
                [ 5., nan]]])
        Coordinates:
          * x        (x) int64 0 5
          * y        (y) int64 0 5
        Dimensions without coordinates: samples
    """

    arr_iter = iter(arrays)

    if not isinstance(indexes, Mapping):
        prototype = next(arr_iter)

        indexes = {dim: prototype.get_index(dim) for dim in indexes}

    return [prototype] + [
        arr.reindex(indexes, tolerance=tolerance, method=method)
        for arr in arr_iter
    ]
