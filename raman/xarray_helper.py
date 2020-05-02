# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

import functools

import xarray as xr

from . import utils
from . import preprocessing
from . import baseline


def _function_to_method(func):
    def wrap_method(method):
        def wrapped_method(self, *args, **kwargs):

            return func(*method(self), *args, **kwargs)

        new_func = functools.update_wrapper(wrapped_method, func)

        orig_doc = func.__doc__.splitlines()[0] if func.__doc__ else 'Not documented'
        new_func.__doc__ = (
            f'{orig_doc}\n\n'
            f'Refer to `{func.__module__}.{func.__name__}` for full documentation.\n\n'
            'See also:\n    '
            f'{func.__module__}.{func.__name__} : equivalent function'
        )
        new_func.__name__ = method.__name__

        return new_func
    return wrap_method

class XArrayHelper:
    """Helper class to turn module functions into xarray methods"""

    def __init__(self, array: xr.DataArray):
        self._array = array

    @_function_to_method(utils.ensure_dims)
    def ensure_dims(self):
        return (self._array, )

    @_function_to_method(utils.stack_dims)
    def stack_dims(self):
        return (self._array, )

    @_function_to_method(utils.cubify)
    def cubify(self):
        return (self._array, )

    @_function_to_method(baseline.remove)
    def remove_baseline(self):
        return (self._array, )

    @_function_to_method(preprocessing.normalize)
    def normalize(self):
        return (self._array, )

    @_function_to_method(preprocessing.mask_saturated_pixels)
    def mask_saturated_pixels(self):
        return (self._array, )

    @_function_to_method(preprocessing.interpolate_masked_pixels)
    def interpolate_masked_pixels(self):
        return (self._array, )

    @_function_to_method(preprocessing.delete_invalid_pixels)
    def delete_invalid_pixels(self):
        return (self._array, )
