# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from .io import read_wdf, read_matlab
from . import utils
from . import preprocessing
from . import baseline
from . import alignment

def _register_xarray_helper():
    # pylint: disable=import-outside-toplevel
    import xarray as xr
    from .xarray_helper import XArrayHelper

    xr.register_dataarray_accessor('rm')(XArrayHelper)

_register_xarray_helper()
