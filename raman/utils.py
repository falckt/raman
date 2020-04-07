# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause


def stack_xy(arr):
    return arr.stack(sample=('x', 'y')).reset_index('sample')
