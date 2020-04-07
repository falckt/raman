# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.colors as colors
import numpy as np

from .utils import stack_xy

def plot_animation(arr, dim='f', fig_kw={}):
    fig, ax = plt.subplots(**fig_kw)

    values = arr[dim]

    vmin, vmax = np.percentile(arr, [2, 98])
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    n = arr.get_axis_num(dim)
    m = list(range(len(arr.dims)))
    m.remove(n)
    
    x_idx = arr.get_index(arr.dims[m[0]])
    y_idx = arr.get_index(arr.dims[m[1]])

    ylim = (x_idx.min(), x_idx.max())
    xlim = (y_idx.min(), y_idx.max())

    def init():
        mesh = arr.sel(**{dim: values[0]}).plot(
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            norm=norm,
        )

        return mesh, 

    def update(val):
        mesh = arr.sel(**{dim: val}).plot.imshow(
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            norm=norm,
            add_colorbar=False, 
            add_labels=True,
        )

        return mesh,

    ani = FuncAnimation(fig, update, frames=values[1:], init_func=init, blit=False)

    plt.close(fig)

    return HTML(ani.to_jshtml())

def plot_central_spectrum(arr, dim='f', **kwargs):
    sel = {k: arr.sizes[k] // 2 for k in arr.dims if k != dim}

    arr.isel(**sel).plot(**kwargs)

def plot_spectrum_facets(arr, dim='f', rows=4, cols=4, **kwargs):
    num = (rows, cols)
    sel = {}
    n = 0
    for k in arr.dims:
        if k == dim:
            continue
        if n >= 2:
            break
        
        sz = arr.sizes[k]
        sel[k] = range(0, sz, int(sz / num[n] + .5))

    k_rows, k_cols = sel.keys()
    arr.isel(**sel).plot(row=k_rows, col=k_cols, **kwargs)

def plot_all_2d(arr, **kwargs):
    stack_xy(arr).plot.imshow(**kwargs)