# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

import peakutils as pk
import numpy as np
import cvxpy as cp
import xarray as xr

def remove(arr, algorithm, dim='f', **kwargs):
    return xr.apply_ufunc(
        algorithm,
        arr[dim],
        arr,
        kwargs=kwargs,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
    )

def peakutils(x, y, degree=3, axis=-1):
    def solve1d(y):
        return y - pk.baseline(y, deg=degree)

    return np.apply_along_axis(solve1d, axis=axis, arr=y)

def iterative_minimum_polyfit(x, y, degree=3, max_iter=200, impl='qr'):
    """
    Iterative polyfit of elementwise minimum

    Reference
    ----------
    
    Lieber, C. A., & Mahadevan-Jansen, A. (2003). Automated Method for Subtraction of Fluorescence from 
      Biological Raman Spectra. Applied Spectroscopy, 57(11), 1363–1367. 
      https://doi.org/10.1366/000370203322554518

    """

    N = len(x)
    rcond = N * np.finfo(x.dtype).eps

    lhs = np.vander(x, degree+1)
    scale = np.sqrt((lhs*lhs).sum(axis=0))
    lhs /= scale

    rhs = y.reshape((-1, N)).T

    if impl == 'svd':
        W = np.linalg.svd(lhs, full_matrices=False, compute_uv=True)[0]
    elif impl == 'qr':
        W = np.linalg.qr(lhs, mode='reduced')[0]

    def solve_mul(rhs):
        return W @ (W.T @ rhs)

    def solve_lstsq(rhs):
        x = np.linalg.lstsq(lhs, rhs, rcond)[0]

        return lhs @ x

    if impl in ('svd', 'qr'):
        solve = solve_mul
    elif impl == 'lstsq':
        solve = solve_lstsq
    else:
        raise ValueError(f'Implementation `{impl}` not supported')

    for _ in range(max_iter):
        rhs_h = solve(rhs)
        rhs = np.minimum(rhs, rhs_h)

    return y - rhs_h.T.reshape(y.shape)

def iterative_minimum_polyfit_slow(x, y, degree=3, max_iter=200, tol=1e-4, axis=-1):
    """
    Iterative polyfit of elementwise minimum

    Reference
    ----------
    
    Lieber, C. A., & Mahadevan-Jansen, A. (2003). Automated Method for Subtraction of Fluorescence from 
      Biological Raman Spectra. Applied Spectroscopy, 57(11), 1363–1367. 
      https://doi.org/10.1366/000370203322554518

    """

    def solve1d(yi):
        y0 = yi

        for _ in range(max_iter):
            p = np.polynomial.Polynomial.fit(x, yi, degree)
            y_h = p(x)

            y_m = np.minimum(yi, y_h)
            e = yi - y_m
            yi = y_m

            if (e**2).mean() < tol**2:
                break

        return y0 - y_h

    return np.apply_along_axis(solve1d, axis=axis, arr=y)

def lower_polyfit(x, y, degree=3, loss='l1', huber_m=1, axis=-1, verbose=False, solver=None, solver_opts={}):
    N = y.shape[axis]
    x = (x - x.min()) / (x.max() - x.min())

    c = cp.Variable(degree+1, name='c')
    y_p = cp.Parameter(N)

    X = np.vander(x, degree+1)
    scale = (X*X).sum(0)
    X /= scale

    e = y_p - X @ c
    constr = [e >= 0]
    if loss == 'l1':
        obj = cp.sum(e)
    elif loss == 'l2':
        obj = cp.sum_squares(e)
    elif loss == 'huber':
        obj = cp.sum(cp.huber(e, M=huber_m))
    else:
        raise ValueError(f'Loss function `{loss}` is not supported')

    prob = cp.Problem(cp.Minimize(obj), constr)
    
    opts = {'verbose': False}
    opts.update(solver_opts)

    def solve1d(y):
        y_p.value = y
        try:
            prob.solve(solver=solver, **opts)
        except cp.SolverError:
            return np.empty_like(y) * np.nan

        return e.value
    
    result = np.apply_along_axis(solve1d, axis=axis, arr=y)

    st = prob.solver_stats
    if verbose and st is not None:    
        print(
            f'Problem was solved using {st.solver_name} in {st.num_iters} iteration.\n'
            f'Setup time was {st.setup_time}s while {st.solve_time}s on the solution.'
        )

    return result