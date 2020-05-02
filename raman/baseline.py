# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Union

import numpy as np
import xarray as xr

def remove(
        arr: xr.DataArray,
        algorithm: Union[str, Callable] = 'iterative_minimum_polyfit',
        dim: Hashable = 'f',
        **kwargs
        ) -> xr.DataArray:

    if isinstance(algorithm, str):
        func = removal_methods[algorithm]
    else:
        func = algorithm

    return xr.apply_ufunc(
        func,
        arr[dim],
        arr,
        kwargs=kwargs,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
    ).assign_attrs(arr.attrs)

def peakutils(
        x: np.ndarray,   # pylint: disable=unused-argument
        y: np.ndarray,
        degree: int = 3,
        axis: int = -1
        ) -> np.ndarray:

    import peakutils as pk  # pylint: disable=import-outside-toplevel

    def solve1d(y: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(y)
        y[valid] -= pk.baseline(y[valid], deg=degree)
        return y

    return np.apply_along_axis(solve1d, axis=axis, arr=y)

def _iterative_minimum_fit(
        lhs: np.ndarray,
        rhs: np.ndarray,
        max_iter: int,
        impl: str
        ) -> np.ndarray:
    if impl == 'svd':
        W = np.linalg.svd(lhs, full_matrices=False, compute_uv=True)[0]
    elif impl == 'qr':
        W = np.linalg.qr(lhs, mode='reduced')[0]
    elif impl == 'lstsq':
        N = len(lhs)
        rcond = N * np.finfo(lhs.dtype).eps
    else:
        raise ValueError(f'Implementation `{impl}` not supported')

    def solve_mul(rhs):
        return W @ (W.T @ rhs)

    def solve_lstsq(rhs):
        x = np.linalg.lstsq(lhs, rhs, rcond)[0]

        return lhs @ x

    if impl in ('svd', 'qr'):
        solve = solve_mul
    elif impl == 'lstsq':
        solve = solve_lstsq

    for _ in range(max_iter):
        rhs_h = solve(rhs)
        rhs = np.minimum(rhs, rhs_h)

    return rhs_h

def iterative_minimum_polyfit(
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 3,
        max_iter: int = 200,
        impl: str = 'qr'
        ):
    """
    Iterative polyfit of elementwise minimum

    Reference
    ----------

    Lieber, C. A., & Mahadevan-Jansen, A. (2003). Automated Method for Subtraction of Fluorescence
      from Biological Raman Spectra. Applied Spectroscopy, 57(11), 1363–1367.
      https://doi.org/10.1366/000370203322554518

    """

    N = len(x)

    lhs = np.vander(x, degree+1)
    scale = np.sqrt((lhs*lhs).sum(axis=0))
    lhs /= scale

    rhs = y.reshape((-1, N)).T.copy()

    valid = ~np.isnan(rhs)
    valid_cols = valid.all(axis=0)

    rhs[:, valid_cols] = _iterative_minimum_fit(lhs, rhs[:, valid_cols], max_iter, impl)

    for n in np.where(~valid_cols)[0]:
        valid_rows = valid[:, n]

        rhs[valid_rows, n] = _iterative_minimum_fit(
            lhs[valid_rows, :], rhs[valid_rows, n], max_iter, impl
        )

    return y - rhs.T.reshape(y.shape)

def iterative_minimum_polyfit_slow(
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 3,
        max_iter: int = 200,
        tol: float = 1e-4,
        axis: int = -1
        ):
    """
    Iterative polyfit of elementwise minimum

    Reference
    ----------

    Lieber, C. A., & Mahadevan-Jansen, A. (2003). Automated Method for Subtraction of Fluorescence
      from Biological Raman Spectra. Applied Spectroscopy, 57(11), 1363–1367.
      https://doi.org/10.1366/000370203322554518

    """

    def solve1d(yi):
        y0 = yi

        valid = ~np.isnan(yi)
        yi = yi[valid]
        xi = x[valid]

        for _ in range(max_iter):
            p = np.polynomial.Polynomial.fit(xi, yi, degree)
            y_h = p(xi)

            y_m = np.minimum(yi, y_h)
            e = yi - y_m
            yi = y_m

            if (e**2).mean() < tol**2:
                break

        y0[valid] -= y_h

        return y0

    return np.apply_along_axis(solve1d, axis=axis, arr=y)

def lower_polyfit(
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 3,
        loss: str = 'l1',
        huber_m: float = 1,
        axis: int = -1,
        verbose: bool = False,
        solver: Optional[str] = None,
        solver_opts: Optional[Mapping[str, Any]] = None
        ):
    import cvxpy as cp  # pylint: disable=import-outside-toplevel

    N = y.shape[axis]
    x = (x - x.min()) / (x.max() - x.min())

    c = cp.Variable(degree+1, name='c')
    y_p = cp.Parameter(N)

    X = np.vander(x, degree+1)
    scale = (X*X).sum(0)
    X /= scale

    e = y_p - X @ c
    constr = [e >= 0]

    def get_loss(e):
        if loss == 'l1':
            return cp.sum(e)
        elif loss == 'l2':
            return cp.sum_squares(e)
        elif loss == 'huber':
            return cp.sum(cp.huber(e, M=huber_m))
        else:
            raise ValueError(f'Loss function `{loss}` is not supported')

    prob = cp.Problem(cp.Minimize(get_loss(e)), constr)

    opts = {'verbose': False}
    if solver_opts:
        opts.update(solver_opts)

    def solve1d(y):
        valid = ~np.isnan(y)

        has_nans = not np.all(valid)

        if has_nans:
            y_p.value[valid] = y[valid]
            P = cp.Problem(
                cp.Minimize(get_loss(e[valid])),
                [e[valid] >= 0]
            )
        else:
            y_p.value = y
            P = prob

        try:
            P.solve(solver=solver, **opts)
        except cp.SolverError:
            return np.empty_like(y) * np.nan

        if has_nans:
            return np.where(valid, e.value, np.nan)
        else:
            return e.value

    result = np.apply_along_axis(solve1d, axis=axis, arr=y.copy())

    st = prob.solver_stats
    if verbose and st is not None:
        print(
            f'Problem was solved using {st.solver_name} in {st.num_iters} iteration.\n'
            f'Setup time was {st.setup_time}s while {st.solve_time}s on the solution.'
        )

    return result

removal_methods: Dict[str, Callable] = {
    'iterative_minimum_polyfit': iterative_minimum_polyfit,
    'modpoly': iterative_minimum_polyfit,
    'lower_polyfit': lower_polyfit,
}
