# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

import collections
from itertools import product

import cvxpy as cp
import numpy as np

def sunsal_tv(A, Y, lambda_1, lambda_tv, sweep='prod', tv_type='iso', additional_constraint='none'):
    r"""
    Sparse unmixing via variable splitting and augmented Lagrangian and total variation (SUnSAL-TV)

    solves the following optimization problem

    min      || Y - A * X ||_F + lambda_1 || X ||_1 + lambda_TV || X ||_TV
     X
    subject to  X >= 0                 # if additional_constraint is 'positive'
                sum(X, axis=0) == 1    # if additional_constraint is 'sum_to_one'

    with
      || X ||_1 = \sum_i | x_i |    # for a flattened array X
      || X ||_TV = \sum_i (\sum_j |X_ij|^p)^(1/p)  # p = 1 for non-isotropic and p = 2 for isotropic

    Parameters
    ----------
    A: array - N x L, spectral library, where L is the number of library elements and N the number of points in each spectrum
    Y: array - N x m_1 x ... x m_d, target spectra, m_1, ..., m_d are spatial dimnesions
    lambda_1: float - regularization constant for elementwise sparsity inducing term
    lambda_TV: float - regularization constant for TV regularizer (sparse changes along spatial dimensions)
    sweep: {'prod', 'zip'} -
    tv_type: {'iso', 'non-iso'} - type of total variation norm, isotropic or non-isotropic
    additional_constraint: {'none', 'positive', 'sum_to_one'} - additional constraint on solution

    Returns
    -------
    X: array - L x m_1 x ... x m_d

    References
    ----------
    [1] M. Iordache, J. M. Bioucas-Dias and A. Plaza, "Total Variation Spatial Regularization for
        Sparse Hyperspectral Unmixing," in IEEE Transactions on Geoscience and Remote Sensing,
        vol. 50, no. 11, pp. 4484-4502, Nov. 2012.

    [2] Matlab implementation, downloaded from
        https://github.com/ricardoborsoi/MUA_SparseUnmixing/blob/57802d5b2f77649fb32c2e4c75258f8d91084f7d/sunsal_tv.m

    [3] https://dsp.stackexchange.com/questions/57977/isotropic-and-anisotropic-in-the-total-variation-framework
    """
    # get dimensions
    num_spectra, lib_size = A.shape
    sample_dims = Y.shape[1:]
    assert Y.shape[0] == num_spectra, 'Size of library does not size of target variables'

    # reshape Y from [spectra x Xpos x Ypos x ...] --> [spectra x (Xpos * Ypos * ...)]
    Y = Y.reshape((num_spectra, -1))
    num_samples = Y.shape[1]

    # create optimization variables
    positive_solution = (additional_constraint == 'positive')
    X = cp.Variable((lib_size, num_samples), nonneg=positive_solution)
    p_lambda_1 = cp.Parameter(1, nonneg=True)
    p_lambda_tv = cp.Parameter(1, nonneg=True)

    # calculate first differences in each direction
    idx = np.r_[:num_samples]
    idx_s = idx.reshape(sample_dims)

    differences = []
    for n, d in enumerate(sample_dims):
        ia = np.ravel(idx_s.take(indices=np.r_[np.r_[1:d], 0], axis=n))
        ib = np.ravel(idx_s.take(indices=np.r_[:d], axis=n))

        differences.append(X[:, ia] - X[:, ib])

    # compute TV norm
    if tv_type == 'iso':
        D = [x*x for x in differences]
        D = cp.sqrt(cp.sum(D))
        tv = cp.sum(D)
    elif tv_type == 'non-iso':
        D = [cp.sum(cp.abs(x)) for x in differences]
        tv = cp.sum(D)
    else:
        raise ValueError(f'TV norm type `{tv_type}` is not defined')

    # define object function
    obj = cp.norm(Y - A @ X, p='fro') + p_lambda_1 * cp.pnorm(X, p=1) + p_lambda_tv * tv

    # constraints
    constr = []
    if additional_constraint == 'sum_to_one':
        constr.append(cp.sum(X, axis=0) == 1)

    # opimiztion problem
    prob = cp.Problem(cp.Minimize(obj), constr)

    # init parameter sweep
    # if lambda_1 and lambda_tv are scalar return result
    # otherwise return a dict with (lambda_1, lambda_tv): result
    lambda_scalar = True
    if not isinstance(lambda_1, collections.Iterable):
        lambda_1 = [lambda_1]
    else:
        lambda_scalar = False

    if not isinstance(lambda_tv, collections.Iterable):
        lambda_tv = [lambda_tv]
    else:
        lambda_scalar = False

    if sweep == 'prod':
        l_iter = product(lambda_1, lambda_tv)
    elif sweep == 'zip':
        l_iter = zip(lambda_1, lambda_tv)
    else:
        raise ValueError(f'Parameter sweep `{sweep}` not supported')

    results = {}
    for l_1, l_tv in l_iter:
        p_lambda_1.value = l_1
        p_lambda_tv.value = l_tv

        # solution
        prob.solve(solver=cp.SCS, verbose=True)

        results[(l_1, l_tv)] = X.value.reshape((lib_size, ) + sample_dims)

    if lambda_scalar:
        return results.popitem()[1]
    else:
        return results
