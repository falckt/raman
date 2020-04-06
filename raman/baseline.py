# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

import peakutils as pk
import numpy as np
import cvxpy as cp

def peakutils(x, degree=3):
    b = pk.baseline(x, deg=degree)
    return x - b

def iterative_minimum_polyfit(x, y, degree=3, max_iter=200, tol=1e-4):
    """
    Iterative polyfit of elementwise minimum

    Reference
    ----------
    
    Lieber, C. A., & Mahadevan-Jansen, A. (2003). Automated Method for Subtraction of Fluorescence from 
      Biological Raman Spectra. Applied Spectroscopy, 57(11), 1363–1367. 
      https://doi.org/10.1366/000370203322554518

    """
    y_out = np.empty_like(y)
    
    # USE: numpy.apply_along_axis

    for idx in np.ndindex(*y.shape[:-1]):
        y0 = y[idx + np.index_exp[:]]

        for _ in range(max_iter):
            p = np.polynomial.Polynomial.fit(x, y, degree)
            y_h = p(x)

            y_m = np.minimum(y, y_h)
            e = y - y_m
            y = y_m

            if (e**2).mean() < tol**2:
                break

        y_out[idx + np.index_exp[:]] = y0 - y_h

    return y_out

def lower_polyfit(x, y, degree):
    # y_out = np.empty_like(y)

    N = y.shape[-1]
    x = (x - x.min()) / (x.max() - x.min())

    c = cp.Variable(degree+1, name='c')
    e = cp.Variable(N, nonneg=True, name='e')
    y_p = cp.Parameter(N)

    X = np.vander(x, degree+1)
    scale = (X*X).sum(0)
    X /= scale

    constr = [y_p == X @ c + e]
    obj = cp.sum_squares(e)

    prob = cp.Problem(cp.Minimize(obj), constr)
    
    def solve1d(y):
        y_p.value = y
        prob.solve(verbose=False)

        return e.value

    # for idx in np.ndindex(*y.shape[:-1]):
    #     y_p.value = y[idx + np.index_exp[:]]
    #     prob.solve(verbose=False)

    #     y_out[idx + np.index_exp[:]] = e.value
    
    return np.apply_along_axis(solve1d, -1, y)

def iterative_reweigthing_polyfit():
    """
    Reference
    ---------

    H. Ruan and L.K. Dai, "Automated Background Subtraction Algorithm for Raman Spectra Based on 
      Iterative Weighted Least Squares," in Asian Journal of Chemistry vol. 23, no. 12, 
      pp. 5229-5234, 2011.
      http://www.asianjournalofchemistry.co.in/user/journal/viewarticle.aspx?ArticleID=23_12_11

    """
    pass