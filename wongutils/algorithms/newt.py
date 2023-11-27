__copyright__ = """Copyright (C) 2023 George N. Wong"""
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
from scipy.linalg import lu_solve, lu_factor
import warnings


def print_verbose(verbose, *args):
    """
    Wrapper for print(*args).

    :arg verbose: Whether or not to print.
    """
    if verbose > 0:
        print(*args)


def fmin(x, func, *args):
    """
    Return scalar akin to magnitude of func(x, *args).

    :arg func: The function to evaluate.

    :arg x: The point at which to compute the value of the function.

    :returns: sum(f(x[i])^2)/2
    """
    return np.sum(np.power(func(x, *args), 2.)) / 2.


def fdjac(x, func, *args, epsilon=1e-7):
    """
    Calculate the Jacobian matrix of a vector-valued function using finite differences.

    :arg func: The function to be evaluated at x.

    :arg x: The point at which to compute the Jacobian.

    :arg epsilon: (default=1e-7) The step size to use for the finite differences.

    :returns: Jacobian matrix df[i, j] := d(f_i) / d(x_j)
    """

    df = np.zeros((len(x), len(x)))
    fvec = func(x, *args)

    for j in range(len(x)):
        temp = x[j]
        h = epsilon * np.abs(temp)
        if h == 0.0:
            h = epsilon
        x[j] = temp + h
        h = x[j] - temp
        f = func(x, *args)
        x[j] = temp
        df[:, j] = (f - fvec) / h

    return df


def linesearch(x, g, p, step_max, func, *args, max_its=100, alpha=1.e-4,
               lam_min_factor=0.1, verbose=0, tol_x=1.e-9):
    """
    Find lam to minimize f(x + lam*p) following the implementation in Numerical Recipes.

    :arg x: Starting point of path along which to search.

    :arg g: Gradient of func at x.

    :arg p: Direction of path from point x along which to search.

    :arg step_max: The maximum value of step length along p.

    :arg func: This should be a scalar function of x that we want to minimize.

    :returns: The fraction lam how far to move along p and a check boolean.
    """

    def __check_acceptance(g_trial, g_initial, lam, slope, alpha):
        return g_trial <= g_initial + alpha * slope * lam

    # first normalize p to obey step_max
    normalization = 1.
    plen = np.sqrt(np.power(p, 2.).sum())
    if plen > step_max:
        print_verbose(verbose, ' .. in line search, normalizing p by', step_max / plen)
        normalization = step_max / plen
        p *= normalization

    # to check for small step sizes
    lam_min = tol_x / np.max(p / np.maximum(np.abs(x), 1.))

    # these values persist throughout the evaluation
    g0 = func(x, *args)
    slope = np.dot(p, g)

    # first try with the full Newton step
    lam2 = 1.
    g2 = func(x + lam2*p, *args)

    for it in range(max_its):
        if np.isfinite(g2):
            break
        lam2 /= 2.
        g2 = func(x + lam2*p, *args)

    if not np.isfinite(g2):
        raise ValueError('Unable to find lam in linesearch(...) with finite evaluation.')

    message = f' .. in line search full Newton step gives {g2} compared to original {g0}'
    print_verbose(verbose, message)

    if __check_acceptance(g2, g0, lam2, slope, alpha):
        print_verbose(verbose, '    accepted lam', lam2)
        return lam2 * normalization, False

    # now adjust lam and try for quadratic step
    lam1 = - slope / 2. / (g2 - g0 - slope)
    if lam1 < lam_min_factor:
        lam1 = lam_min_factor

    print_verbose(verbose, '   .. now trying for quadratic step with lam', lam1)

    g1 = func(x + lam1*p, *args)
    while not np.isfinite(g1):
        lam1 /= 2.
        g1 = func(x + lam1*p, *args)

    print_verbose(verbose, '      got', g1, 'compared to original', g0)

    if __check_acceptance(g1, g0, lam1, slope, alpha):
        print_verbose(verbose, '        accepted lam', lam1)
        return lam1 * normalization, False

    # now iteratively try to solve with cubic steps
    for it in range(max_its):

        a = (g1-slope*lam1-g0) / lam1/lam1 - (g2-slope*lam2-g0) / lam2/lam2
        b = - lam2*(g1-slope*lam1-g0) / lam1/lam1 + lam1*(g2-slope*lam2-g0) / lam2/lam2

        a /= (lam1 - lam2)
        b /= (lam1 - lam2)

        g2 = g1
        lam2 = lam1

        if a == 0:
            lam1 = -slope / 2/b
        else:
            disc = b*b - 3.*a*slope
            if disc < 0:
                raise Exception('Roundoff problem in lnsrch.')
            else:
                lam1 = (-b + np.sqrt(disc)) / 3./a

        lam1 = max(lam1, lam_min_factor*lam2)

        if lam1 < lam_min:
            print_verbose(verbose, '        tentatively accepting lam with check flag')
            return lam1, True

        print_verbose(verbose, '   .. in cubic search, it', it, 'with lam', lam1)

        g1 = func(x + lam1*p, *args)

        print_verbose(verbose, f'      got {g1} compared to original {g0} at {x+lam1*p}')

        if __check_acceptance(g1, g0, lam1, slope, alpha):
            print_verbose(verbose, '        accepted lam', lam1)
            return lam1 * normalization, False

    if verbose >= 0:
        warning = f'linesearch(...) failed to find acceptable criterion after {max_its} '
        warning += f'steps. Returning most recent value for lam = {lam1}.'
        warnings.warn(warning)

    return lam1 * normalization, True


def newt(x, func, *args, step_max=100., max_its=200, tol_f=1.e-8, tol_x=1.e-9,
         tol_min=1.e-10, verbose=0):
    """
    Multi-dimensional globally convergence Newton-Raphson root finder with line search as
    described in Numerical Recipes (3rd Ed.) Section 9.7.

    :arg x: Initial guess for root.

    :arg func: Function to find root of to be called as func(x, *args).

    :arg *args: Arguments to be passed to func(...) as func(x, *args).

    :arg step_max: (default=100.) Maximum length for any NR step in line search.

    :arg max_its: (default=200) Maximum number of NR iterations to perform.

    :arg tol_f: (default=1.e-8) Tolerance for root of objective function.

    :arg tol_x: (default=1.e-9) Tolerance for max(|dx|) step.

    :arg tol_min: (default=1.e-10) Criterion for deciding spurious convergence.

    :arg verbose: (default=0) Integer value for verbosity of this (and called) functions.

    :returns: x, check_for_local_minimum
    """

    # is the initial guess a solution?
    if fmin(x, func, *args) < 0.01 * tol_f:
        return x, False

    # limit step size in line search algorithm
    step_max *= max(len(x), np.power(x, 2.).sum())

    # begin iterations
    for it in range(max_its):

        fjac = fdjac(x, func, *args)
        fvec = func(x, *args)
        g = np.sum(fvec * fjac.T, axis=1)

        if not np.isfinite(fjac).all():
            if verbose >= 0:
                warning = 'newt(...) found non-finite value for jacobian on '
                warning += f'iteration {it}. Returning last known valid value.'
                warnings.warn(warning)
            return x, True

        print_verbose(verbose, 'on iteration', it, 'at point', x)
        print_verbose(verbose, 'computed jacobian:\n', fjac)

        p = lu_solve(lu_factor(fjac), -fvec)
        print_verbose(verbose, 'got direction to move', p)

        alpha, check = linesearch(x, g, p, step_max, fmin, func, *args,
                                  tol_x=tol_x, verbose=verbose-1)
        print_verbose(verbose, 'should move along direction by amount', alpha, '\n')
        x_old = x.copy()
        x = x + alpha * p
        fval = fmin(x, func, *args)

        if fval < tol_f:
            print_verbose(verbose, 'found minimum within acceptable tol_f')
            return x, False

        if check:
            test = np.max(np.abs(g)*np.maximum(np.abs(x), 1.) / max(0.5*len(x), fval))
            if test < tol_min:
                print_verbose(verbose, 'possible spurious convergence to function min')
                return x, True
            print_verbose(verbose, 'found minimum within acceptable tolerance')
            return x, False

        if np.max(np.abs(x - x_old) / np.maximum(np.abs(x), 1.)) < tol_x:
            print_verbose(verbose, 'found minimum within acceptable tol_x')
            return x, False

    if verbose >= 0:
        warning = f'newt(...) failed to find minimum after {max_its} iterations. '
        warning += f'Returning most recent value x = {x}.'
        warnings.warn(warning)

    return x, True
