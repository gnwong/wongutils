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


def get_ks_from_eks(x1, x2, x3):
    """Return 3d ks R, H, P arrays from 1d eks x1, x2, x3."""
    return np.meshgrid(np.exp(x1), np.pi*x2, x3, indexing='ij')


def get_ks_from_fmks(coordinate_info, x1, x2, x3):
    """Return 3d ks R, H, P arrays from 1d fmks x1, x2, x3 lists with coordinate_info."""

    Rin = coordinate_info['Rin']
    hslope = coordinate_info['hslope']
    poly_xt = coordinate_info['poly_xt']
    poly_alpha = coordinate_info['poly_alpha']
    mks_smooth = coordinate_info['mks_smooth']
    poly_norm = coordinate_info['poly_norm']

    r = np.exp(x1)

    hg = np.pi*x2 + (1.-hslope) * np.sin(2.*np.pi * x2)/2.
    X1, HG, X3 = np.meshgrid(x1, hg, x3, indexing='ij')

    y = 2.*x2 - 1.
    hj = poly_norm*y*(1.+np.power(y/poly_xt, poly_alpha)/(poly_alpha + 1.)) + 0.5*np.pi
    R, HJ, P = np.meshgrid(r, hj, x3, indexing='ij')
    H = HG + np.exp(mks_smooth*(np.log(Rin) - X1)) * (HJ - HG)

    return R, H, P


def get_dxdX_ks_eks_from_ks(R, H, P=None):
    """Return dx^ks / dx^eks from input ks coordinate mesh. Assumes regularity in P."""

    N1 = R.shape[0]
    N2 = R.shape[1]

    if P is not None:
        R = R[:, :, 0]

    dxdX = np.zeros((N1, N2, 4, 4))
    dxdX[:, :, 0, 0] = 1.
    dxdX[:, :, 1, 1] = R
    dxdX[:, :, 2, 2] = 1.
    dxdX[:, :, 3, 3] = 1.

    if P is not None:
        N3 = R.shape[2]
        dxdX2d = dxdX
        dxdX = np.zeros((N1, N2, N3, 4, 4))
        dxdX[:, :, :, :, :] = dxdX2d[:, :, None, :, :]

    return dxdX


def get_dxdX_ks_fmks_from_fmks(coordinate_info, X1, X2, X3=None):
    """Return dx^ks / dx^fmks from input fmks coordinate mesh with coordinate_info.
    Assumes regularity in P."""

    Rin = coordinate_info['Rin']
    hslope = coordinate_info['hslope']
    poly_xt = coordinate_info['poly_xt']
    poly_alpha = coordinate_info['poly_alpha']
    mks_smooth = coordinate_info['mks_smooth']
    poly_norm = coordinate_info['poly_norm']

    N1 = X1.shape[0]
    N2 = X2.shape[1]

    if X3 is not None or len(X1.shape) == 3:
        X1 = X1[:, :, 0]
        X2 = X2[:, :, 0]

    R = np.exp(X1)

    dxdX = np.zeros((N1, N2, 4, 4))
    dxdX[:, :, 0, 0] = 1.
    dxdX[:, :, 1, 1] = R
    dxdX[:, :, 3, 3] = 1.

    dxdX[:, :, 2, 1] = - np.exp(mks_smooth*(np.log(Rin)-X1))*mks_smooth \
        * (np.pi/2. - np.pi*X2 + poly_norm*(2.*X2-1.)
           * (1. + (np.power((-1.+2.*X2)/poly_xt, poly_alpha))
              / (1.+poly_alpha)) - 1./2.*(1.-hslope)*np.sin(2.*np.pi*X2))

    dxdX[:, :, 2, 2] = np.pi + (1. - hslope)*np.pi*np.cos(2.*np.pi*X2) \
        + np.exp(mks_smooth*(np.log(Rin)-X1)) \
        * (-np.pi + 2.*poly_norm*(1. + np.power((2.*X2-1.)/poly_xt, poly_alpha)
                                  / (poly_alpha+1.))
           + (2*poly_alpha*poly_norm*(2*X2-1)*np.power((2*X2-1)/poly_xt, poly_alpha-1))
           / ((1.+poly_alpha)*poly_xt) - (1.-hslope)*np.pi*np.cos(2.*np.pi*X2))

    if X3 is not None:
        N3 = X3.shape[2]
        dxdX2d = dxdX
        dxdX = np.zeros((N1, N2, N3, 4, 4))
        dxdX[:, :, :, :, :] = dxdX2d[:, :, None, :, :]

    return dxdX
