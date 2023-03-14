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


def get_ks_from_eks(x1, x2, x3, coodinate_info):
    """Return 3d ks R, H, P arrays from 1d eks x1, x2, x3 lists from coordinate_info."""
    return np.meshgrid(np.exp(x1), np.pi*x2, x3, indexing='ij')


def get_ks_from_fmks(x1, x2, x3, coordinate_info):
    """Return 3d ks R, H, P arrays from 1d fmks x1, x2, x3 lists from coordinate_info."""

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
