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
from wongutils.geometry import coordinates


def get_gcov_ks_from_ks(bhspin, R, H, P=None):
    """Return gcov with ks components from ks coordinate
    mesh (R, H, P). This function assumes regularity in P."""

    # get grid geometry
    if P is not None:
        N1, N2, N3 = R.shape
        R = R[:, :, 0]
        H = H[:, :, 0]
    else:
        N1, N2 = R.shape

    # generate metric over 2d mesh
    gcov = np.zeros((N1, N2, 4, 4))
    cth = np.cos(H)
    sth = np.sin(H)
    s2 = sth*sth
    rho2 = R*R + bhspin*bhspin*cth*cth
    gcov[:, :, 0, 0] = (-1. + 2. * R / rho2)
    gcov[:, :, 0, 1] = (2. * R / rho2)
    gcov[:, :, 0, 3] = (-2. * bhspin * R * s2 / rho2)
    gcov[:, :, 1, 0] = gcov[:, :, 0, 1]
    gcov[:, :, 1, 1] = (1. + 2. * R / rho2)
    gcov[:, :, 1, 3] = (-bhspin * s2 * (1. + 2. * R / rho2))
    gcov[:, :, 2, 2] = rho2
    gcov[:, :, 3, 0] = gcov[:, :, 0, 3]
    gcov[:, :, 3, 1] = gcov[:, :, 1, 3]
    gcov[:, :, 3, 3] = s2 * (rho2 + bhspin*bhspin * s2 * (1. + 2. * R / rho2))

    # extend along P dimension if applicable
    if P is not None:
        gcov2d = gcov
        gcov = np.zeros((N1, N2, N3, 4, 4))
        gcov[:, :, :, :, :] = gcov2d[:, :, None, :, :]

    return gcov


def get_gcov_eks_from_ks(bhspin, R, H, P=None):
    """Return gcov with eks components from ks coordinate
    mesh (R, H, P). This function assumes regularity in P."""

    # get grid geometry
    if P is not None:
        N1, N2, N3 = R.shape
        R = R[:, :, 0]
        H = H[:, :, 0]
    else:
        N1, N2 = R.shape

    gcov_ks = get_gcov_ks_from_ks(bhspin, R, H)

    # transform to eks
    dxdX = coordinates.get_dxdX_ks_eks_from_ks(R, H)
    gcov_eks = np.einsum('abki,abkj->abij', dxdX,
                         np.einsum('ablj,abkl->abkj', dxdX, gcov_ks))

    # extend along P dimension if applicable
    if P is not None:
        gcov2d_eks = gcov_eks
        gcov_eks = np.zeros((N1, N2, N3, 4, 4))
        gcov_eks[:, :, :, :, :] = gcov2d_eks[:, :, None, :, :]

    return gcov_eks


def get_gcov_eks_from_eks(bhspin, X1, X2, X3=None):
    """Return gcov with eks components from eks coordinate
    mesh (X1, X2, X3). This function assumes regularity in X3."""

    R = np.exp(X1)
    H = X2
    P = X3

    return get_gcov_eks_from_ks(bhspin, R, H, P=P)


def get_gcov_fmks_from_fmks(coordinate_info, X1, X2, X3=None):
    """Return gcov with fmks components from fmks coordinate
    mesh (x1, x2, x3). This function assumes regularity in x3."""

    # get grid geometry
    if X3 is not None:
        N1, N2, N3 = X1.shape
        x1 = X1[:, 0, 0]
        x2 = X2[0, :, 0]
        x3 = X3[0, 0, :]
    else:
        N1, N2 = X1.shape

    R, H, P = coordinates.get_ks_from_fmks(coordinate_info, x1, x2, x3)
    R = R[:, :, 0]
    H = H[:, :, 0]

    gcov_ks = get_gcov_ks_from_ks(coordinate_info['bhspin'], R, H)

    # transform to fmks
    dxdX = coordinates.get_dxdX_ks_fmks_from_fmks(coordinate_info, X1, X2)
    gcov_fmks = np.einsum('abki,abkj->abij', dxdX,
                          np.einsum('ablj,abkl->abkj', dxdX, gcov_ks))

    print(x1[0], x2[0], R[0, 0], H[0, 0], dxdX[0, 0], gcov_ks[0, 0], gcov_fmks[0, 0])

    # extend along x3 dimension if applicable
    if X3 is not None:
        gcov2d_fmks = gcov_fmks
        gcov_fmks = np.zeros((N1, N2, N3, 4, 4))
        gcov_fmks[:, :, :, :, :] = gcov2d_fmks[:, :, None, :, :]

    return gcov_fmks
