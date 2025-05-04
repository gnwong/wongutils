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

    # check if R is a scalar (bit of a kludge)
    input_was_scalar = False
    if np.isscalar(R):
        R = np.array([R]).reshape((1, 1))
        H = np.array([H]).reshape((1, 1))
        input_was_scalar = True

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

    # if input was a scalar, return a scalar
    if input_was_scalar:
        gcov = gcov[0, 0]

    return gcov


def get_gcov_bl_from_bl(bhspin, R, H, P=None):
    """Return gcov with bl components from bl/ks coordinate
    mesh (R, H, P). This function assumes regularity in P."""

    # check if R is a scalar (bit of a kludge)
    input_was_scalar = False
    if np.isscalar(R):
        R = np.array([R]).reshape((1, 1))
        H = np.array([H]).reshape((1, 1))
        input_was_scalar = True

    # get grid geometry
    if P is not None:
        N1, N2, N3 = R.shape
        R = R[:, :, 0]
        H = H[:, :, 0]
    else:
        N1, N2 = R.shape

    Sigma = R*R + bhspin*bhspin * np.cos(H)*np.cos(H)
    Delta = R*R - 2.*R + bhspin*bhspin

    # generate metric over 2d mesh
    gcov = np.zeros((N1, N2, 4, 4))
    gcov[:, :, 0, 0] = - (1. - 2.*R / Sigma)
    gcov[:, :, 1, 1] = Sigma / Delta
    gcov[:, :, 2, 2] = Sigma
    gcov[:, :, 3, 3] = R*R + bhspin*bhspin + 2.*R*bhspin*bhspin*np.sin(H)*np.sin(H)/Sigma
    gcov[:, :, 3, 3] *= np.sin(H) * np.sin(H)
    gcov[:, :, 0, 3] = -2. * R * bhspin * np.sin(H)*np.sin(H) / Sigma
    gcov[:, :, 3, 0] = gcov[:, :, 0, 3]

    # extend along P dimension if applicable
    if P is not None:
        gcov2d = gcov
        gcov = np.zeros((N1, N2, N3, 4, 4))
        gcov[:, :, :, :, :] = gcov2d[:, :, None, :, :]

    # if input was a scalar, return a scalar
    if input_was_scalar:
        gcov = gcov[0, 0]

    return gcov


def get_gcon_bl_from_bl(bhspin, R, H, P=None):
    """Return gcon with bl components from bl/ks coordinate
    mesh (R, H, P). This function assumes regularity in P."""

    # check if R is a scalar (bit of a kludge)
    input_was_scalar = False
    if np.isscalar(R):
        R = np.array([R]).reshape((1, 1))
        H = np.array([H]).reshape((1, 1))
        input_was_scalar = True

    # get grid geometry
    if P is not None:
        N1, N2, N3 = R.shape
        R = R[:, :, 0]
        H = H[:, :, 0]
    else:
        N1, N2 = R.shape

    Sigma = R*R + bhspin*bhspin * np.cos(H)*np.cos(H)
    Delta = R*R - 2.*R + bhspin*bhspin

    # generate metric over 2d mesh
    gcon = np.zeros((N1, N2, 4, 4))
    gcon[:, :, 0, 0] = - (R*R + bhspin*bhspin + 2.*R*bhspin*bhspin
                          / Sigma*np.sin(H)*np.sin(H)) / Delta
    gcon[:, :, 1, 1] = Delta / Sigma
    gcon[:, :, 2, 2] = 1. / Sigma
    gcon[:, :, 3, 3] = Delta - bhspin*bhspin*np.sin(H)*np.sin(H)
    gcon[:, :, 3, 3] /= Sigma * Delta * np.sin(H)*np.sin(H)
    gcon[:, :, 0, 3] = -2. * R * bhspin / Sigma / Delta
    gcon[:, :, 3, 0] = gcon[:, :, 0, 3]

    # extend along P dimension if applicable
    if P is not None:
        gcon2d = gcon
        gcon = np.zeros((N1, N2, N3, 4, 4))
        gcon[:, :, :, :, :] = gcon2d[:, :, None, :, :]

    # if input was a scalar, return a scalar
    if input_was_scalar:
        gcon = gcon[0, 0]

    return gcon


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


def get_gcov_cks_from_cks(bhspin, X1, X2, X3):
    """Return gcov with cks components from cks coordinate
    mesh (X1, X2, X3)."""

    # check if X1 is a scalar (bit of a kludge)
    input_was_scalar = False
    if np.isscalar(X1):
        X1 = np.array([X1]).reshape((1, 1, 1))
        X2 = np.array([X2]).reshape((1, 1, 1))
        X3 = np.array([X3]).reshape((1, 1, 1))
        input_was_scalar = True

    R = np.sqrt(X1*X1 + X2*X2 + X3*X3)
    r = np.sqrt(R**2 - bhspin**2 + np.sqrt((R**2-bhspin**2)**2 + 4*bhspin**2*X3**2)) \
        / np.sqrt(2.0)

    f = 2. * np.power(r, 3.) / (np.power(r, 4.) + bhspin*bhspin*X3*X3)
    l0 = 1.
    l1 = (r*X1 + bhspin*X2) / (r**2 + bhspin**2)
    l2 = (r*X2 - bhspin*X1) / (r**2 + bhspin**2)
    l3 = X3 / r

    Nx, Ny, Nz = X1.shape
    gcov_cks = np.zeros((Nx, Ny, Nz, 4, 4))

    gcov_cks[..., 0, 0] = -1. + f * l0*l0
    gcov_cks[..., 0, 1] = f * l0*l1
    gcov_cks[..., 1, 0] = gcov_cks[..., 0, 1]
    gcov_cks[..., 0, 2] = f * l0*l2
    gcov_cks[..., 2, 0] = gcov_cks[..., 0, 2]
    gcov_cks[..., 0, 3] = f * l0*l3
    gcov_cks[..., 3, 0] = gcov_cks[..., 0, 3]
    gcov_cks[..., 1, 1] = 1. + f * l1*l1
    gcov_cks[..., 1, 3] = f * l1*l3
    gcov_cks[..., 3, 1] = gcov_cks[..., 1, 3]
    gcov_cks[..., 2, 2] = 1. + f * l2*l2
    gcov_cks[..., 2, 3] = f * l2*l3
    gcov_cks[..., 3, 2] = gcov_cks[..., 2, 3]
    gcov_cks[..., 1, 2] = f * l1*l2
    gcov_cks[..., 2, 1] = gcov_cks[..., 1, 2]
    gcov_cks[..., 3, 3] = 1. + f * l3*l3

    # if input was a scalar, return a scalar
    if input_was_scalar:
        gcov_cks = gcov_cks[0, 0, 0]

    return gcov_cks


def get_gcon_cks_from_cks(bhspin, X1, X2, X3):
    """Return gcon with cks components from cks coordinate
    mesh (X1, X2, X3)."""

    # check if X1 is a scalar (bit of a kludge)
    input_was_scalar = False
    if np.isscalar(X1):
        X1 = np.array([X1]).reshape((1, 1, 1))
        X2 = np.array([X2]).reshape((1, 1, 1))
        X3 = np.array([X3]).reshape((1, 1, 1))
        input_was_scalar = True

    R = np.sqrt(X1*X1 + X2*X2 + X3*X3)
    r = np.sqrt(R**2 - bhspin**2 + np.sqrt((R**2-bhspin**2)**2 + 4*bhspin**2*X3**2)) \
        / np.sqrt(2.0)

    f = 2. * np.power(r, 3.) / (np.power(r, 4.) + bhspin*bhspin*X3*X3)
    l0 = -1.
    l1 = (r*X1 + bhspin*X2) / (r**2 + bhspin**2)
    l2 = (r*X2 - bhspin*X1) / (r**2 + bhspin**2)
    l3 = X3 / r

    Nx, Ny, Nz = X1.shape
    gcon_cks = np.zeros((Nx, Ny, Nz, 4, 4))

    gcon_cks[..., 0, 0] = -1. - f * l0*l0
    gcon_cks[..., 0, 1] = -f * l0*l1
    gcon_cks[..., 1, 0] = gcon_cks[..., 0, 1]
    gcon_cks[..., 0, 2] = -f * l0*l2
    gcon_cks[..., 2, 0] = gcon_cks[..., 0, 2]
    gcon_cks[..., 0, 3] = -f * l0*l3
    gcon_cks[..., 3, 0] = gcon_cks[..., 0, 3]
    gcon_cks[..., 1, 1] = 1. - f*l1*l1
    gcon_cks[..., 1, 3] = -f * l1*l3
    gcon_cks[..., 3, 1] = gcon_cks[..., 1, 3]
    gcon_cks[..., 2, 2] = 1. - f * l2*l2
    gcon_cks[..., 2, 3] = -f * l2*l3
    gcon_cks[..., 3, 2] = gcon_cks[..., 2, 3]
    gcon_cks[..., 1, 2] = -f * l1*l2
    gcon_cks[..., 2, 1] = gcon_cks[..., 1, 2]
    gcon_cks[..., 3, 3] = 1. - f * l3*l3

    # if input was a scalar, return a scalar
    if input_was_scalar:
        gcon_cks = gcon_cks[0, 0, 0]

    return gcon_cks


def get_gcov_fmks_from_fmks(coordinate_info, X1, X2, X3=None):
    """Return gcov with fmks components from fmks coordinate
    mesh (X1, X2, X3). This function assumes regularity in x3."""

    # deal with possible input scalars
    input_was_scalar = False
    if np.isscalar(X1):
        X1 = np.array([X1]).reshape((1, 1))
        X2 = np.array([X2]).reshape((1, 1))
        input_was_scalar = True

    # get grid geometry
    if X3 is not None:
        N1, N2, N3 = X1.shape
        x1 = X1[:, 0, 0]
        x2 = X2[0, :, 0]
        x3 = X3[0, 0, :]
    else:
        N1, N2 = X1.shape
        x1 = X1[:, 0]
        x2 = X2[0, :]
        x3 = [0.]

    R, H, P = coordinates.get_ks_from_fmks(coordinate_info, x1, x2, x3)
    R = R[:, :, 0]
    H = H[:, :, 0]

    gcov_ks = get_gcov_ks_from_ks(coordinate_info['bhspin'], R, H)

    # transform to fmks
    dxdX = coordinates.get_dxdX_ks_fmks_from_fmks(coordinate_info, X1, X2)
    gcov_fmks = np.einsum('abki,abkj->abij', dxdX,
                          np.einsum('ablj,abkl->abkj', dxdX, gcov_ks))

    # extend along x3 dimension if applicable
    if X3 is not None:
        gcov2d_fmks = gcov_fmks
        gcov_fmks = np.zeros((N1, N2, N3, 4, 4))
        gcov_fmks[:, :, :, :, :] = gcov2d_fmks[:, :, None, :, :]

    # if input was a scalar, return a scalar
    if input_was_scalar:
        gcov_fmks = gcov_fmks[0, 0]

    return gcov_fmks
