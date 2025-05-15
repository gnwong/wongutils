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

"""
Functionality to translate between various coordinate systems for the
coordinate locations as well as tensors.

x_[Coordinates A]_from_[Coordinates B]:
    Takes arbitrarily shaped coordinate arrays and translates them from
    coordinate system A to coordinate system B. May require the user to
    supply coordinate_information.

get_[Coordinates A]_from_[Coordinates B]:
    Takes 1d coordinate arrays and returns 3d coordinate arrays in
    coordinate system A. May require the user to supply coordinate
    information.

get_dxdX_[Coordinates A]_[Coordinates B]_from_[Coordinates C]:
    Return the Jacobian dx^A / dX^B from coordinate system B to A at
    the coordinate point(s) specified in coordinate system C. Output
    from this function can be used as:

    xcon_A = [...]
    vcon_B = [...]

    dxdX_CB = get_dxdX_...(xcon_A)
    vcon_A = np.einsum('...ij,...j->...i', dxdX_CB, vcon_B)

"""


def x_ks_from_cks(coordinate_info, x, y, z):
    """Translate x, y, z from CKS to KS given 'coordinate_info'
    dictionary containing 'bhspin'. Works for arbitrary shapes."""
    bhspin = coordinate_info['bhspin']
    R = np.sqrt(np.power(x, 2.) + np.power(y, 2.) + np.power(z, 2.))
    r = np.sqrt(R**2 - bhspin*bhspin
                + np.sqrt(np.power(np.power(R, 2.) - bhspin*bhspin, 2.)
                          + 4.*bhspin*bhspin * np.power(z, 2.))) / np.sqrt(2.)
    h = np.arccos(z / r)
    p = np.arctan2(r * y - bhspin * x, r * x + bhspin * y)
    if np.isscalar(p):
        if p < 0:
            p += 2. * np.pi
    else:
        p[p < 0] += 2. * np.pi
    return r, h, p


def x_cks_from_ks(coordinate_info, r, h, p):
    """Translate r, h, p from KS to CKS given 'coordinate_info'
    dictionary containing 'bhspin'. Works for arbitrary shapes."""
    bhspin = coordinate_info['bhspin']
    x = r * np.cos(p) * np.sin(h) - bhspin * np.sin(p) * np.sin(h)
    y = r * np.sin(p) * np.sin(h) + bhspin * np.cos(p) * np.sin(h)
    z = r * np.cos(h)
    return x, y, z


def x_ks_from_eks(x1, x2, x3):
    """Translate x1, x2, x3 from EKS to KS coordinates."""
    r = np.exp(x1)
    h = np.pi*x2
    p = x3
    return r, h, p


def x_ks_from_fmks(coordinate_info, x1, x2, x3):
    """Translate x1, x2, x3 from FMKS to KS coordinates given
    'coordinate_info' dictionary. Works for arbitrary shapes."""
    Rin = coordinate_info['Rin']
    hslope = coordinate_info['hslope']
    poly_xt = coordinate_info['poly_xt']
    poly_alpha = coordinate_info['poly_alpha']
    mks_smooth = coordinate_info['mks_smooth']
    poly_norm = coordinate_info['poly_norm']

    r = np.exp(x1)
    hg = np.pi*x2 + (1.-hslope) * np.sin(2.*np.pi * x2)/2.
    y = 2.*x2 - 1.
    hj = poly_norm*y*(1.+np.power(y/poly_xt, poly_alpha)/(poly_alpha + 1.)) + 0.5*np.pi
    h = hg + np.exp(mks_smooth*(np.log(Rin) - x1)) * (hj - hg)
    p = x3

    return r, h, p


def x_eks_from_ks(r, h, p):
    """Translate r, h, p from KS to EKS coordinates."""
    x1 = np.log(r)
    x2 = h / np.pi
    x3 = p
    return x1, x2, x3


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


def _fmks_compute_poly_norm(coordinate_info):
    """Compute poly_norm factor in fmks coordinate system given
    coordinate_info dictionary."""
    poly_alpha = coordinate_info['poly_alpha']
    poly_xt = coordinate_info['poly_xt']
    poly_norm = np.pi / (1. + 1. / (poly_alpha + 1.) / pow(poly_xt, poly_alpha)) / 2.
    return poly_norm


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


def get_dxdX_ks_bl_from_ks(bhspin, r, h):
    """Return dx^ks / dx^bl from input bl coordinate mesh with
    bhspin."""
    # vcon_bl = np.einsum('ij,j->i', dxdx, vcon_ks)
    a = bhspin
    dxdx = np.zeros((*r.shape, 4, 4))
    dxdx[..., 0, 0] = 1.
    dxdx[..., 1, 1] = 1.
    dxdx[..., 2, 2] = 1.
    dxdx[..., 3, 3] = 1.
    dxdx[..., 0, 1] = 2. * r / (r * r - 2. * r + a * a)
    dxdx[..., 3, 1] = a / (r * r - 2. * r + a * a)
    return dxdx


def get_dxdX_ks_cks_from_cks(bhspin, x, y, z):
    """Return dx^ks / dx^cks from input cks coordinate mesh with
    bhspin."""
    # vcon_ks = np.einsum('ij,j->i', dxdx, vcon_cks)
    a = bhspin
    R = np.sqrt(x**2+y**2+z**2)
    r = np.sqrt(R**2 - a**2 + np.sqrt((R**2-a**2)**2 + 4.0*a**2*z**2))/np.sqrt(2.0)
    SMALL = 1e-15
    sqrt_term = 2.0*r**2-R**2+a**2
    dx_dx = np.zeros((*R.shape, 4, 4))
    dx_dx[..., 0, 0] = 1.
    dx_dx[..., 1, 1] = (x*r)/sqrt_term
    dx_dx[..., 1, 2] = (y*r)/sqrt_term
    dx_dx[..., 1, 3] = z/r * (r**2+a**2)/sqrt_term
    dx_dx[..., 2, 1] = (x*z)/(r * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL)
    dx_dx[..., 2, 2] = (y*z)/(r * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL)
    dx_dx[..., 2, 3] = ((z*z)*(r**2+a**2)/(r**3 * sqrt_term*np.sqrt(1.-z**2/r**2)+SMALL)
                        - 1.0/(r*np.sqrt(1.0-z**2/r**2) + SMALL))
    dx_dx[..., 3, 1] = (-y/(x**2+y**2+SMALL) + a*r*x/((r**2+a**2)*sqrt_term))
    dx_dx[..., 3, 2] = (x/(x**2+y**2+SMALL) + a*r*y/((r**2+a**2)*sqrt_term))
    dx_dx[..., 3, 3] = a*z/r/sqrt_term
    return dx_dx


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
