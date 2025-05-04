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
import wongutils.geometry.metrics as metrics


def _r_isco(bhspin):
    """Return radius of ISCO for black hole with spin bhspin."""
    z1 = 1. + np.power(1. - bhspin * bhspin, 1. / 3.) * (np.power(1. + bhspin, 1. / 3.)
                                                         + np.power(1. - bhspin, 1. / 3.))
    z2 = np.sqrt(3. * bhspin * bhspin + z1 * z1)
    if bhspin < 0:
        r_isco = 3. + z2 + np.sqrt((3. - z1) * (3. + z1 + 2. * z2))
    else:
        r_isco = 3. + z2 - np.sqrt((3. - z1) * (3. + z1 + 2. * z2))
    return r_isco


def _normalization(metric, v):
    """
    Get prefactor to normalize the four-velocity using the metric.
    """
    norm = np.einsum('a,a->', np.einsum('ab,b->a', metric, v), v)
    return np.sqrt(-1./norm)


def _set_subkep_bl_Ucon(r, h, bhspin, subkep):
    """
    Return the Boyer-Lindquist four-velocity for a subkeplerian orbit.
    This is a Keplerian orbit with the specific angular momentum
    rescaled by the subkeplerian factor.
    """

    # get Boyer-Lindquist metric
    gcov_bl = metrics.get_gcov_bl_from_bl(bhspin, r, h)
    gcon_bl = metrics.get_gcon_bl_from_bl(bhspin, r, h)

    # get BL Keplerian angular velocity (in BL coordinates)
    Omega_kep = 1. / (np.power(r, 1.5) + bhspin)
    bl_Ucon = np.zeros(4)
    bl_Ucon[0] = 1.
    bl_Ucon[3] = Omega_kep
    bl_Ucon *= _normalization(gcov_bl, bl_Ucon)

    # get angular momentum for Keplerian orbit and rescale
    bl_Ucov = np.einsum('ab,b->a', gcov_bl, bl_Ucon)
    L = - bl_Ucov[3] / bl_Ucov[0] * subkep
    bl_Ucov[0] = -1.
    bl_Ucov[1] = 0.
    bl_Ucov[2] = 0.
    bl_Ucov[3] = L
    bl_Ucov *= _normalization(gcon_bl, bl_Ucov)
    bl_Ucon = np.einsum('ab,b->a', gcon_bl, bl_Ucov)

    return bl_Ucon, bl_Ucov


def _bl_subkep_cunningham(r, h, bhspin, subkep):
    """
    Works for any radial position, evaluates for spherical rather
    than cylindrical radius.
    """

    # get Boyer-Lindquist metric
    gcon_bl = metrics.get_gcon_bl_from_bl(bhspin, r, h)

    # get special radii
    reh = 1. + np.sqrt(1. - bhspin * bhspin)
    r_isco = _r_isco(bhspin)

    if r < reh:
        # set to normal observer within horizon
        bl_Ucov = np.zeros(4)
        bl_Ucov[0] = -1.
        bl_Ucov[1] = 0.
        bl_Ucov[2] = 0.
        bl_Ucov[3] = 0.
        bl_Ucov *= _normalization(gcon_bl, bl_Ucov)
        bl_Ucon = np.einsum('ab,b->a', gcon_bl, bl_Ucov)
    elif r < r_isco:
        # keep same E, L as at ISCO
        bl_Ucon, bl_Ucov = _set_subkep_bl_Ucon(r_isco, h, bhspin, subkep)
        E = bl_Ucov[0]
        L = bl_Ucov[3]
        vr = 1. + gcon_bl[0, 0]*E*E + 2.*gcon_bl[0, 3]*E*L + gcon_bl[3, 3]*L*L
        vr /= gcon_bl[1, 1]
        vr = - np.sqrt(max(0, -vr))
        bl_Ucov[0] = E
        bl_Ucov[1] = vr
        bl_Ucov[2] = 0.
        bl_Ucov[3] = L
        bl_Ucov *= _normalization(gcon_bl, bl_Ucov)
        bl_Ucon = np.einsum('ab,b->a', gcon_bl, bl_Ucov)
    else:
        # do usual thing outside of ISCO
        bl_Ucon, bl_Ucov = _set_subkep_bl_Ucon(r, h, bhspin, subkep)

    return bl_Ucon, bl_Ucov


def ucon_bl_general_subkep(r, h, bhspin, subkep, beta_r, beta_phi):
    """
    Return ucon in BL coordinates for "general subkeplerian/freefall" velocity
    given a location (r, h) also supplied in BL coordaintes.

    :arg r: radial coordinate in BL coordinates
    :arg h: height coordinate in BL coordinates
    :arg bhspin: black hole spin
    :arg subkep: subkeplerian factor (1 = keplerian)
    """

    if False:
        # broken kludge
        input_was_scalar = False
        if np.isscalar(r):
            input_was_scalar = True
            r = np.array([r]).reshape((1, 1))
            h = np.array([h]).reshape((1, 1))

    # get Boyer-Lindquist metric
    gcov_bl = metrics.get_gcov_bl_from_bl(bhspin, r, h)
    gcon_bl = metrics.get_gcon_bl_from_bl(bhspin, r, h)

    # get subkep velocity
    bl_Ucon_subkep, bl_Ucov_subkep = _bl_subkep_cunningham(r, h, bhspin, subkep)

    # get freefall velocity
    ur_ff = - np.sqrt((-1. - gcon_bl[..., 0, 0]) * gcon_bl[..., 1, 1])
    bl_Ucon_ff = np.zeros(4)
    bl_Ucon_ff[..., 0] = - gcon_bl[..., 0, 0]
    bl_Ucon_ff[..., 1] = ur_ff
    bl_Ucon_ff[..., 2] = 0.
    bl_Ucon_ff[..., 3] = - gcon_bl[..., 0, 3]

    # combine stuff
    ur = bl_Ucon_subkep[..., 1] + (1.-beta_r)*(bl_Ucon_ff[..., 1]-bl_Ucon_subkep[..., 1])
    Omega_circ = bl_Ucon_subkep[..., 3] / bl_Ucon_subkep[..., 0]
    Omega_ff = bl_Ucon_ff[3] / bl_Ucon_ff[0]
    Omega = Omega_circ + (1. - beta_phi) * (Omega_ff - Omega_circ)
    bl_Ucon = np.zeros(4)
    bl_Ucon[0] = 1. + gcov_bl[1, 1] * ur * ur
    bl_Ucon[0] /= gcov_bl[0, 0] + 2. * Omega * gcov_bl[0, 3] + gcov_bl[3, 3] * Omega**2.
    bl_Ucon[0] = np.sqrt(-bl_Ucon[0])
    bl_Ucon[1] = ur
    bl_Ucon[2] = 0.
    bl_Ucon[3] = Omega * bl_Ucon[0]
    bl_Ucov = np.einsum('ab,b->a', gcov_bl, bl_Ucon)

    # recast to scalar if appropriate
    if False and input_was_scalar:
        print(bl_Ucon.shape)

    return bl_Ucon, bl_Ucov
