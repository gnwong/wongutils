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
    norm = np.einsum('...a,...a->...', np.einsum('...ab,...b->...a', metric, v), v)
    return np.sqrt(-1./norm)


def _set_subkep_bl_Ucon(r, h, bhspin, subkep):
    """
    Return the Boyer-Lindquist four-velocity for a subkeplerian orbit.
    This is a Keplerian orbit with the specific angular momentum
    rescaled by the subkeplerian factor.
    """

    was_scalar = False
    if np.isscalar(r):
        was_scalar = True
        r = np.array([r]).reshape((1, 1))
        h = np.array([h]).reshape((1, 1))

    # get Boyer-Lindquist metric
    gcov_bl = metrics.get_gcov_bl_from_bl(bhspin, r, h)
    gcon_bl = metrics.get_gcon_bl_from_bl(bhspin, r, h)

    # get BL Keplerian angular velocity (in BL coordinates)
    Omega_kep = 1. / (np.power(r, 1.5) + bhspin)
    bl_Ucon = np.zeros((*r.shape, 4))
    bl_Ucon[..., 0] = 1.
    bl_Ucon[..., 3] = Omega_kep
    bl_Ucon *= np.expand_dims(_normalization(gcov_bl, bl_Ucon), axis=-1)

    # get angular momentum for Keplerian orbit and rescale
    bl_Ucov = np.einsum('...ab,...b->...a', gcov_bl, bl_Ucon)
    L = - bl_Ucov[..., 3] / bl_Ucov[..., 0] * subkep
    bl_Ucov[..., 0] = -1.
    bl_Ucov[..., 1] = 0.
    bl_Ucov[..., 2] = 0.
    bl_Ucov[..., 3] = L
    bl_Ucov *= np.expand_dims(_normalization(gcon_bl, bl_Ucov), axis=-1)
    bl_Ucon = np.einsum('...ab,...b->...a', gcon_bl, bl_Ucov)

    if was_scalar:
        bl_Ucon = bl_Ucon[0, 0]
        bl_Ucov = bl_Ucov[0, 0]

    return bl_Ucon, bl_Ucov


def _bl_subkep_cunningham(r, h, bhspin, subkep):
    """
    Works for any radial position, evaluates for spherical rather
    than cylindrical radius.
    """

    was_scalar = False
    if np.isscalar(r):
        was_scalar = True
        r = np.array([r]).reshape((1, 1))
        h = np.array([h]).reshape((1, 1))

    # get Boyer-Lindquist metric
    gcon_bl = metrics.get_gcon_bl_from_bl(bhspin, r, h)

    # get special radii
    reh = 1. + np.sqrt(1. - bhspin * bhspin)
    r_isco = _r_isco(bhspin)

    # mask as needed
    mask_eh = r < reh
    mask_isco = r < r_isco

    # compute normal observer
    bl_Ucov_normobs = np.zeros((*r.shape, 4))
    if mask_eh.any():
        bl_Ucov_normobs[..., 0] = -1.
        bl_Ucov_normobs[..., 0] *= _normalization(gcon_bl, bl_Ucov_normobs)
    bl_Ucon_normobs = np.einsum('...ab,...b->...a', gcon_bl, bl_Ucov_normobs)

    # compute subkeplerian velocity within the ISCO
    r_isco = np.ones_like(h) * r_isco
    bl_Ucon_isco, bl_Ucov_isco = _set_subkep_bl_Ucon(r_isco, h, bhspin, subkep)
    if mask_isco.any():
        E = bl_Ucov_isco[..., 0]
        L = bl_Ucov_isco[..., 3]
        vr = 1.+gcon_bl[..., 0, 0]*E*E+2.*gcon_bl[..., 0, 3]*E*L+gcon_bl[..., 3, 3]*L*L
        vr /= gcon_bl[..., 1, 1]
        vr = - np.sqrt(np.maximum(0, -vr))
        bl_Ucov_isco[..., 0] = E
        bl_Ucov_isco[..., 1] = vr
        bl_Ucov_isco[..., 2] = 0.
        bl_Ucov_isco[..., 3] = L
        bl_Ucov_isco *= np.expand_dims(_normalization(gcon_bl, bl_Ucov_isco), axis=-1)
        bl_Ucon_isco = np.einsum('...ab,...b->...a', gcon_bl, bl_Ucov_isco)

    # compute subkeplerian velocity outside the ISCO
    bl_Ucon_subkep, bl_Ucov_subkep = _set_subkep_bl_Ucon(r, h, bhspin, subkep)

    bl_Ucon_subkep[mask_isco, :] = bl_Ucon_isco[mask_isco, :]
    bl_Ucon_subkep[mask_eh, :] = bl_Ucon_normobs[mask_eh, :]

    bl_Ucov_subkep[mask_isco, :] = bl_Ucov_isco[mask_isco, :]
    bl_Ucov_subkep[mask_eh, :] = bl_Ucov_normobs[mask_eh, :]

    if was_scalar:
        bl_Ucon_subkep = bl_Ucon_subkep[0, 0]
        bl_Ucov_subkep = bl_Ucov_subkep[0, 0]

    return bl_Ucon_subkep, bl_Ucov_subkep


def ucon_bl_general_subkep(r, h, bhspin, subkep, beta_r, beta_phi):
    """
    Return ucon in BL coordinates for "general subkeplerian/freefall" velocity
    given a location (r, h) also supplied in BL coordaintes.

    :arg r: radial coordinate in BL coordinates
    :arg h: height coordinate in BL coordinates
    :arg bhspin: black hole spin
    :arg subkep: subkeplerian factor (1 = keplerian)
    """

    was_scalar = False
    if np.isscalar(r):
        was_scalar = True
        r = np.array([r]).reshape((1, 1))
        h = np.array([h]).reshape((1, 1))

    # get Boyer-Lindquist metric
    gcov_bl = metrics.get_gcov_bl_from_bl(bhspin, r, h)
    gcon_bl = metrics.get_gcon_bl_from_bl(bhspin, r, h)

    # get subkep velocity
    bl_Ucon_subkep, _ = _bl_subkep_cunningham(r, h, bhspin, subkep)

    # get freefall velocity
    bl_Ucon_ff = np.zeros((*r.shape, 4))
    bl_Ucon_ff[..., 0] = - gcon_bl[..., 0, 0]
    bl_Ucon_ff[..., 1] = - np.sqrt((-1. - gcon_bl[..., 0, 0]) * gcon_bl[..., 1, 1])
    bl_Ucon_ff[..., 2] = 0.
    bl_Ucon_ff[..., 3] = - gcon_bl[..., 0, 3]

    # combine velocity models
    ur = bl_Ucon_subkep[..., 1] + (1.-beta_r)*(bl_Ucon_ff[..., 1]-bl_Ucon_subkep[..., 1])
    Omega_circ = bl_Ucon_subkep[..., 3] / bl_Ucon_subkep[..., 0]
    Omega_ff = bl_Ucon_ff[..., 3] / bl_Ucon_ff[..., 0]
    Omega = Omega_circ + (1. - beta_phi) * (Omega_ff - Omega_circ)
    bl_Ucon = np.zeros((*r.shape, 4))
    bl_Ucon[..., 0] = 1. + gcov_bl[..., 1, 1] * ur * ur
    denom = gcov_bl[..., 0, 0] + 2*Omega*gcov_bl[..., 0, 3] + gcov_bl[..., 3, 3]*Omega**2
    bl_Ucon[..., 0] /= denom
    bl_Ucon[..., 0] = np.sqrt(-bl_Ucon[..., 0])
    bl_Ucon[..., 1] = ur
    bl_Ucon[..., 3] = Omega * bl_Ucon[..., 0]
    bl_Ucov = np.einsum('...ab,...b->...a', gcov_bl, bl_Ucon)

    if was_scalar:
        bl_Ucon = bl_Ucon[0, 0]
        bl_Ucov = bl_Ucov[0, 0]

    return bl_Ucon, bl_Ucov
