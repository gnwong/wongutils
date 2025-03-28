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
from scipy.special import ellipk  # pylint: disable=no-name-in-module


def get_Q(bhspin, r):
    """
    Normalized Carter constant in Kerr spacetime.

    :arg bhspin: dimensionless spin of black hole
    :arg r: radius of desired orbit in units of GM/c^2

    :returns: Q
    """
    Q = r*r*r * (r*r*r-6.*r*r + 9.*r - 4.*bhspin*bhspin)
    return - Q / bhspin/bhspin/(r-1.)/(r-1.)


def get_Phi(bhspin, r):
    """
    Normalized conserved z angluar momentum in Kerr spacetime.

    :arg bhspin: dimensionless spin of black hole
    :arg r: radius of desired orbit in units of GM/c^2

    :returns: Phi
    """
    Phi = r*r*r - 3.*r*r + bhspin*bhspin*r + bhspin*bhspin
    return - Phi / bhspin / (r - 1.)


def get_rpm(bhspin):
    """
    Returns minimum and maximum radii of circular photon orbits in Kerr spacetime.
    By convention r_+ > r_-, which means that r_+ is retrograde orbit while r_- is
    prograde.

    :arg bhspin: dimensionless spin of black hole

    :returns: r_- and r_+
    """
    rp = 1. + np.cos(2./3. * np.arccos(bhspin))
    rm = 1. + np.cos(2./3. * np.arccos(-bhspin))
    return 2.*rp, 2.*rm


def get_rpolar(bhspin):
    """
    Returns radius of polar orbit in Kerr spacetime.

    :arg bhspin: dimensionless spin of black hole

    :returns: r_polar
    """
    r_polar = np.arccos((1. - bhspin*bhspin)/np.power(1. - bhspin*bhspin/3., 3./2))/3.
    r_polar = np.cos(r_polar) * np.sqrt(1. - bhspin*bhspin/3.) * 2 + 1.

    return r_polar


def get_upmsq(bhspin, r):
    """
    Returns the square of roots of transformed theta equation of motion. Note that
    these are the same as u_pm of arxiv:1907.04329 (i.e., not the square of those
    quantities). Here u := cos(theta). See also Equation A14 of arxiv:2009.06641.

    :arg bhspin: dimensionless spin of black hole
    :arg r: radius of desired orbit in units of GM/c^2

    :returns: (u_p)^2, (u_m)^2
    """

    Phi = get_Phi(bhspin, r)
    Q = get_Q(bhspin, r)

    upsq = np.sqrt(np.power(Q + Phi*Phi - bhspin*bhspin, 2.) + 4.*bhspin*bhspin*Q)
    umsq = - upsq

    upsq += bhspin*bhspin - Q - Phi*Phi
    umsq += bhspin*bhspin - Q - Phi*Phi

    upsq /= 2. * bhspin*bhspin
    umsq /= 2. * bhspin*bhspin

    return upsq, umsq


def get_lyapunov_theta(bhspin, r):
    """
    Returns Lyapunov exponent for circular photon orbit at given radius in Kerr
    spacetime with given spin. This exponent corresponds to theta oscillations.

    :arg bhspin: dimensionless spin of black hole
    :arg r: radius of desired orbit in units of GM/c^2

    :returns: Lyapunov exponent
    """

    upsq, umsq = get_upmsq(bhspin, r)

    Delta = r*r - 2.*r + bhspin*bhspin

    lyap = 4. / (bhspin * np.sqrt(-umsq)) * np.sqrt(r*r - r*Delta / (r-1.)/(r-1.))
    lyap *= ellipk(upsq/umsq)

    return lyap


def get_critical_curve(bhspin, inc, npts=100000):
    """
    Return the "critical curve" (the boundary of the black hole shadow) given
    a black hole spin and inclination angle sampled along the "top half" of
    the image plane.

    :arg bhspin: dimensionless spin of the black hole
    :arg inc: inclination angle in degrees
    :arg npts: (default) number of radial points to evaluate for curve

    :returns: (alpha, beta) coordinates of critical curve in GM/c^2 on image plane
    """

    inc = np.radians(inc)

    rmin, rmax = get_rpm(bhspin)
    rs = np.linspace(rmin, rmax, npts)

    xi = - (rs**2. * (rs - 3.) + bhspin**2. * (rs + 1.)) / bhspin / (rs - 1.)
    eta = rs**3. * (4. * bhspin**2. - rs * (rs - 3.)**2.)
    eta = eta / (bhspin * (rs-1.))**2.

    alphas = - xi / np.sin(inc)
    betas = np.sqrt(eta + bhspin**2.*np.cos(inc)**2. - xi**2./np.tan(inc)**2.)

    mask = np.isfinite(alphas) & np.isfinite(betas)

    return alphas[mask], betas[mask]
