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


def fishbone_moncrief(R, H, gamma, rin, rmax, bhspin, rho_floor=1.e-7, uu_floor=1.e-7):
    """Return Fishbone-Moncrief torus profile for 2d KS R, H mesh with
    parameters rin, rmax and for black hole with spin bhspin.

    Fishbone & Moncrief 1976, ApJ 207 962
    Fishbone 1977, ApJ 215 323"""

    def lfish_calc(bhspin, R):
        return (((bhspin*bhspin - 2.*bhspin*np.sqrt(R) + R*R)
                 * ((-2.*bhspin * R * (bhspin*bhspin - 2.*bhspin*np.sqrt(R) + R * R))
                    / np.sqrt(2. * bhspin * np.sqrt(R) + (-3.+R)*R)
                    + ((bhspin + (-2.+R) * np.sqrt(R)) * (R*R*R+bhspin*bhspin*(2.+R)))
                    / np.sqrt(1 + (2. * bhspin) / np.power(R, 1.5) - 3. / R)))
                / (R*R*R * np.sqrt(2. * bhspin * np.sqrt(R) + (-3. + R) * R)
                   * (bhspin*bhspin + (-2+R) * R)))

    kappa = 1.e-3

    thin = np.pi/2.

    sth = np.sin(H)
    cth = np.cos(H)
    sthin = np.sin(thin)
    cthin = np.cos(thin)

    bhspinsq = bhspin*bhspin

    lc = lfish_calc(bhspin, rmax)

    # solve for torus profile
    DD = R*R - 2.*R + bhspinsq
    AA = (R*R + bhspinsq)**2 - DD * bhspinsq * sth*sth
    SS = R*R + bhspinsq * cth*cth
    DDin = rin*rin - 2.*rin + bhspinsq
    AAin = (rin*rin + bhspinsq) * (rin*rin + bhspinsq) - DDin * bhspinsq * sthin*sthin
    SSin = rin*rin + bhspinsq * cthin*cthin

    lnh = 0.5 * np.log((1 + np.sqrt(1. + 4*(lc*lc * SS*SS) * DD / (AA*AA*sth*sth)))
                       / (SS*DD/AA)) - 0.5 * np.sqrt(1 + 4*(lc*lc*SS*SS) * DD
                                                     / (AA*AA * sth*sth)) \
        - 2. * bhspin * R * lc / AA \
        - (0.5 * np.log((1. + np.sqrt(1. + 4. * (lc*lc*SSin*SSin)
                                      * DDin / (AAin*AAin * sthin*sthin)))
                        / (SSin * DDin / AAin))
           - 0.5 * np.sqrt(1. + 4. * (lc*lc*SSin*SSin) * DDin
                           / (AAin*AAin * sthin*sthin)) - 2. * bhspin * rin * lc / AAin)
    lnh[np.where(R < rin)] = 1

    hm1 = np.exp(lnh) - 1.
    rho = np.power(hm1 * (gamma - 1.) / (kappa * gamma), 1. / (gamma - 1.))
    uu = kappa * np.power(rho, gamma) / (gamma - 1.)

    # fluid velocity
    expm2chi = SS * SS * DD / (AA * AA * sth * sth)
    up1 = np.sqrt((-1. + np.sqrt(1. + 4. * lc*lc * expm2chi)) / 2.)
    up = 2*bhspin*R*np.sqrt(1+up1*up1)/np.sqrt(AA*SS*DD)+np.sqrt(SS/AA)*up1/sth

    # more flooring
    rho[lnh < 0] = rho_floor
    rho[R < rin] = rho_floor
    uu[lnh < 0] = uu_floor
    uu[R < rin] = uu_floor

    return rho, uu, up
