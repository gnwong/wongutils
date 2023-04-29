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
    parameters gamma, rin, rmax and for black hole with spin bhspin.

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


def chakrabarti(R, H, bhspin, rin, rmax, k_adi, gamma_adi, rho_max,
                potential_r_pow, potential_rho_pow, potential_cutoff,
                potential_falloff, is_mad=True, rho_floor=1.e-7, uu_floor=1.e-7):
    """Return Chakrabarti torus profile for 2d KS R, H mesh with
    parameters pgen.

    WARNING: THIS FUNCTION DOES NOT COMPUTE THE FOUR-VELOCITY
    AND THUS RETURNS AN ARRAY OF ZEROES.

    Chakrabarti, S. 1985, ApJ 288, 1"""

    def LogHAux(bhspin, rin, rmax, cc, nn, r, sin_theta):
        """Helper function to compute log enthalpy."""
        l_ang = calculate_L(bhspin, cc, nn, r, sin_theta)
        u_t = calculate_covariant_ut(bhspin, r, sin_theta, l_ang)
        l_edge = calculate_L(bhspin, cc, nn, rin, 1.0)
        u_t_edge = calculate_covariant_ut(bhspin, rin, 1.0, l_edge)
        h = u_t_edge/u_t
        if nn == 1.0:
            h *= (l_edge/l_ang)**(cc**2/(cc**2-1.0))
        else:
            pow_c = 2.0/nn
            pow_l = 2.0-2.0/nn
            pow_abs = nn/(2.0-2.0*nn)
            h *= (abs(1.0 - cc**pow_c*l_ang**pow_l)**pow_abs
                  * abs(1.0 - cc**pow_c*l_edge**pow_l)**(-1.0*pow_abs))
        if np.isfinite(h) and h >= 1.0:
            logh = np.log(h)
        else:
            logh = -1.0
        return logh

    def calculate_cn(bhspin, rin, rmax):
        """Compute the Chakrabarti parameter and power-law index."""
        l_edge = ((rin ** 2 + bhspin ** 2 - 2.0 * bhspin * np.sqrt(rin))
                  / (np.sqrt(rin) * (rin - 2.0) + bhspin))
        l_peak = ((rmax ** 2 + bhspin ** 2 - 2.0 * bhspin * np.sqrt(rmax))
                  / (np.sqrt(rmax) * (rmax - 2.0) + bhspin))
        lambda_edge = np.sqrt((l_edge * (-2.0 * bhspin * l_edge + rin ** 3
                                         / + bhspin ** 2 * (2.0 + rin)))
                              (2.0 * bhspin + l_edge * (rin - 2.0)))
        lambda_peak = np.sqrt((l_peak * (-2.0 * bhspin * l_peak + rmax ** 3
                                         + bhspin ** 2 * (2.0 + rmax)))
                              / (2.0 * bhspin + l_peak * (rmax - 2.0)))
        nn = np.log(l_peak / l_edge) / np.log(lambda_peak / lambda_edge)
        cc = l_edge * lambda_edge ** (-nn)
        return cc, nn

    def calculate_covariant_ut(bhspin, r, sin_theta, l):
        """Compute u_t for torus in BL coordinates."""
        sigma = r**2 + bhspin**2*(1.0 - sin_theta**2)
        g_00 = -1.0 + 2.0*r/sigma
        g_03 = -2.0*bhspin*r/sigma*sin_theta**2
        g_33 = (r**2 + bhspin**2
                + 2.0*bhspin**2*r/sigma*sin_theta**2)*sin_theta**2
        u_t = -np.sqrt(max((g_03**2 - g_00*g_33)/(g_33 + 2.0*l*g_03 + l**2*g_00), 0.0))
        return u_t

    def calculate_L(bhspin, cc, nn, r, sin_theta):
        """Calculate L in Chakrabarti torus."""
        sigma = r**2 + bhspin**2 * (1.0 - sin_theta**2)
        g_00 = -1.0 + 2.0 * r / sigma
        g_03 = -2.0 * bhspin * r / sigma * sin_theta**2
        g_33 = (r**2 + bhspin**2 + 2.0*bhspin**2*r/sigma*sin_theta**2) * sin_theta**2

        # Perform bisection
        l_min = 1.0
        l_max = 100.0
        l_val = 0.5 * (l_min + l_max)
        max_iterations = 25
        tol_rel = 1.0e-8
        for n in range(max_iterations):
            error_rel = 0.5 * (l_max - l_min) / l_val
            if error_rel < tol_rel:
                break
            residual = (l_val / cc) ** (2.0 / nn) \
                + (l_val * g_33 + l_val**2 * g_03) / (g_03 + l_val * g_00)
            if residual < 0.0:
                l_min = l_val
                l_val = 0.5 * (l_min + l_max)
            elif residual > 0.0:
                l_max = l_val
                l_val = 0.5 * (l_min + l_max)
            elif residual == 0.0:
                break

        return l_val

    rho = np.zeros_like(R)
    pgas = np.zeros_like(R)
    up = np.zeros_like(R)

    Aphi = np.zeros_like(R)

    gm1 = gamma_adi - 1.

    cc, nn = calculate_cn(bhspin, rin, rmax)

    log_h_edge = LogHAux(bhspin, rin, rmax, cc, nn, rin, 1.0)
    log_h_peak = LogHAux(bhspin, rin, rmax, cc, nn, rmax, 1.0) - log_h_edge
    pgas_over_rho_peak = gm1/gamma_adi * (np.exp(log_h_peak)-1.0)
    rho_peak = np.power(pgas_over_rho_peak/k_adi, 1.0/gm1) / rho_max

    N1, N2 = R.shape
    for i in range(N1):
        for j in range(N2):
            r = R[i, j]
            theta = H[i, j]
            sin_theta = np.sin(theta)
            sin_vartheta = np.abs(sin_theta)

            if r > rin:
                log_h = LogHAux(bhspin, rin, rmax, cc, nn, r, sin_vartheta) - log_h_edge
                if log_h >= 0.:

                    pgas_over_rho = gm1 / gamma_adi * (np.exp(log_h) - 1.0)
                    rho[i, j] = np.power(pgas_over_rho / k_adi, 1.0/gm1) / rho_peak
                    pgas[i, j] = pgas_over_rho * rho[i, j]

                    if is_mad:
                        Aphi[i, j] = (max((rho[i, j]*np.power((r/rin) * sin_vartheta,
                                                              potential_r_pow)
                                           * np.exp(-r/potential_falloff)
                                           - potential_cutoff), 0.0))
                    else:
                        Aphi[i, j] = (np.power(r, potential_r_pow)
                                      * np.power(max(rho[i, j]-potential_cutoff, 0.0),
                                                 potential_rho_pow))

    uu = pgas / gm1

    return rho, uu, up, Aphi
