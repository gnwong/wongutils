import numpy as np
from scipy.special import kv
from scipy.integrate import quad


# physical constants in cgs
EE = 4.80320680e-10
CL = 2.99792458e10
ME = 9.1093826e-28


def jnu_synch(nu, Ne, Thetae, B, theta):
    """
    Fitting formula for synchrotron emissivity from Leung et al. (2011).

    :arg nu: frequency in Hz

    :arg Ne: electron number density in cm^-3

    :arg Thetae: electron temperature in units of m_e c^2

    :arg B: magnetic field in Gauss

    :arg theta: pitch angle between the line of sight and the magnetic field in radians

    :returns: synchrotron emissivity in erg s^-1 cm^-3 Hz^-1.
    """

    # lower limit in energy for the formula
    THETAE_MIN = 0.3

    if Thetae < THETAE_MIN:
        return 0.

    K2 = kv(2, 1./Thetae)

    nuc = EE * B / (2. * np.pi * ME * CL)
    sth = np.sin(theta)
    nus = (2. / 9.) * nuc * Thetae * Thetae * sth

    # upper limit in frequency for the formula
    if nu > 1.e12 * nus:
        return 0.

    x = nu / nus
    xp1 = np.power(x, 1./3)
    xx = np.sqrt(x) + np.power(2., 11./12) * np.sqrt(xp1)
    f = xx * xx

    return np.sqrt(2.) * np.pi * EE * EE * Ne * nus / (3. * CL * K2) * f * np.exp(-xp1)


def Jnu_synch(nu, Ne, Thetae, B):
    """
    Numerically integrate synchrotron emissivity over pitch angle.

    :arg nu: frequency in Hz

    :arg Ne: electron number density in cm^-3

    :arg Thetae: electron temperature in units of m_e c^2

    :arg B: magnetic field in Gauss

    :returns: frequency-integrated synchrotron emissivity in erg s^-1 cm^-3.
    """
    return quad(lambda th: jnu_synch(nu, Ne, Thetae, B, th)*np.sin(th), 0, np.pi)[0]


def __tests__():
    """
    Test synchrotron functions by comparing against known values from symphony code.
    """

    def _reldiff(a, b):
        return 2. * np.abs(a - b) / (np.abs(a) + np.abs(b))

    params = {
        (230.e9, 1.e6, 40., 10., np.pi/3.): 1.0987e-16,
        (230.e9, 1.e6, 40., 40., np.pi/3.): 5.00139e-16,
        (230.e9, 1.e6, 12., 40., np.pi/3.): 2.86537e-16
    }
    for (nu, Ne, Thetae, B, theta), val in params.items():
        assert _reldiff(jnu_synch(nu, Ne, Thetae, B, theta), val) < 1.e-2

    params = {
        (230.e9, 1.e6, 40., 10.): 1.95483e-16,
        (230.e9, 1.e6, 40., 40.): 9.03094e-16,
        (230.e9, 1.e6, 12., 40.): 5.05105e-16
    }
    for (nu, Ne, Thetae, B), val in params.items():
        assert _reldiff(Jnu_synch(nu, Ne, Thetae, B), val) < 1.e-2
