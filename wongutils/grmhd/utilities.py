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


def U123_to_ucon(U1, U2, U3, gcon, gcov):
    """
    Convert primitive variables for velocity to four-velocity components.

    :arg U1: U1 primitive over grid
    :arg U2: U2 primitive over grid
    :arg U3: U3 primitive over grid
    :arg gcon: contravariant metric tensor with shape (*U1.shape, 4, 4)
    :arg gcov: covariant metric tensor with shape (*U1.shape, 4, 4)

    :returns: ucon, ucov: four-velocity and covariant four-velocity
    """
    U = np.stack((U1, U2, U3), axis=-1)
    alpha = 1. / np.sqrt(-gcon[..., 0, 0])
    gamma = np.sqrt(1. + np.einsum('...i,...i->...', np.einsum('...ij,...i->...j',
                                                               gcov[..., 1:, 1:],
                                                               U), U))
    ucon = np.zeros((*U1.shape, 4))
    ucon[..., 1:] = -gamma[..., None]*alpha[..., None]*gcon[..., 0, 1:]
    ucon[..., 1:] += U
    ucon[..., 0] = gamma / alpha
    ucov = np.einsum('...ij,...i->...j', gcov, ucon)

    return ucon, ucov


def B123_to_bcon(B1, B2, B3, ucon, gcov):
    """
    Convert primitive variables for magnetic field to four-magnetic field
    given four-velocity.

    :arg B1: B1 primitive over grid
    :arg B2: B2 primitive over grid
    :arg B3: B3 primitive over grid
    :arg ucon: four-velocity over grid with shape (*B1.shape, 4)
    :arg gcov: covariant metric tensor with shape (*B1.shape, 4, 4)

    :returns: bcon, bcov: four-magnetic field and covariant magnetic inductiln vector
    """
    B = np.stack((B1, B2, B3), axis=-1)
    bcon = np.zeros_like(ucon)
    bcon[..., 0] = np.einsum('...i,...i->...', B, ucon[..., 1:])
    bcon[..., 1:] = B + ucon[..., 1:] * bcon[..., 0, None]
    bcon[..., 1:] /= ucon[..., 0, None]
    bcov = np.einsum('...ij,...i->...j', gcov, bcon)
    return bcon, bcov


def ucon_to_U123(ucon, gcon):
    """
    Convert four-velocity components to primitive variables for velocity.

    :arg ucon: four-velocity over grid with shape (*U1.shape, 4)
    :arg gcon: contravariant metric tensor with shape (*U1.shape, 4, 4)

    :returns: U1, U2, U3: primitive variables for velocity
    """
    U1 = ucon[..., 1] - ucon[..., 0] * gcon[..., 0, 1] / gcon[..., 0, 0]
    U2 = ucon[..., 2] - ucon[..., 0] * gcon[..., 0, 2] / gcon[..., 0, 0]
    U3 = ucon[..., 3] - ucon[..., 0] * gcon[..., 0, 3] / gcon[..., 0, 0]
    return U1, U2, U3


def bcon_to_B123(bcon, ucon):
    """
    Convert four-magnetic field components to primitive variables for
    magnetic field given four-velocity.

    :arg bcon: four-magnetic field over grid with shape (*B1.shape, 4)
    :arg ucon: four-velocity over grid with shape (*B1.shape, 4)

    :returns: B1, B2, B3: primitive variables for magnetic field
    """
    B1 = bcon[..., 1] + ucon[..., 0] * bcon[..., 0] / ucon[..., 0]
    B2 = bcon[..., 2] + ucon[..., 0] * bcon[..., 0] / ucon[..., 0]
    B3 = bcon[..., 3] + ucon[..., 0] * bcon[..., 0] / ucon[..., 0]
    return B1, B2, B3
