import numpy as np
import wongutils.geometry.metrics as metrics
import wongutils.grmhd.utilities as utilities


xs = np.array([8.9469, -0.55538, 11.500, 14.237, 18.937,
               9.0760, 15.544, 17.165, -5.2547, 19.532])

ys = np.array([1.2877, 1.1463, 18.016, -3.1950, 1.0695,
               1.2334, -9.5234, -10.917, 6.9509, 16.045])

zs = np.array([16.282, -0.19605, 5.8583, -7.8569, -5.4160,
               -17.467, 5.5033, -11.992, -5.7138, -12.544])

U1 = np.array([0.37635, -0.0076638, -0.070749, 0.14415, 0.046248,
               0.32571, 0.22094, 0.16444, -0.22362, -0.091955])
U2 = np.array([0.30966, 0.0092377, 0.13793, 0.19725, 0.18228,
               0.25988, 0.14721, 0.16277, -0.040941, 0.14666])
U3 = np.array([1.1666, -0.0011124, 0.062998, -0.11527, -0.0044419,
               -1.0317, 0.035971, -0.050656, 0.028897, 0.034990])

B1 = np.array([0.031852, -0.32640, 0.033849, -0.016020, 0.022993,
               -0.033946, -0.010617, 0.015730, -0.041921, 0.017274])
B2 = np.array([-0.070350, -2.6024, -0.015772, 0.067947, -0.016045,
               0.064289, -0.055261, 0.042666, -0.085247, -0.016336])
B3 = np.array([0.055585, 1.3215, -0.011489, 0.022757, -0.0021396,
               0.048842, 0.021226, 0.0095743, 0.088001, -0.014819])
bsq = np.array([4.3944e-03, 8.0814, 1.3866e-03, 4.6911e-03, 7.4098e-04,
                4.0791e-03, 3.1920e-03, 1.9705e-03, 1.4765e-02, 7.3543e-04])

xs = xs.reshape(-1, 1, 1)
ys = ys.reshape(-1, 1, 1)
zs = zs.reshape(-1, 1, 1)

U1 = U1.reshape(-1, 1, 1)
U2 = U2.reshape(-1, 1, 1)
U3 = U3.reshape(-1, 1, 1)

B1 = B1.reshape(-1, 1, 1)
B2 = B2.reshape(-1, 1, 1)
B3 = B3.reshape(-1, 1, 1)


def test_prims_to_fourvectors():
    bhspin = 0.9
    gcon_cks = metrics.get_gcon_cks_from_cks(bhspin, xs, ys, zs)
    gcov_cks = metrics.get_gcov_cks_from_cks(bhspin, xs, ys, zs)
    ucon_cks, ucov_cks = utilities.U123_to_ucon(U1, U2, U3, gcon_cks, gcov_cks)
    usq = np.einsum('...i,...i->...', ucon_cks, ucov_cks).flatten()
    assert np.allclose(usq, -1.), \
        f"usq: {usq} != -1.0"
    bcon_cks, bcov_cks = utilities.B123_to_bcon(B1, B2, B3, ucon_cks, gcov_cks)
    bsq_test = np.einsum('...i,...i->...', bcon_cks, bcov_cks).flatten()
    assert np.allclose(bsq, bsq_test, rtol=1.e-3), \
        f"bsq: {bsq} != {bsq_test}"


def test_fourvectors_to_prims():
    bhspin = 0.9
    gcon_cks = metrics.get_gcon_cks_from_cks(bhspin, xs, ys, zs)
    gcov_cks = metrics.get_gcov_cks_from_cks(bhspin, xs, ys, zs)
    ucon_cks, _ = utilities.U123_to_ucon(U1, U2, U3, gcon_cks, gcov_cks)
    bcon_cks, _ = utilities.B123_to_bcon(B1, B2, B3, ucon_cks, gcov_cks)
    U1_test, U2_test, U3_test = utilities.ucon_to_U123(ucon_cks, gcon_cks)
    B1_test, B2_test, B3_test = utilities.bcon_to_B123(bcon_cks, ucon_cks)
    assert np.allclose(U1, U1_test), \
        f"U1: {U1} != {U1_test}"
    assert np.allclose(U2, U2_test), \
        f"U2: {U2} != {U2_test}"
    assert np.allclose(U3, U3_test), \
        f"U3: {U3} != {U3_test}"
    assert np.allclose(B1, B1_test), \
        f"B1: {B1} != {B1_test}"
    assert np.allclose(B2, B2_test), \
        f"B2: {B2} != {B2_test}"
    assert np.allclose(B3, B3_test), \
        f"B3: {B3} != {B3_test}"


if __name__ == "__main__":

    test_prims_to_fourvectors()
    test_fourvectors_to_prims()
