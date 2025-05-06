import numpy as np
import wongutils.grmhd.velocities as velocities


def test_midplane():
    """Reproduces original tests in kgeo for Velocity."""
    tests = {
        (1, 0.7, 0.7, 0.3, 4.0): [1.65511, -0.243725, 0, 0.141355],
        (0.3, 0.4, 0.1, 0.9375, 1.6): [14.0203, -0.934042, 0, 3.63511],
        (0.1, 0.72, 0.36, 0.1707, 3.4): [2.04774, -0.549666, 0, 0.0269644]
    }
    for (subkep, f_r, f_p, a, r), ucon in tests.items():
        ucon_bl, _ = velocities.ucon_bl_general_subkep(r, np.pi/2, a, subkep, f_r, f_p)
        assert np.allclose(ucon_bl, ucon, rtol=1.e-5), \
            f"ucon_bl {r}: {ucon_bl} != {ucon}"


def test_off_midplane():
    """Reproduces original tests in kgeo for Velocity."""
    tests = {
        (1., 1., 1., 0.22, 9., 0.314159): [1.14124, 0, 0, 0.0419266],
        (0.8, 1., 1., 0.4, 6.4, 0.314159): [1.21232, 0, 0, 0.0591806],
        (0.8, 1., 1., 0.4, 4.1, 0.314159): [1.47326, -0.224901, 0, 0.136996],
        (0.4, 0.7, 0.3, 0.7, 6., 0.314159): [1.24774, -0.172259, 0, 0.0166473],
        (0.6, 0.9, 0.83, 0.47, 5.5, 2.13628): [1.28167, -0.0603954, 0, 0.0511066],
        (0.6, 0.9, 0.83, 0.47, 1.95, 2.13628): [23.3886, -0.688401, 0, 2.96553]
    }
    for (subkep, f_r, f_p, a, r, h), ucon in tests.items():
        ucon_bl, _ = velocities.ucon_bl_general_subkep(r, h, a, subkep, f_r, f_p)
        assert np.allclose(ucon_bl, ucon, rtol=1.e-5), \
            f"ucon_bl {r}: {ucon_bl} != {ucon}"


def test_input_sizes():
    """Check if the function works with both scalars and arrays."""
    bhspin = 0.42
    subkep = 0.87
    f_r = 0.81
    f_p = 0.93
    rvals = np.linspace(2., 12., 11)
    hvals = np.linspace(0.1, 2., 11)
    R, H = np.meshgrid(rvals, hvals, indexing='ij')
    ucon_bl, ucov_bl = velocities.ucon_bl_general_subkep(R, H, bhspin, subkep, f_r, f_p)
    for ir, tr in enumerate(rvals):
        for ih, th in enumerate(hvals):
            ucon_bl_scalar, ucov_bl_scalar = \
                velocities.ucon_bl_general_subkep(tr, th, bhspin, subkep, f_r, f_p)
            assert np.allclose(ucon_bl[ir, ih], ucon_bl_scalar), \
                f"ucon_bl {tr}, {th}: {ucon_bl[ir, ih]} != {ucon_bl_scalar}"
            assert np.allclose(ucov_bl[ir, ih], ucov_bl_scalar), \
                f"ucov_bl {tr}, {th}: {ucov_bl[ir, ih]} != {ucov_bl_scalar}"


if __name__ == "__main__":

    test_midplane()
    test_off_midplane()
    test_input_sizes()
