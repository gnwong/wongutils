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


def _test_input_sizes():
    """Check if the function works with both scalars and arrays."""
    bhspin = 0.42
    subkep = 0.87
    f_r = 0.81
    f_p = 0.93
    rvals = np.linspace(2., 12., 11)
    hvals = np.linspace(0.1, 2., 11)
    R, H = np.meshgrid(rvals, hvals, indexing='ij')
    ucon_bl, _ = velocities.ucon_bl_general_subkep(R, H, bhspin, subkep, f_r, f_p)
    print(ucon_bl)


if __name__ == "__main__":

    test_midplane()
    # TODO: test off midplane (still need comparison data)
    #test_input_sizes()
