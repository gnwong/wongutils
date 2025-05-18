import pytest
import numpy as np
from wongutils.geometry.interpolators import AxisymmetricInterpolator


def grid():
    n1, n2, n3 = 4, 5, 6
    x1 = np.repeat(np.linspace(1, 2, n1)[:, None, None], n2, axis=1)
    x1 = np.repeat(x1, n3, axis=2)
    x2 = np.repeat(np.linspace(0.1, np.pi-0.1, n2)[None, :, None], n1, axis=0)
    x2 = np.repeat(x2, n3, axis=2)
    x3 = np.repeat(np.linspace(0.1, 2*np.pi-0.1, n3)[None, None, :], n1, axis=0)
    x3 = np.repeat(x3, n2, axis=1)
    return x1, x2, x3


def test_basic_interpolation():
    x1, x2, x3 = grid()
    interp = AxisymmetricInterpolator(x1, x2, x3)

    # Simple dataset: product of indices (broadcasts well)
    n1, n2, n3 = x1.shape
    data = np.zeros((n1, n2, n3))
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                data[i, j, k] = i + j + k
    interp.add_dataset(data, "test")

    # Sample at central grid point
    mid = (x1[n1//2, n2//2, 0], x2[n1//2, n2//2, 0], x3[n1//2, n2//2, n3//2])
    result = interp("test", *mid)
    assert np.isclose(result, (n1//2 + n2//2 + n3//2))


def test_extend_flags():
    x1, x2, x3 = grid()
    x1p = np.ones((1, 1, 1)) * 1.5
    x2p = np.zeros((1, 1, 1))
    x3p = np.zeros((1, 1, 1))

    data = np.ones(x1.shape)
    interp = AxisymmetricInterpolator(x1, x2, x3, extend_x2_poles=False,
                                      extend_x3_cyclic=False)
    interp.add_dataset(data, "ones")
    assert np.isnan(interp("ones", x1p, x2p, x3p))

    interp = AxisymmetricInterpolator(x1, x2, x3)
    interp.add_dataset(data, "ones")
    result = interp("ones", x1p, x2p, x3p)
    assert np.allclose(result, 1.)

    interp = AxisymmetricInterpolator(x1, x2, x3, extend_x2_poles=True,
                                      extend_x3_cyclic=False)
    interp.add_dataset(data, "ones")
    assert np.isnan(interp("ones", x1p, x2p, x3p))
    x3p = np.ones((1, 1, 1))
    assert np.allclose(interp("ones", x1p, x2p, x3p), 1.)

    interp = AxisymmetricInterpolator(x1, x2, x3, extend_x2_poles=False,
                                      extend_x3_cyclic=True)
    interp.add_dataset(data, "ones")
    x3p = np.zeros((1, 1, 1))
    assert np.isnan(interp("ones", x1p, x2p, x3p))
    x2p = np.ones((1, 1, 1))
    assert np.allclose(interp("ones", x1p, x2p, x3p), 1.)


def test_inline_dataset():
    x1, x2, x3 = grid()
    interp = AxisymmetricInterpolator(x1, x2, x3)

    n1, n2, n3 = x1.shape
    data = np.random.random((n1, n2, n3))

    # use __call__ with raw dataset, not pre-added
    x1p = x1[n1//2, n2//2, 0]
    x2p = x2[n1//2, n2//2, 0]
    x3p = x3[n1//2, n2//2, n3//2]
    with pytest.raises(KeyError):
        interp("not_a_dataset", x1p, x2p, x3p)

    assert np.allclose(interp(data, x1p, x2p, x3p), data[n1//2, n2//2, n3//2])


def test_log_scaling():
    x1, x2, x3 = grid()
    x1_log = np.logspace(0, 1, x1.shape[0])[:, None, None]
    x1_log = np.repeat(x1_log, x2.shape[1], axis=1)
    x1_log = np.repeat(x1_log, x3.shape[2], axis=2)

    interp = AxisymmetricInterpolator(x1_log, x2, x3)
    assert interp.use_log_x1


def test_bad_input():
    x1, x2, x3 = grid()

    # break symmetry across axis 2
    x1_bad = x1.copy()
    x1_bad[:, :, 1] += 1e-3

    with pytest.raises(ValueError):
        _ = AxisymmetricInterpolator(x1_bad, x2, x3)


if __name__ == "__main__":

    test_basic_interpolation()
    test_extend_flags()
    test_inline_dataset()
    test_log_scaling()
    test_bad_input()
