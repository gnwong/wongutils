import numpy as np
import wongutils.geometry.coordinates as coordinates
import wongutils.geometry.metrics as metrics
import os


def test_metrics():
    """Regression test to compare gcov in various
    coordinates against known values."""

    saved_metrics_file = os.path.dirname(__file__) + '/data/metric_tests.npy'
    saved_metrics = np.load(saved_metrics_file, allow_pickle=True).item()

    # test gcov_bl against known values
    gcov_bl_tests = saved_metrics['gcov_bl']
    for test in gcov_bl_tests:
        bhspin, r_bl, h_bl = test
        gcov_bl = metrics.get_gcov_bl_from_bl(bhspin, r_bl, h_bl)
        gcov_bl_known = gcov_bl_tests[test]
        assert np.allclose(gcov_bl, gcov_bl_known), \
            f"gcov_bl test failed for {test}: {gcov_bl} != {gcov_bl_known}"
        gcon_bl = metrics.get_gcon_bl_from_bl(bhspin, r_bl, h_bl)
        gginv = np.einsum('...ij,...jk->...ik', gcov_bl, gcon_bl)
        assert np.allclose(gginv, np.eye(4)), \
            f"gcov_bl test failed for {test}: {gginv} != I"

    # test gcov_ks against known values
    gcov_ks_tests = saved_metrics['gcov_ks']
    for test in gcov_ks_tests:
        bhspin, r_ks, h_ks = test
        gcov_ks = metrics.get_gcov_ks_from_ks(bhspin, r_ks, h_ks)
        gcov_ks_known = gcov_ks_tests[test]
        assert np.allclose(gcov_ks, gcov_ks_known), \
            f"gcov_ks test failed for {test}: {gcov_ks} != {gcov_ks_known}"

    # test gcov_cks against known values
    gcov_cks_tests = saved_metrics['gcov_cks']
    for test in gcov_cks_tests:
        bhspin, x1_cks, x2_cks, x3_cks = test
        gcov_cks = metrics.get_gcov_cks_from_cks(bhspin, x1_cks, x2_cks, x3_cks)
        gcov_cks_known = gcov_cks_tests[test]
        assert np.allclose(gcov_cks, gcov_cks_known), \
            f"gcov_cks test failed for {test}: {gcov_cks} != {gcov_cks_known}"
        gcon_cks = metrics.get_gcon_cks_from_cks(bhspin, x1_cks, x2_cks, x3_cks)
        gginv = np.einsum('...ij,...jk->...ik', gcov_cks, gcon_cks)
        assert np.allclose(gginv, np.eye(4)), \
            f"gcov_cks test failed for {test}: {gginv} != I"

    # TODO: eks tests
        # get_gcov_eks_from_ks
        # get_gcov_eks_from_eks

    # test gcov_cks from fmks using known values
    gcov_fmks_tests = saved_metrics['gcov_fmks']
    for test in gcov_fmks_tests:
        x1_fmks, x2_fmks = test
        coordinate_info = gcov_fmks_tests[test][0]
        gcov_fmks_known = gcov_fmks_tests[test][1]
        gcov_fmks = metrics.get_gcov_fmks_from_fmks(coordinate_info, x1_fmks, x2_fmks)
        assert np.allclose(gcov_fmks, gcov_fmks_known), \
            f"gcov_fmks test failed for {test}: {gcov_fmks} != {gcov_fmks_known}"


def test_input_sizes():
    """Regression test to compare gcov in various
    coordinates against known values."""

    # metric functions should be able to accept scalars or 2d arrays
    bhspin = 0.42
    reh = 1. + np.sqrt(1. - bhspin*bhspin)

    # use range of values to test
    rvals = np.linspace(1.95, 3., 11)
    hvals = np.linspace(0.1, 2., 11)
    RR, HH = np.meshgrid(rvals, hvals, indexing='ij')

    gcov_bl = metrics.get_gcov_bl_from_bl(bhspin, RR, HH)
    gcon_bl = metrics.get_gcon_bl_from_bl(bhspin, RR, HH)
    gcov_ks = metrics.get_gcov_ks_from_ks(bhspin, RR, HH)

    for ir, r in enumerate(rvals):
        for ih, h in enumerate(hvals):
            gcov_bl_test = metrics.get_gcov_bl_from_bl(bhspin, r, h)
            gcon_bl_test = metrics.get_gcon_bl_from_bl(bhspin, r, h)
            gcov_ks_test = metrics.get_gcov_ks_from_ks(bhspin, r, h)
            assert np.allclose(gcov_bl[ir, ih], gcov_bl_test), \
                f"gcov_bl {r},{h}: {gcov_bl[ir, ih]} != {gcov_bl_test}"
            assert np.allclose(gcon_bl[ir, ih], gcon_bl_test), \
                f"gcon_bl {r},{h}: {gcon_bl[ir, ih]} != {gcon_bl_test}"
            assert np.allclose(gcov_ks[ir, ih], gcov_ks_test), \
                f"gcov_ks {r},{h}: {gcov_ks[ir, ih]} != {gcov_ks_test}"

    # tests for cks
    x1vals = np.linspace(-5., 5., 6)
    x2vals = np.linspace(-5., 5., 6)
    x3vals = np.linspace(-5., 5., 6)
    X1, X2, X3 = np.meshgrid(x1vals, x2vals, x3vals, indexing='ij')

    gcov_cks = metrics.get_gcov_cks_from_cks(bhspin, X1, X2, X3)
    gcon_cks = metrics.get_gcon_cks_from_cks(bhspin, X1, X2, X3)

    for i, x1 in enumerate(x1vals):
        for j, x2 in enumerate(x2vals):
            for k, x3 in enumerate(x3vals):
                r, _, _ = coordinates.x_ks_from_cks(dict(bhspin=bhspin), x1, x2, x3)
                if r < reh:
                    continue
                gcov_cks_test = metrics.get_gcov_cks_from_cks(bhspin, x1, x2, x3)
                gcon_cks_test = metrics.get_gcon_cks_from_cks(bhspin, x1, x2, x3)
                assert np.allclose(gcov_cks[i, j, k], gcov_cks_test), \
                    f"gcov_cks {x1},{x2},{x3}: {gcov_cks[i, j, k]} != {gcov_cks_test}"
                assert np.allclose(gcon_cks[i, j, k], gcon_cks_test), \
                    f"gcon_cks {x1},{x2},{x3}: {gcon_cks[i, j, k]} != {gcon_cks_test}"

    # tests for fmks
    coordinate_info = {
        'bhspin': bhspin,
        'Rin': 1.2175642950007606,
        'Rout': 1000.0,
        'mks_smooth': 0.5,
        'hslope': 0.3,
        'poly_alpha': 14.0,
        'poly_xt': 0.82,
        'poly_norm': 0.7578173169894967
    }
    x1_fmks = np.linspace(0.75, 5., 6)
    x2_fmks = np.linspace(0.019, 0.93, 6)
    X1_fmks, X2_fmks = np.meshgrid(x1_fmks, x2_fmks, indexing='ij')

    gcov_fmks = metrics.get_gcov_fmks_from_fmks(coordinate_info, X1_fmks, X2_fmks)

    for i, x1 in enumerate(x1_fmks):
        for j, x2 in enumerate(x2_fmks):
            r, h, p = coordinates.x_ks_from_fmks(coordinate_info, x1, x2, 0.)
            if r < reh:
                continue
            gcov_fmks_test = metrics.get_gcov_fmks_from_fmks(coordinate_info, x1, x2)
            assert np.allclose(gcov_fmks[i, j], gcov_fmks_test), \
                f"gcov_fmks {x1},{x2}: {gcov_fmks[i, j]} != {gcov_fmks_test}"

    # TODO: eks tests
        # get_gcov_eks_from_ks
        # get_gcov_eks_from_eks


def test_coordinate_transforms():
    """Regression and unit tests for coordinate transforms."""

    r_ks = 5.
    h_ks = np.pi/1.2
    p_ks = 0.3

    # check ks -> eks and inverse
    x1_eks, x2_eks, x3_eks = coordinates.x_eks_from_ks(r_ks, h_ks, p_ks)
    x_eks_truth = [np.log(r_ks), h_ks/np.pi, p_ks]
    assert np.allclose(x_eks_truth, [x1_eks, x2_eks, x3_eks]), \
        f"KS->EKS fail: {x_eks_truth} -> {x1_eks}, {x2_eks}, {x3_eks}"
    rp_ks, hp_ks, pp_ks = coordinates.x_ks_from_eks(x1_eks, x2_eks, x3_eks)
    assert np.allclose([r_ks, h_ks, p_ks], [rp_ks, hp_ks, pp_ks]), \
        f"EKS->KS fail: {r_ks}, {h_ks}, {p_ks} -> {rp_ks}, {hp_ks}, {pp_ks}"

    # check ks -> cks and inverse
    x_cks, y_cks, z_cks = coordinates.x_cks_from_ks(dict(bhspin=0.), r_ks, h_ks, p_ks)
    assert np.allclose([x_cks, y_cks, z_cks], [2.38834122, 0.73880052, -4.33012702]), \
        f"KS->CKS fail: {x_cks}, {y_cks}, {z_cks} != 2.388341, 0.7388005, -4.330127"
    rp_ks, hp_ks, pp_ks = coordinates.x_ks_from_cks(dict(bhspin=0.), x_cks, y_cks, z_cks)
    assert np.allclose([r_ks, h_ks, p_ks], [rp_ks, hp_ks, pp_ks]), \
        f"CKS->KS fail: {r_ks}, {h_ks}, {p_ks} -> {rp_ks}, {hp_ks}, {pp_ks}"
    # now check handedness (increasing phi goes counter-clockwise)
    rs = np.ones(3) * 5.
    hs = np.ones(3) * np.pi/2.
    ps = np.array([0., np.pi/2., np.pi])
    x_cks, y_cks, z_cks = coordinates.x_cks_from_ks(dict(bhspin=0.), rs, hs, ps)
    x_cks_truth = np.array([5., 0., -5])
    y_cks_truth = np.array([0., 5., 0.])
    z_cks_truth = np.array([0., 0., 0.])
    P_cks = np.array([x_cks, y_cks, z_cks])
    P_cks_truth = np.array([x_cks_truth, y_cks_truth, z_cks_truth])
    assert np.allclose(P_cks, P_cks_truth), \
        f"CKS->KS fail (handedness): {P_cks} != {P_cks_truth}"

    # check fmks -> ks
    coord_info = {
        'Rin': 1.2175642950007606,
        'Rout': 1000.0,
        'mks_smooth': 0.5,
        'hslope': 0.3,
        'poly_alpha': 14.0,
        'poly_xt': 0.82
    }
    coord_info['poly_norm'] = coordinates._fmks_compute_poly_norm(coord_info)
    assert np.allclose(coord_info['poly_norm'], 0.7578173169894967), \
        f"Failure computing FMKS 'poly_norm': {coordinates['poly_norm']} != 0.757817317"
    x1_fmks = 0.3978299442843154
    x2_fmks = 0.06510416666666666
    x3_fmks = 0.4417864669110646

    r_ks, h_ks, p_ks = coordinates.x_ks_from_fmks(coord_info, x1_fmks, x2_fmks, x3_fmks)
    assert np.allclose([r_ks, h_ks, p_ks], [1.488590865, 0.7666458987, 0.4417864669]), \
        f"FMKS->KS fail: {r_ks}, {h_ks}, {p_ks} != 1.488591, 0.7666459, 0.4417865"


def test_dxdx_metrics():
    """Unit tests to ensure dxdX functions correctly translate between
    metrics in different coordinate systems."""

    bhspin = 0.2

    # coordinates
    X_cks = np.array([4.]).reshape((1, 1, 1))
    Y_cks = np.array([0.]).reshape((1, 1, 1))
    Z_cks = np.array([0.]).reshape((1, 1, 1))
    R_ks, H_ks, P_ks = coordinates.x_ks_from_cks(dict(bhspin=bhspin), X_cks, Y_cks, Z_cks)
    R_bl = R_ks
    H_bl = H_ks
    P_bl = np.zeros_like(P_ks)  # metric is axisymetric so this doesn't matter

    # get known metrics
    gcon_cks = metrics.get_gcon_cks_from_cks(bhspin, X_cks, Y_cks, Z_cks)
    gcov_ks = metrics.get_gcov_ks_from_ks(bhspin, R_ks, H_ks, P_ks)
    gcon_ks = np.linalg.inv(gcov_ks)
    gcon_bl = metrics.get_gcon_bl_from_bl(bhspin, R_bl, H_bl, P_bl)

    # squeeze
    gcon_cks = np.squeeze(gcon_cks)
    gcon_ks = np.squeeze(gcon_ks)
    gcon_bl = np.squeeze(gcon_bl)

    # test dxdX functions
    dxdX_ks_cks = coordinates.get_dxdX_ks_cks_from_cks(bhspin, X_cks, Y_cks, Z_cks)
    gcon_ks_test = np.einsum('...ij,...jk->...ik', dxdX_ks_cks, gcon_cks)
    gcon_ks_test = np.einsum('...ij,...kj->...ki', dxdX_ks_cks, gcon_ks_test)
    assert np.allclose(gcon_ks_test, gcon_ks), \
        f"gcon cks -> ks test failed: {gcon_ks_test} != {gcon_ks}"

    dxdX_ks_bl = coordinates.get_dxdX_ks_bl_from_ks(bhspin, R_ks, H_ks)
    gcon_ks_test = np.einsum('...ij,...jk->...ik', dxdX_ks_bl, gcon_bl)
    gcon_ks_test = np.einsum('...ij,...kj->...ki', dxdX_ks_bl, gcon_ks_test)
    assert np.allclose(gcon_ks_test, gcon_ks), \
        f"gcon bl -> ks test failed: {gcon_ks_test} != {gcon_ks}"

    # test inverse dxdX functions
    dxdX_bl_ks = np.linalg.inv(dxdX_ks_bl)
    gcon_bl_test = np.einsum('...ij,...jk->...ik', dxdX_bl_ks, gcon_ks)
    gcon_bl_test = np.einsum('...ij,...kj->...ki', dxdX_bl_ks, gcon_bl_test)
    assert np.allclose(gcon_bl_test, gcon_bl), \
        f"gcon ks -> bl test failed: {gcon_bl_test} != {gcon_bl}"

    dxdX_cks_ks = np.linalg.inv(dxdX_ks_cks)
    gcon_cks_test = np.einsum('...ij,...jk->...ik', dxdX_cks_ks, gcon_ks)
    gcon_cks_test = np.einsum('...ij,...kj->...ki', dxdX_cks_ks, gcon_cks_test)
    assert np.allclose(gcon_cks_test, gcon_cks), \
        f"gcon ks -> cks test failed: {gcon_cks_test} != {gcon_cks}"


if __name__ == "__main__":

    test_coordinate_transforms()
    test_metrics()
    test_dxdx_metrics()
    test_input_sizes()
