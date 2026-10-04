"""Analytic tests for meshblock lookup and interpolation across AMR levels."""

import numpy as np
import pytest

from wongutils.grmhd.meshblocks import Meshblocks, BalancedMeshblocks


def field(x, y, z):
    return 2.0 + 3.0*x - 0.5*y + 0.25*z


def block_data(bounds, dtype=np.float64):
    """Sample an affine field at cell centers, including one ghost layer."""
    data = np.empty((len(bounds), 6, 6, 6, 2), dtype=dtype)
    for block, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(bounds):
        x = xmin + (np.arange(6) - 0.5) * (xmax-xmin)/4
        y = ymin + (np.arange(6) - 0.5) * (ymax-ymin)/4
        z = zmin + (np.arange(6) - 0.5) * (zmax-zmin)/4
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        data[block, ..., 0] = field(xx, yy, zz)
        data[block, ..., 1] = 2*field(xx, yy, zz) + 1
    return data


def test_lookup_and_interpolation_across_same_level_blocks():
    bounds = np.array([[0, 1, 0, 1, 0, 1], [1, 2, 0, 1, 0, 1]], dtype=float)
    blocks = Meshblocks(bounds, np.array([0, 0]), 4, 4, 4)
    data = block_data(bounds)
    points = np.array([[0.125, 0.375, 0.625], [0.7, 0.4, 0.6],
                       [1.0, 0.5, 0.5], [1.3, 0.6, 0.3], [-0.1, 0.5, 0.5]])
    result = blocks.interpolate_data_at(data, points)
    expected = field(*points[:-1].T)
    np.testing.assert_allclose(result[:-1, 0], expected, atol=1e-14)
    np.testing.assert_allclose(result[:-1, 1], 2*expected + 1, atol=1e-14)
    assert np.isnan(result[-1]).all()
    assert blocks.find_blocks(points[[-1]])[0] == -1
    with pytest.raises(ValueError, match='Invalid shape'):
        blocks.find_blocks(np.ones((2, 2)))


@pytest.mark.parametrize(
    'dtype, atol', [(np.float32, 2e-6), (np.float64, 1e-14)])
def test_balanced_ghost_values_at_cell_centers(dtype, atol):
    # One coarse block touches two fine blocks along x. The fine blocks
    # also share a same-level face with each other.
    bounds = np.array([[0, 1, 0, 1, 0, 1],
                       [1, 1.5, 0, 0.5, 0, 0.5],
                       [1.5, 2, 0, 0.5, 0, 0.5]], dtype=float)
    levels = np.array([0, 1, 1])
    logical = np.array([[0, 0, 0], [2, 0, 0], [3, 0, 0]])
    blocks = BalancedMeshblocks(bounds, levels, logical, 4, 4, 4)
    assert blocks.is_balanced
    data = block_data(bounds, dtype)

    for block in range(len(bounds)):
        ghost = blocks.interpolate_ghostzones(data, block)
        assert ghost is not None
        # Work out expected values in physical coordinates from the target
        # block's cell width, independent of source block selection.
        indices = blocks._ghost_local_indices
        left = bounds[block, ::2]
        width = (bounds[block, 1::2] - left)/4
        xyz = left + (indices + 0.5)*width
        expected = field(*xyz.T)
        valid = np.isfinite(ghost[:, 0])
        assert valid.any()
        np.testing.assert_allclose(ghost[valid, 0], expected[valid], atol=atol)
        np.testing.assert_allclose(ghost[valid, 1], 2*expected[valid]+1, atol=atol)

    # Both coarse/fine and fine/fine x faces must actually be represented.
    for block, face in ((0, -1), (1, 0), (1, -1), (2, 0)):
        ghost = blocks.interpolate_ghostzones(data, block)
        face_index = 0 if face == 0 else 1
        start = sum(count for _, _, _, count in blocks.ghost_faces[:face_index])
        count = blocks.ghost_faces[face_index][3]
        assert np.isfinite(ghost[start:start+count, 0]).any()


def test_unbalanced_topology_rejects_fast_path():
    bounds = np.array([[0, 1, 0, 1, 0, 1], [1, 1.25, 0, 0.25, 0, 0.25]])
    blocks = BalancedMeshblocks(bounds, np.array([0, 2]),
                                np.array([[0, 0, 0], [4, 0, 0]]), 4, 4, 4)
    assert not blocks.is_balanced
