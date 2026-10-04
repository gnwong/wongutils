"""Small, generated GRMHD files for loader and coordinate regression tests."""

import struct

import numpy as np
import pytest

from wongutils.grmhd.athenak import AthenaKSnapshot, AthenaKRestart
from wongutils.grmhd.iharm import iharmSnapshot, load_snapshot


def write_athenak_snapshot(path, blocks, variables=('dens', 'eint', 'velx',
                                                    'vely', 'velz', 'bcc1',
                                                    'bcc2', 'bcc3')):
    """Write the small subset of AthenaK format 1.1 used by the loader."""
    nx1, nx2, nx3 = 4, 2, 2
    header = (f'<mesh>\nnx1={nx1 * len(blocks)}\nnx2={nx2}\nnx3={nx3}\n'
              f'nghost=0\nx1min=0\nx1max={len(blocks)}\n'
              f'x2min=0\nx2max=1\nx3min=0\nx3max=1\n'
              f'<meshblock>\nnx1={nx1}\nnx2={nx2}\nnx3={nx3}\n').encode()
    with path.open('wb') as fp:
        fp.write(b'Athena binary output version=1.1\n')
        fp.write(b'5\n')
        fp.write(b'time=1.25\ncycle=7\nsize of location=8\nsize of variable=8\n')
        fp.write(f'number of variables={len(variables)}\n'.encode())
        fp.write(('variables: ' + ' '.join(variables) + '\n').encode())
        fp.write(f'header size={len(header)}\n'.encode())
        fp.write(header)
        for block in blocks:
            fp.write(np.array([0, nx1-1, 0, nx2-1, 0, nx3-1],
                              dtype=np.int32).tobytes())
            fp.write(np.array([block, 0, 0, 0], dtype=np.int32).tobytes())
            fp.write(np.array([block, block + 1, 0, 1, 0, 1],
                              dtype=np.float64).tobytes())
            # The file stores x1 as the fastest varying index.
            values = np.empty((len(variables), nx3, nx2, nx1))
            for v in range(len(variables)):
                for k in range(nx3):
                    for j in range(nx2):
                        for i in range(nx1):
                            values[v, k, j, i] = (1000*v + 100*block
                                                  + 10*i + 2*j + k)
            fp.write(values.tobytes())


def test_athenak_snapshot_layout_and_centers(tmp_path):
    path = tmp_path / 'tiny.bin'
    write_athenak_snapshot(path, blocks=(0, 1))
    snap = AthenaKSnapshot(str(path), populate_ghostzones=False)

    assert (snap.data['time'], snap.data['cycle'], snap.data['n_mbs']) == (1.25, 7, 2)
    assert snap.prims.shape == (2, 6, 4, 4, 8)
    for block in (0, 1):
        for i, j, k in ((0, 0, 0), (3, 1, 1), (1, 0, 1)):
            for component, source in ((0, 0), (1, 1), (2, 2), (7, 7)):
                expected = 1000*source + 100*block + 10*i + 2*j + k
                assert snap.prims[block, i+1, j+1, k+1, component] == expected

    x, y, z = snap.get_cell_centers()
    assert x.shape == y.shape == z.shape == snap.prims.shape[:-1]
    np.testing.assert_allclose(x[1, :, 1, 1], np.arange(-1, 5)/4 + 1.125)
    np.testing.assert_allclose(y[0, 1, :, 1], [-0.25, 0.25, 0.75, 1.25])
    np.testing.assert_allclose(z[0, 1, 1, :], [-0.25, 0.25, 0.75, 1.25])


def test_athenak_variable_mapping_and_bad_queries(tmp_path):
    path = tmp_path / 'tiny.bin'
    write_athenak_snapshot(path, blocks=(0,))
    snap = AthenaKSnapshot(str(path), populate_ghostzones=False,
                           variable_mapping=('bcc3', 'dens'))
    assert snap.prims.shape == (1, 6, 4, 4, 2)
    np.testing.assert_array_equal(snap.prims[0, 1:-1, 1:-1, 1:-1, 0],
                                  np.asarray(snap.data['mb_data']['bcc3'][0])
                                  .transpose(2, 1, 0))
    with pytest.raises(ValueError, match='same shape'):
        snap.get_primitives_at(np.zeros((2,)), np.zeros((1,)), np.zeros((2,)))
    with pytest.raises(ValueError, match='Unknown interpolation'):
        snap.get_primitives_at(np.array([0.5]), np.array([0.5]),
                               np.array([0.5]), interpolation='cubic')


def test_athenak_populates_shared_ghost_face(tmp_path):
    path = tmp_path / 'two_blocks.bin'
    write_athenak_snapshot(path, blocks=(0, 1))
    snap = AthenaKSnapshot(str(path), populate_ghostzones=True)
    np.testing.assert_array_equal(snap.prims[0, -1, 1:-1, 1:-1],
                                  snap.prims[1, 1, 1:-1, 1:-1])
    np.testing.assert_array_equal(snap.prims[1, 0, 1:-1, 1:-1],
                                  snap.prims[0, -2, 1:-1, 1:-1])
    np.testing.assert_array_equal(snap.prims[0, 0, 1:-1, 1:-1], 0)
    np.testing.assert_array_equal(snap.prims[1, -1, 1:-1, 1:-1], 0)


def test_athenak_rejects_bad_format(tmp_path):
    path = tmp_path / 'bad.bin'
    path.write_bytes(b'Athena binary output version=9.9\n')
    with pytest.raises(TypeError, match='unsupported file fmt version'):
        AthenaKSnapshot(str(path))


def test_athenak_restart_round_trip_with_payload_edit(tmp_path):
    path = tmp_path / 'input.rst'
    parameter_dump = (b'<mesh>\nx1=2\nnx2=2\nnx3=2\nnghost=0\n'
                      b'x1min=0\nx1max=1\nx2min=0\nx2max=1\n'
                      b'x3min=0\nx3max=1\n<meshblock>\n'
                      b'nx1=2\nnx2=2\nnx3=2\n<par_end>\n')
    mesh_size = np.array([0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5], dtype=np.float64)
    region = np.array([0, 2, 2, 2, 0, 1, 0, 1, 0, 1,
                       1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    payload = np.arange(8, dtype=np.float64)
    raw = (parameter_dump + struct.pack('@ii', 1, 0) + mesh_size.tobytes()
           + region.tobytes()*2 + np.array([1.25, 0.01]).tobytes()
           + struct.pack('@i', 7) + np.array([0, 0, 0, 0], dtype=np.int32).tobytes()
           + np.array([1.0], dtype=np.float32).tobytes()
           + struct.pack('@Q', payload.nbytes) + payload.tobytes())
    path.write_bytes(raw)

    restart = AthenaKRestart(str(path))
    assert restart.data['n_records'] == 1
    assert restart.data['ncycle'] == 7
    assert restart.data['time'] == 1.25
    assert restart.data['payload_raw'] == payload.tobytes()

    edited = payload.copy()
    edited[3] = 42.0
    restart.data['payload_records'] = (edited.tobytes(),)
    restart.data['time'] = 2.5
    output = tmp_path / 'output.rst'
    restart.write(str(output))
    reloaded = AthenaKRestart(str(output))
    assert reloaded.data['time'] == 2.5
    assert reloaded.data['payload_raw'] == edited.tobytes()
    assert reloaded.data['logical_locations'].tolist() == [[0, 0, 0, 0]]


def test_iharm_eks_grid_periodicity_and_fourvectors(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'tiny.h5'
    shape = (3, 3, 4)
    prims = np.zeros(shape + (8,))
    prims[..., 0] = 2.0
    prims[..., 1] = 0.2
    prims[..., 2:5] = [0.1, -0.2, 0.3]
    prims[..., 5:8] = [0.4, 0.5, -0.1]
    prims[..., 0] += np.arange(shape[2])[None, None, :]
    with h5py.File(path, 'w') as fp:
        fp['prims'] = prims
        fp['t'] = 1.25
        header = fp.create_group('header')
        header['metric'] = np.bytes_('eks')
        for axis, size in enumerate(shape, start=1):
            header[f'n{axis}'] = size
        geom = header.create_group('geom').create_group('eks')
        geom['a'] = 0.0
        geom['r_in'] = 2.0
        geom['r_out'] = 8.0

    snap = iharmSnapshot(str(path))
    x1 = np.linspace(np.log(2), np.log(8), shape[0]+1)
    x2 = np.linspace(0, 1, shape[1]+1)
    x3 = np.linspace(0, 2*np.pi, shape[2]+1)
    r = np.full((1, 1, 1), np.exp((x1[1] + x1[2])/2))
    h = np.full_like(r, np.pi*(x2[1] + x2[2])/2)
    p = np.full_like(r, (x3[0] + x3[1])/2)
    np.testing.assert_allclose(snap.get_primitives_at(r, h, p, in_coords='ks')[0, 0, 0],
                               prims[1, 1, 0])
    np.testing.assert_allclose(snap.get_primitives_at(r, h, p+2*np.pi,
                                                      in_coords='ks')[0, 0, 0],
                               prims[1, 1, 0])

    gcov = np.broadcast_to(np.diag([-1., 1., 1., 1.]), shape + (4, 4))
    rho, energy, u, b, ucon, ucov, bcon, bcov = load_snapshot(
        str(path), gcov=gcov, gcon=gcov)
    np.testing.assert_array_equal(rho, prims[..., 0])
    np.testing.assert_array_equal(energy, prims[..., 1])
    np.testing.assert_array_equal(u, prims[..., 2:5])
    np.testing.assert_array_equal(b, prims[..., 5:8])
    np.testing.assert_allclose(np.sum(ucon*ucov, axis=-1), -1., atol=1e-14)
    np.testing.assert_allclose(np.sum(ucon*bcov, axis=-1), 0., atol=1e-14)
    np.testing.assert_allclose(np.sum(bcon*bcov, axis=-1),
                               np.sum(bcon*bcon*np.array([-1, 1, 1, 1]), axis=-1))
