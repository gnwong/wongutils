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

from scipy.interpolate import RegularGridInterpolator
import numpy as np
import h5py

from wongutils.geometry import metrics


def get_from_header(header, blockname, keyname):
    """Return variable stored in header under blockname/keyname."""
    blockname = blockname.strip()
    keyname = keyname.strip()
    if not blockname.startswith('<'):
        blockname = '<' + blockname
    if blockname[-1] != '>':
        blockname += '>'
    block = '<none>'
    for line in [entry for entry in header]:
        if line.startswith('<'):
            block = line
            continue
        key, value = line.split('=')
        if block == blockname and key.strip() == keyname:
            return value
    raise KeyError(f'no parameter called {blockname}/{keyname}')


def load_values_at(fname, X, Y, Z, value_list):
    """Return array of values loaded from `fname' at Cartesian grid points X, Y, Z."""

    value_list = [v.strip().lower() for v in value_list if len(v.strip()) > 0]
    nvalues = len(value_list)
    n1, n2, n3 = X.shape

    populated = np.zeros((n1, n2, n3))
    values = np.zeros((nvalues, n1, n2, n3))

    data = load_athdf(fname)
    nprim, nmb, mbn3, mbn2, mbn1 = data['uov'].shape

    for mbi in range(nmb):

        mb_x1min = data['x1f'][mbi].min()
        mb_x1max = data['x1f'][mbi].max()
        mb_x2min = data['x2f'][mbi].min()
        mb_x2max = data['x2f'][mbi].max()
        mb_x3min = data['x3f'][mbi].min()
        mb_x3max = data['x3f'][mbi].max()

        mb_mask = (mb_x1min < X) & (X <= mb_x1max)
        mb_mask &= (mb_x2min < Y) & (Y <= mb_x2max)
        mb_mask &= (mb_x3min < Z) & (Z <= mb_x3max)

        # don't process meshblocks that don't contribute to the domain
        if np.count_nonzero(mb_mask) == 0:
            continue
        if np.sum(populated[mb_mask]) > 0:
            print("! we seem to have encountered overlapping zones")
        populated[mb_mask] += 1

        # get location of meshblock cell centers
        x1v = data['x1v'][mbi]
        x2v = data['x2v'][mbi]
        x3v = data['x3v'][mbi]

        # get metric information at cell centers
        if any(i in ['bsq', 'b.b'] for i in value_list):

            mbX_cks, mbY_cks, mbZ_cks = np.meshgrid(x1v, x2v, x3v, indexing='ij')

            mbgcov_cks = metrics.get_gcov_cks_from_cks(data['bhspin'], mbX_cks,
                                                       mbY_cks, mbZ_cks)
            mbgcon_cks = metrics.get_gcon_cks_from_cks(data['bhspin'], mbX_cks,
                                                       mbY_cks, mbZ_cks)

            mbgcov_cks = mbgcov_cks.transpose((0, 1, 4, 3, 2))
            mbgcon_cks = mbgcon_cks.transpose((0, 1, 4, 3, 2))

            # construct u^mu
            mbUprims_cks = data['uov'][1:4, mbi]
            alpha = 1. / np.sqrt(-mbgcon_cks[0, 0, :, :, :])  # ij c b a
            gamma = np.sqrt(1. + np.einsum('jcba,jcba->cba',
                                           np.einsum('ijcba,icba->jcba',
                                                     mbgcov_cks[1:, 1:],
                                                     mbUprims_cks),
                                           mbUprims_cks))
            ucon_cks = np.zeros((4, mbn3, mbn2, mbn1))
            ucon_cks[1:] = mbUprims_cks
            ucon_cks[1:] -= gamma[None, :, :, :]*alpha[None, :, :, :] * mbgcov_cks[0, 1:]
            ucon_cks[0] = gamma / alpha
            ucov_cks = np.einsum('ijcba,icba->jcba', mbgcov_cks, ucon_cks)
            # usq = np.einsum('icba,icba->cba', ucon_cks, ucov_cks)  # should be -1
            # print(np.allclose(usq, -1.))

            # construct b^mu
            mbBprims_cks = data['B'][:, mbi]
            bcon_cks = np.zeros_like(ucon_cks)
            bcon_cks[0] = np.einsum('icba,icba->cba', mbBprims_cks, ucov_cks[1:])
            bcon_cks[1:] = (mbBprims_cks + ucon_cks[1:] * bcon_cks[0, None, :, :, :])
            bcon_cks[1:] /= ucon_cks[0, None, :, :, :]
            bcov_cks = np.einsum('ijcba,icba->jcba', mbgcov_cks, bcon_cks)
            bsq = np.einsum('icba,icba->cba', bcon_cks, bcov_cks)
            # bdotu = np.einsum('icba,icba->cba', bcon_cks, ucov_cks)  # should be 0
            # print(np.allclose(bdotu, 0.))

        for vi, value in enumerate(value_list):

            # density
            if value in ['rho', 'density', 'dens']:
                data_to_interpolate = data['uov'][0, mbi].transpose((2, 1, 0))

            # internal energy
            elif value in ['internal energy', 'u', 'uint']:
                data_to_interpolate = data['uov'][4, mbi].transpose((2, 1, 0))

            # B primitives
            elif value == 'b1':
                data_to_interpolate = data['B'][0, mbi].transpose((2, 1, 0))
            elif value == 'b2':
                data_to_interpolate = data['B'][1, mbi].transpose((2, 1, 0))
            elif value == 'b3':
                data_to_interpolate = data['B'][2, mbi].transpose((2, 1, 0))

            elif value in ['bsq', 'b.b']:
                data_to_interpolate = bsq.transpose((2, 1, 0))

            # fill values array
            rgi = RegularGridInterpolator((x1v, x2v, x3v), data_to_interpolate,
                                          method='linear', bounds_error=False,
                                          fill_value=None)
            values[vi, mb_mask] = rgi((X[mb_mask], Y[mb_mask], Z[mb_mask]))

    n_populated = populated.sum()
    n_total = n1 * n2 * n3
    if n_populated != n_total:
        print(f"! unable to fill all requested zones ({n_populated} of {n_total})")

    return values


def load_athdf(fname):
    """Return dictionary of variables from athdf file."""

    data = {}

    hfp = h5py.File(fname, 'r')
    data['time'] = hfp.attrs['Time']
    data['header'] = hfp.attrs['Header']
    data['x1v'] = np.array(hfp['x1v'])
    data['x2v'] = np.array(hfp['x2v'])
    data['x3v'] = np.array(hfp['x3v'])
    data['x1f'] = np.array(hfp['x1f'])
    data['x2f'] = np.array(hfp['x2f'])
    data['x3f'] = np.array(hfp['x3f'])
    data['uov'] = np.array(hfp['uov'])
    data['B'] = np.array(hfp['B'])
    data['LogicalLocations'] = np.array(hfp['LogicalLocations'])
    data['Levels'] = np.array(hfp['Levels'])
    data['variable_names'] = np.array(hfp.attrs['VariableNames'])
    hfp.close()

    data['adiabatic_gamma'] = float(get_from_header(data['header'], 'mhd', 'gamma'))
    data['bhspin'] = float(get_from_header(data['header'], 'coord', 'a'))

    return data
