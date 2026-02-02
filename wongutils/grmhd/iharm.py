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
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from wongutils.geometry import coordinates, metrics


def get_native_grid(fname, verbose=False, corners=True, array3d=False):
    """Get native grid underlying iharm-style snapshot file."""

    coordinate_info = get_header_coordinates(fname, verbose=verbose)

    N1 = coordinate_info['N1']
    N2 = coordinate_info['N2']
    N3 = coordinate_info['N3']

    Rin = coordinate_info['Rin']
    Rout = coordinate_info['Rout']

    x1 = np.linspace(np.log(Rin), np.log(Rout), N1+1)
    x2 = np.linspace(0., 1., N2+1)
    x3 = np.linspace(0., 2.*np.pi, N3+1)

    if not corners:
        x1 = (x1[:-1] + x1[1:]) / 2.
        x2 = (x2[:-1] + x2[1:]) / 2.
        x3 = (x3[:-1] + x3[1:]) / 2.

    if array3d:
        return np.meshgrid(x1, x2, x3, indexing='ij')

    return x1, x2, x3


def get_header_coordinates(fname, verbose=False):
    """Load coordinate information from iharm-style header."""

    with h5py.File(fname, 'r') as hfp:

        # support both 'header' and 'fluid_header' groups for image files
        header_name = 'header'
        if 'fluid_header' in hfp.keys():
            header_name = 'fluid_header'

        metric = hfp[header_name]['metric'][()].decode('utf-8').lower()
        if verbose:
            print(f" - metric coordinates are '{metric}' for {fname}")

        # create coordinate information dictionary
        coordinate_info = dict(metric=metric)

        if metric == 'mks':
            raise NotImplementedError("mks coordinates not implemented")

        elif metric in ['eks', 'mmks', 'fmks']:

            # load black hole spin
            coordinate_info['bhspin'] = hfp[header_name]['geom'][metric]['a'][()]

            # load size of coordinate grid
            coordinate_info['N1'] = hfp[header_name]['n1'][()]
            coordinate_info['N2'] = hfp[header_name]['n2'][()]
            coordinate_info['N3'] = hfp[header_name]['n3'][()]

            # load inner and outer edge of coordinate system. must support
            # the legacy names for these...
            if 'r_in' in hfp[header_name]['geom'][metric]:
                coordinate_info['Rin'] = hfp[header_name]['geom'][metric]['r_in'][()]
            else:
                coordinate_info['Rin'] = hfp[header_name]['geom'][metric]['Rin'][()]
            if 'r_out' in hfp[header_name]['geom'][metric]:
                coordinate_info['Rout'] = hfp[header_name]['geom'][metric]['r_out'][()]
            else:
                coordinate_info['Rout'] = hfp[header_name]['geom'][metric]['Rout'][()]

            # load extra parameters for mmks/fmks
            if metric in ['mmks', 'fmks']:
                hslope = hfp[header_name]['geom'][metric]['hslope'][()]
                poly_xt = hfp[header_name]['geom'][metric]['poly_xt'][()]
                poly_alpha = hfp[header_name]['geom'][metric]['poly_alpha'][()]
                mks_smooth = hfp[header_name]['geom'][metric]['mks_smooth'][()]
                poly_norm = 0.5 * np.pi
                poly_norm /= (1. + 1./(poly_alpha + 1.)*1./np.power(poly_xt, poly_alpha))
                coordinate_info['mks_smooth'] = mks_smooth
                coordinate_info['hslope'] = hslope
                coordinate_info['poly_alpha'] = poly_alpha
                coordinate_info['poly_xt'] = poly_xt
                coordinate_info['poly_norm'] = poly_norm

        else:
            raise NotImplementedError(f"unknown metric {metric}")

    return coordinate_info


def load_snapshot(fname, gcov=None, gcon=None):
    """Load fluid information from iharm-style snapshot file."""

    if gcov is None:
        coordinate_info = get_header_coordinates(fname)
        x1, x2, x3 = get_native_grid(fname, corners=False)
        X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
        gcov = metrics.get_gcov_fmks_from_fmks(coordinate_info, X1, X2, X3=X3)

    if gcon is None:
        n3 = gcov.shape[2]
        if np.allclose(gcov[:, :, 0], gcov[:, :, n3//3]) and \
           np.allclose(gcov[:, :, 0], gcov[:, :, n3//2]):
            gcov2d = gcov[:, :, 0, :, :]
            gcon2d = np.linalg.inv(gcov2d)
            gcon = np.zeros_like(gcov)
            gcon[:, :, :, :, :] = gcon2d[:, :, None, :, :]
        else:
            gcon = np.linalg.inv(gcov)

    # load fluid data from snapshot file
    hfp = h5py.File(fname, 'r')
    rho = np.array(hfp['prims'][:, :, :, 0])
    UU = np.array(hfp['prims'][:, :, :, 1])
    U = np.array(hfp['prims'][:, :, :, 2:5])
    B = np.array(hfp['prims'][:, :, :, 5:8])
    hfp.close()

    N1, N2, N3 = rho.shape

    # compute velocity four-vectors
    alpha = 1. / np.sqrt(-gcon[:, :, :, 0, 0])
    gamma = np.sqrt(1. + np.einsum('abci,abci->abc', np.einsum('abcij,abci->abcj',
                                                               gcov[:, :, :, 1:, 1:],
                                                               U), U))
    ucon = np.zeros((N1, N2, N3, 4))
    ucon[:, :, :, 1:] = -gamma[:, :, :, None]*alpha[:, :, :, None]*gcon[:, :, :, 0, 1:]
    ucon[:, :, :, 1:] += U
    ucon[:, :, :, 0] = gamma / alpha
    ucov = np.einsum('abcij,abci->abcj', gcov, ucon)

    # compute magnetic field four-vectors
    bcon = np.zeros_like(ucon)
    bcon[:, :, :, 0] = np.einsum('abci,abci->abc', B, ucov[:, :, :, 1:])
    bcon[:, :, :, 1:] = B + ucon[:, :, :, 1:] * bcon[:, :, :, 0, None]
    bcon[:, :, :, 1:] /= ucon[:, :, :, 0, None]
    bcov = np.einsum('abcij,abci->abcj', gcov, bcon)

    return (rho, UU, U, B, ucon, ucov, bcon, bcov)


class iharmSnapshot:
    def __init__(self, fname, verbose=False):
        """
        Load iharm snapshot file and return data in a dictionary.

        :arg fname: filename of snapshot file to load
        :arg verbose: (default=False) whether to print information messages
        """

        self.fname = fname

        if fname.endswith('.h5'):
            self.data = self._load_h5(fname)

        self.data.update(self._load_header(fname))
        self._coordinate_info = get_header_coordinates(fname)
        self._x2_lookup_interp = None
        self._prims_interp = None

    def __repr__(self):
        """Return a string representation of the iharmSnapshot object."""
        t = self.data.get('time', '?')
        n_step = self.data.get('n_step', '?')
        n_dump = self.data.get('n_dump', '?')
        nvars = self.data.get('nvars', '?')
        return (f"<iharmSnapshot: fname={self.fname}, time={t}, "
                f"n_step={n_step}, n_dump={n_dump}, vars={nvars}>")

    def get_primitives_at(self, R, H, P, in_coords=None):
        """
        Get primitive variables at the specified Kerr-Schild
        coordinates.

        :arg R: KS radial coordinate
        :arg H: KS polar angle theta (colatitude, 0 to pi)
        :arg P: KS azimuthal angle phi (0 to 2pi)
        :arg in_coords: (default=None) coordinate system to use for input coordinates
                       (default=None) use native coordinates
                       'ks': use Kerr-Schild coordinates

        :returns: primitive variables at the specified coordinates,
                  shape (n1, n2, n3, 8) with [rho, UU, U1, U2, U3, B1, B2, B3]
        """

        if in_coords not in ['ks']:
            raise NotImplementedError("in_coords only supported for 'ks' coordinates")

        if self._prims_interp is None:
            self._build_x2_lookup()

        nx, ny, nz = R.shape
        coord_info = self._coordinate_info
        metric = coord_info['metric']

        P = np.mod(P, 2. * np.pi)

        # KS -> native (x1, x2, x3)
        x1 = np.log(R)
        x3 = P
        if metric == 'eks':
            x2 = H / np.pi
        else:
            x2 = self._x2_lookup_interp(np.column_stack((R.ravel(), H.ravel())))
            x2 = x2.reshape(R.shape)

        points = np.column_stack((x1.ravel(), x2.ravel(), x3.ravel()))
        interpolated = self._prims_interp(points)
        return interpolated.reshape((nx, ny, nz, self.data['nvars']))

    def _load_h5(self, fname):
        """Load iharm HDF5 snapshot and return data dictionary."""
        with h5py.File(fname, 'r') as hfp:
            prims = np.array(hfp['prims'])
            t = float(hfp['t'][()])
            n_step = int(hfp['n_step'][()]) if 'n_step' in hfp else None
            n_dump = int(hfp['n_dump'][()]) if 'n_dump' in hfp else None
        N1, N2, N3, nvars = prims.shape
        return {
            'prims': prims,
            'time': t,
            'n_step': n_step,
            'n_dump': n_dump,
            'N1': N1, 'N2': N2, 'N3': N3,
            'nvars': nvars,
        }

    def _load_header(self, fname):
        """
        Load iharm HDF5 header and return a nested dictionary structure.
        """
        def _h5_to_python(obj):
            """Convert h5py to python."""
            val = obj[()]
            if hasattr(val, 'decode'):
                return val.decode('utf-8')
            if hasattr(val, 'ndim') and val.ndim == 0:
                return val.item()
            return val

        def _visit_group(group, out_dict):
            """Recursively visit HDF5 group and populate dict."""
            for key in group.keys():
                obj = group[key]
                if isinstance(obj, h5py.Dataset):
                    out_dict[key] = _h5_to_python(obj)
                elif isinstance(obj, h5py.Group):
                    out_dict[key] = {}
                    _visit_group(obj, out_dict[key])

        header_dict = {}

        with h5py.File(fname, 'r') as hfp:
            header_name = 'header'

            if header_name in hfp.keys():
                header_dict['header'] = {}
                _visit_group(hfp[header_name], header_dict['header'])

            # include top-level dump metadata
            for key in ['t', 'dt', 'n_step', 'n_dump', 'dump_cadence',
                        'full_dump_cadence', 'is_full_dump']:
                if key in hfp.keys():
                    try:
                        val = hfp[key][()]
                        if hasattr(val, 'ndim') and val.ndim == 0:
                            val = val.item()
                        header_dict[key] = val
                    except (TypeError, ValueError):
                        pass

        return header_dict

    def _build_x2_lookup(self, n_r=None, n_th=None):
        """
        Build KS to FMKS x2 lookup table for coordinate inversion.

        :arg nr: number of r points in lookup (default: 2x native N1)
        :arg nh: number of theta points in lookup (default: 2x native N2)
        """
        coord_info = self._coordinate_info
        metric = coord_info['metric']
        N1 = self.data['N1']
        N2 = self.data['N2']
        Rin = coord_info['Rin']
        Rout = coord_info['Rout']

        x1, x2, x3 = get_native_grid(self.fname, corners=False)
        prims = self.data['prims']

        # Extend grid by one zone on each side; out-of-bounds queries
        # interpolate to nearest edge instead of extrapolating
        dx1 = x1[1] - x1[0]
        dx2 = x2[1] - x2[0]
        dx3 = x3[1] - x3[0]
        x1_ext = np.concatenate([[x1[0] - dx1], x1, [x1[-1] + dx1]])
        x2_ext = np.concatenate([[x2[0] - dx2], x2, [x2[-1] + dx2]])
        x3_ext = np.concatenate([[x3[0] - dx3], x3, [x3[-1] + dx3]])

        # x1, x2: copy edge values; x3: periodic wrap
        prims_ext = np.concatenate([prims[0:1], prims, prims[-1:]], axis=0)
        prims_ext = np.concatenate([prims_ext[:, 0:1], prims_ext,
                                    prims_ext[:, -1:]], axis=1)
        prims_ext = np.concatenate([prims_ext[:, :, -1:], prims_ext,
                                    prims_ext[:, :, 0:1]], axis=2)

        self._prims_interp = RegularGridInterpolator(
            (x1_ext, x2_ext, x3_ext), prims_ext,
            method='linear', bounds_error=False, fill_value=np.nan
        )

        if metric == 'eks':
            self._x2_lookup_interp = None
            self._r_grid = None
            self._th_grid = None
            return

        if n_r is None:
            n_r = 2 * N1
        if n_th is None:
            n_th = 2 * N2

        r_grid = np.linspace(Rin, Rout, n_r)
        th_grid = np.linspace(0., np.pi, n_th)

        def theta_residual(x2, x1, th_target):
            r, h, _ = coordinates.x_ks_from_fmks(coord_info, x1, x2, 0.)
            return h - th_target

        x2_table = np.zeros((n_r, n_th))
        for i, r in enumerate(r_grid):
            x1 = np.log(r)
            for j, th in enumerate(th_grid):
                try:
                    x2_table[i, j] = brentq(
                        lambda x2, x1=x1, th_target=th: theta_residual(x2, x1, th_target),
                        1e-10, 1. - 1e-10,
                        xtol=1e-12, maxiter=100
                    )
                except ValueError:
                    x2_table[i, j] = np.nan

        self._x2_lookup_interp = RegularGridInterpolator(
            (r_grid, th_grid), x2_table,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        self._r_grid = r_grid
        self._th_grid = th_grid

    def estimate_x2_lookup_error(self):
        """
        Estimate the error in x2 found through the KS to FMKS lookup table
        at each native grid zone. Used to assess whether lookup resolution
        should be refined.

        For each zone (i, j), compute true (r, h) from (x1_i, x2_j), then
        invert via the lookup to get x2_lookup. Error = |x2_j - x2_lookup|.

        :returns: dict with
            - max_absolute_error: maximum |x2_true - x2_lookup| over all zones
            - mean_absolute_error: mean error over all zones
            - error_per_radial_zone: (N1,) array of max error in each radial zone
            - r_per_zone: (N1,) array of radial coordinate at zone center
            - nr, nh: current lookup resolution (for reference)
        """
        if self._prims_interp is None:
            self._build_x2_lookup()

        coord_info = self._coordinate_info
        metric = coord_info['metric']

        if metric == 'eks':
            return {
                'max_absolute_error': 0.0,
                'mean_absolute_error': 0.0,
                'error_per_radial_zone': np.zeros(self.data['N1']),
                'r_per_zone': np.exp(get_native_grid(self.fname, corners=False)[0]),
                'nr': None,
                'nh': None,
                'note': 'EKS has analytic inverse; no lookup error.',
            }

        x1, x2, x3 = get_native_grid(self.fname, corners=False)
        N1, N2 = len(x1), len(x2)

        # (r, h) at each zone (i, j) from FMKS to KS
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')
        r, th, _ = coordinates.x_ks_from_fmks(coord_info, X1, X2, np.zeros_like(X1))

        # x2 from lookup at (r, h)
        points = np.column_stack((r.ravel(), th.ravel()))
        x2_lookup = self._x2_lookup_interp(points).reshape(N1, N2)

        # true x2 at each zone
        x2_true = X2

        # absolute error
        abs_err = np.abs(x2_true - x2_lookup)
        abs_err = np.nan_to_num(abs_err, nan=0.0)

        # relative error
        rel_err = np.abs(x2_true - x2_lookup) / np.abs(x2_true)
        rel_err = np.nan_to_num(rel_err, nan=0.0)

        # per-radial-zone max error
        error_per_radial_zone = np.max(abs_err, axis=1)
        r_per_zone = np.exp(x1)
        error_per_radial_zone_relative = np.max(rel_err, axis=1)

        return {
            'max_absolute_error': float(np.max(abs_err)),
            'mean_absolute_error': float(np.mean(abs_err)),
            'error_per_radial_zone': error_per_radial_zone,
            'max_relative_error': float(np.max(rel_err)),
            'mean_relative_error': float(np.mean(rel_err)),
            'error_per_radial_zone_relative': error_per_radial_zone_relative,
            'r_per_zone': r_per_zone,
            'nr': len(self._r_grid),
            'nh': len(self._th_grid),
        }
