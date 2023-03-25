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
                coordinate_info['Rin'] = hfp[header_name]['geom'][metric]['r_out'][()]
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
