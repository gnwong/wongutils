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
from scipy.interpolate import griddata, RegularGridInterpolator


class AxisymmetricInterpolator:

    def __init__(self, x1, x2, x3, method_grid='linear', method_rgi='linear'):
        """
        Initialize the AxisymmetricInterpolator object by constructing both
        interpolation grid and interpolators for an axisymmetric coordinate
        system.

        :arg x1: x1 coordinate grid

        :arg x2: x2 coordinate grid

        :arg x3: x3 coordinate grid

        :arg method_grid: (default='linear') griddata method

        :arg method_rgi: (default='linear') RegularGridInterpolator method
        """

        self.datasets = {}
        n1, n2, n3 = x1.shape

        # check that x1 and x2 are symmetric across the last axis
        if not (np.all(x1 == x1[:, :, 0][:, :, None])
                and np.all(x2 == x2[:, :, 0][:, :, None])):
            raise ValueError("x1 and x2 must be symmetric across the last axis")

        x1 = x1[:, :, 0]
        x2 = x2[:, :, 0]

        # check if using log scale for the first axis is better
        lin_std = np.std(np.diff(x1[:, n2//2]))
        log_std = np.std(np.diff(np.log(x1[:, n2//2])))
        self.use_log_x1 = lin_std > log_std
        if self.use_log_x1:
            x1 = np.log(x1)

        # construct pseudocoordinates for the grid
        self.a = np.linspace(0, 2, n1)
        self.b = np.linspace(0, 1, n2)
        self.c = x3[n1//2, n2//2, :]
        A, B = np.meshgrid(self.a, self.b, indexing='ij')

        # construct sampling grid
        min_x1 = x1.min()
        max_x1 = x1.max()
        npts_n1 = 200
        min_x2 = x2.min()
        max_x2 = x2.max()
        npts_n2 = 100
        samp_x1 = np.linspace(min_x1, max_x1, npts_n1)
        samp_x2 = np.linspace(min_x2, max_x2, npts_n2)
        X1, X2 = np.meshgrid(samp_x1, samp_x2, indexing='ij')
        samp_points = (X1.ravel(), X2.ravel())

        # construct interpolation grid
        kwargs = dict(method=method_grid, fill_value=np.nan)
        x1x2 = (x1.ravel(), x2.ravel())
        a_interp_data = griddata(x1x2, A.ravel(), samp_points, **kwargs)
        b_interp_data = griddata(x1x2, B.ravel(), samp_points, **kwargs)
        a_interp_data = a_interp_data.reshape((npts_n1, npts_n2))
        b_interp_data = b_interp_data.reshape((npts_n1, npts_n2))
        # ... and construct grid interpolators
        kwargs = dict(method=method_rgi, fill_value=None, bounds_error=False)
        samp_indata = (samp_x1, samp_x2)
        self.a_interp = RegularGridInterpolator(samp_indata, a_interp_data, **kwargs)
        self.b_interp = RegularGridInterpolator(samp_indata, b_interp_data, **kwargs)

    def add_dataset(self, dataset, label):
        """
        Add a dataset to the interpolator.

        :arg dataset: dataset to be added (should be same size as original grid)

        :arg label: label for the dataset
        """
        kwargs = dict(method='linear', fill_value=np.nan, bounds_error=False)
        interp = RegularGridInterpolator((self.a, self.b, self.c), dataset, **kwargs)
        self.datasets[label] = interp

    def sample_data_at(self, label, x1, x2, x3):
        """
        Interpolate the desired dataset at the specified coordinates.

        :arg label: label for the dataset to be sampled

        :arg x1: x1 coordinate grid

        :arg x2: x2 coordinate grid

        :arg x3: x3 coordinate grid

        :returns: interpolated data at the specified coordinates
        """

        # convert to pseudocoordinates
        if self.use_log_x1:
            x1 = np.log(x1)
        A = self.a_interp((x1, x2))
        B = self.b_interp((x1, x2))

        # interpolate and return reshaped data
        data = self.datasets[label]((A.flatten(), B.flatten(), x3.flatten()))
        return data.reshape(x1.shape)
