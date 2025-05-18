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
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator as RGI


class AxisymmetricInterpolator:

    def __init__(self, x1, x2, x3, method_grid='linear', method_rgi='linear',
                 extend_x2_poles=True,
                 extend_x3_cyclic=True):
        """
        Initialize the AxisymmetricInterpolator object by constructing both
        interpolation grid and interpolators for an axisymmetric coordinate
        system.

        :arg x1: x1 coordinate grid
        :arg x2: x2 coordinate grid
        :arg x3: x3 coordinate grid
        :arg method_grid: (default='linear') griddata method
        :arg method_rgi: (default='linear') RegularGridInterpolator method
        :arg extend_x2_poles: (default=True) extend the x2 coordinate to the poles
        :arg extend_x3_cyclic: (default=True) extend the x3 coordinate with ghost zones
        """
        # set parameters
        self.datasets = {}
        self.method_grid = method_grid
        self.method_rgi = method_rgi
        n1, n2, _ = x1.shape

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

        # get heuristics for how many points will be needed for the sampling grid
        min_x1 = x1.min()
        max_x1 = x1.max()
        min_dx1 = np.min(np.diff(x1, axis=0))
        min_x2 = x2.min()
        max_x2 = x2.max()
        min_dx2 = np.min(np.diff(x2, axis=1))
        npts_n1 = int(1.5 * np.ceil((max_x1 - min_x1) / min_dx1))
        npts_n2 = int(1.5 * np.ceil((max_x2 - min_x2) / min_dx2))
        if npts_n1 < 2 or npts_n2 < 2:
            raise ValueError("x1 and x2 must have at least 2 points")
        if npts_n1 > 10000 or npts_n2 > 10000:
            print("Warning: npts_n1 and npts_n2 are large...")

        # extend x2 to the poles if desired
        self.extend_x2_poles = extend_x2_poles
        if self.extend_x2_poles:
            # estimate if x2 should extend between 0 -> pi or 0 -> 1
            upper = np.pi if x2.max() > 1 else 1
            x2 = np.concatenate([np.zeros((n1, 1)), x2, upper * np.ones((n1, 1))], axis=1)
            # extend x1 to match shape
            x1 = np.concatenate([x1[:, [0]], x1, x1[:, [-1]]], axis=1)
            n2 += 2

        # construct x3, extending with ghost zones if desired
        self.c = x3[n1//2, n2//2]
        self.extend_x3_cyclic = extend_x3_cyclic
        if self.extend_x3_cyclic:
            dx3 = self.c[1] - self.c[0]
            self.c = np.r_[self.c[0] - dx3, self.c, self.c[-1] + dx3]

        # construct pseudocoordinates for the grid
        self.a = np.linspace(0, 1, n1)
        self.b = np.linspace(0, 1, n2)
        A, B = np.meshgrid(self.a, self.b, indexing='ij')

        # construct sampling grid
        samp_x1 = np.linspace(min_x1, max_x1, npts_n1)
        samp_x2 = np.linspace(min_x2, max_x2, npts_n2)
        X1, X2 = np.meshgrid(samp_x1, samp_x2, indexing='ij')
        samp_points = np.column_stack((X1.ravel(), X2.ravel()))

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
        self.a_interp = RGI(samp_indata, a_interp_data, **kwargs)
        self.b_interp = RGI(samp_indata, b_interp_data, **kwargs)

    def _extend(self, data):
        """Private utility method to extend a dataset along desired dimensions."""
        if self.extend_x2_poles:
            data = np.concatenate((data[:, 0][:, None, :],
                                   data,
                                   data[:, -1][:, None, :]), axis=1)
        if self.extend_x3_cyclic:
            data = np.concatenate((data[:, :, -1][:, :, None],
                                   data,
                                   data[:, :, 0][:, :, None]), axis=2)
        return data

    def add_dataset(self, dataset, label):
        """
        Add a dataset to the interpolator.

        :arg dataset: dataset to be added (should be same size as original grid)
        :arg label: label for the dataset
        """
        kwargs = dict(method=self.method_rgi, fill_value=np.nan, bounds_error=False)
        interp = RGI((self.a, self.b, self.c), self._extend(dataset), **kwargs)
        self.datasets[label] = interp

    def __call__(self, dataset, x1, x2, x3):
        """
        Interpolate the desired dataset at the specified coordinates.

        :arg dataset: label for the dataset to be sampled or dataset itself
        :arg x1: x1 sample points
        :arg x2: x2 sample points
        :arg x3: x3 sample points

        :returns: interpolated data at the specified coordinates
        """
        # check if dataset is a label or dataset
        if isinstance(dataset, str):
            interp = self.datasets[dataset]
        else:
            kwargs = dict(method=self.method_rgi, fill_value=np.nan, bounds_error=False)
            interp = RGI((self.a, self.b, self.c), self._extend(dataset), **kwargs)

        # convert to pseudocoordinates
        if self.use_log_x1:
            x1 = np.log(x1)
        pts = np.column_stack([x1.ravel(), x2.ravel()])
        A = self.a_interp(pts).reshape(x1.shape)
        B = self.b_interp(pts).reshape(x1.shape)

        # interpolate and return reshaped data
        data = interp((A.flatten(), B.flatten(), x3.flatten()))
        return data.reshape(x1.shape)
