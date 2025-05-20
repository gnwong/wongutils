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


def load_binary(filename):
    """
    Load Athena++ binary file and return data in a dictionary.

    :arg filename: filename of binary file to load

    :returns: dictionary containing information about the binary file
    """

    filedata = {}

    # load file and get size
    with open(filename, "rb") as fp:
        fp.seek(0, 2)
        filesize = fp.tell()
        fp.seek(0, 0)

        # load header information and validate file format
        code_header = fp.readline().split()
        if len(code_header) < 1:
            raise TypeError("unknown file format")
        if code_header[0] != b"Athena":
            raise TypeError(
                f"bad file format \"{code_header[0].decode('utf-8')}\" "
                + '(should be "Athena")'
            )
        version = code_header[-1].split(b"=")[-1]
        if version != b"1.1":
            raise TypeError(f"unsupported file format version {version.decode('utf-8')}")

        pheader_count = int(fp.readline().split(b"=")[-1])
        pheader = {}
        for _ in range(pheader_count - 1):
            key, val = [x.strip() for x in fp.readline().decode("utf-8").split("=")]
            pheader[key] = val
        time = float(pheader["time"])
        cycle = int(pheader["cycle"])
        locsizebytes = int(pheader["size of location"])
        varsizebytes = int(pheader["size of variable"])

        nvars = int(fp.readline().split(b"=")[-1])
        var_list = [v.decode("utf-8") for v in fp.readline().split()[1:]]
        header_size = int(fp.readline().split(b"=")[-1])
        header = [
            line.decode("utf-8").split("#")[0].strip()
            for line in fp.read(header_size).split(b"\n")
        ]
        header = [line for line in header if len(line) > 0]

        if locsizebytes not in [4, 8]:
            raise ValueError(f"unsupported location size (in bytes) {locsizebytes}")
        if varsizebytes not in [4, 8]:
            raise ValueError(f"unsupported variable size (in bytes) {varsizebytes}")

        locfmt = "d" if locsizebytes == 8 else "f"
        varfmt = "d" if varsizebytes == 8 else "f"

        # load grid information from header and validate
        def get_from_header(header, blockname, keyname):
            blockname = blockname.strip()
            keyname = keyname.strip()
            if not blockname.startswith("<"):
                blockname = "<" + blockname
            if blockname[-1] != ">":
                blockname += ">"
            block = "<none>"
            for line in [entry for entry in header]:
                if line.startswith("<"):
                    block = line
                    continue
                try:
                    key, value = line.split("=")
                except ValueError:
                    raise ValueError(f"malformed header line: {line}")
                if block == blockname and key.strip() == keyname:
                    return value
            raise KeyError(f"no parameter called {blockname}/{keyname}")

        Nx1 = int(get_from_header(header, "<mesh>", "nx1"))
        Nx2 = int(get_from_header(header, "<mesh>", "nx2"))
        Nx3 = int(get_from_header(header, "<mesh>", "nx3"))
        nx1 = int(get_from_header(header, "<meshblock>", "nx1"))
        nx2 = int(get_from_header(header, "<meshblock>", "nx2"))
        nx3 = int(get_from_header(header, "<meshblock>", "nx3"))

        nghost = int(get_from_header(header, "<mesh>", "nghost"))

        x1min = float(get_from_header(header, "<mesh>", "x1min"))
        x1max = float(get_from_header(header, "<mesh>", "x1max"))
        x2min = float(get_from_header(header, "<mesh>", "x2min"))
        x2max = float(get_from_header(header, "<mesh>", "x2max"))
        x3min = float(get_from_header(header, "<mesh>", "x3min"))
        x3max = float(get_from_header(header, "<mesh>", "x3max"))

        # load data from each meshblock
        n_vars = len(var_list)
        mb_count = 0

        mb_index = []
        mb_logical = []
        mb_geometry = []

        mb_data = {}
        for var in var_list:
            mb_data[var] = []
        while fp.tell() < filesize:
            mb_index.append(
                np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64) - nghost
            )
            nx1_out = (mb_index[mb_count][1] - mb_index[mb_count][0]) + 1
            nx2_out = (mb_index[mb_count][3] - mb_index[mb_count][2]) + 1
            nx3_out = (mb_index[mb_count][5] - mb_index[mb_count][4]) + 1
            mb_logical.append(np.frombuffer(fp.read(16), dtype=np.int32))
            mb_geometry.append(
                np.frombuffer(
                    fp.read(6 * locsizebytes),
                    dtype=np.float64 if locfmt == "d" else np.float32,
                )
            )

            data = np.fromfile(
                fp,
                dtype=np.float64 if varfmt == "d" else np.float32,
                count=nx1_out * nx2_out * nx3_out * n_vars,
            )
            data = data.reshape(nvars, nx3_out, nx2_out, nx1_out)
            for vari, var in enumerate(var_list):
                mb_data[var].append(data[vari])
            mb_count += 1

    filedata["header"] = header
    filedata["time"] = time
    filedata["cycle"] = cycle
    filedata["var_names"] = var_list

    filedata["Nx1"] = Nx1
    filedata["Nx2"] = Nx2
    filedata["Nx3"] = Nx3
    filedata["nvars"] = nvars

    filedata["x1min"] = x1min
    filedata["x1max"] = x1max
    filedata["x2min"] = x2min
    filedata["x2max"] = x2max
    filedata["x3min"] = x3min
    filedata["x3max"] = x3max

    filedata["n_mbs"] = mb_count
    filedata["nx1_mb"] = nx1
    filedata["nx2_mb"] = nx2
    filedata["nx3_mb"] = nx3
    filedata["nx1_out_mb"] = (mb_index[0][1] - mb_index[0][0]) + 1
    filedata["nx2_out_mb"] = (mb_index[0][3] - mb_index[0][2]) + 1
    filedata["nx3_out_mb"] = (mb_index[0][5] - mb_index[0][4]) + 1

    filedata["mb_index"] = np.array(mb_index)
    filedata["mb_logical"] = np.array(mb_logical)
    filedata["mb_geometry"] = np.array(mb_geometry)
    filedata["mb_data"] = mb_data

    return filedata
