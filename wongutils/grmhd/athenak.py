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

from wongutils.grmhd.meshblocks import Meshblocks


class AthenaKSnapshot:

    def __init__(self, fname):
        """
        Load Athena++ snapshot file and return data in a dictionary.

        :arg fname: filename of snapshot file to load
        """

        if fname.endswith('.bin'):
            self.data = self._load_binary(fname)

    def _load_binary(self, filename):
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
                raise TypeError(f"unsupported file fmt version {version.decode('utf-8')}")

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

    def _populate_ghostzones(self):

        from tqdm import tqdm  # TODO: remove?

        mb_geometry = self.data['mb_geometry']
        mb_levels = self.data['mb_logical'][:, 3]

        # TODO: deal with different datasets
        indata = np.array(self.data['mb_data']['dens'])
        nmb, nz, ny, nx = indata.shape

        # fill centers of data
        data = np.zeros((nmb, nz+2, ny+2, nx+2), dtype=indata.dtype)
        data[:, 1:-1, 1:-1, 1:-1] = indata
        data = data.transpose((0, 3, 2, 1))

        # TODO: check out size
        nx1 = self.data['nx1_mb']
        nx2 = self.data['nx2_mb']
        nx3 = self.data['nx3_mb']

        self.meshblocks = Meshblocks(mb_geometry, mb_levels, nx1, nx2, nx3)

        # perform sweep in two passes based on level constraints
        blocks_to_redo = []

        for mbi in tqdm(range(self.data['n_mbs'])):

            mb_level = mb_levels[mbi]

            geometry = mb_geometry[mbi]
            _, x1v = self.meshblocks._get_edges_and_verts(geometry[0], geometry[1], nx1)
            _, x2v = self.meshblocks._get_edges_and_verts(geometry[2], geometry[3], nx2)
            _, x3v = self.meshblocks._get_edges_and_verts(geometry[4], geometry[5], nx3)
            x1, x2, x3 = np.meshgrid(x1v, x2v, x3v, indexing='ij')

            face_indices = [(0, 0), (0, -1), (1, 0), (1, -1), (2, 0), (2, -1)]

            failures = 0

            for axis, idx in face_indices:

                slicer = [slice(None)] * 3
                slicer[axis] = idx
                face_slice = tuple(slicer)

                # get positions, interpolate, and fill
                positions = np.column_stack((x1[face_slice].ravel(),
                                             x2[face_slice].ravel(),
                                             x3[face_slice].ravel()))
                intrp = self.meshblocks.interpolate_data_at(data, positions,
                                                            levels_condition='gtreq',
                                                            comparison_level=mb_level)
                if intrp is None:
                    failures += 1
                    continue
                intrp = intrp.reshape(x1[face_slice].shape)
                mask = np.isfinite(intrp)
                failures += np.sum(~mask)

                data[mbi][face_slice][mask] = intrp[mask]

            if failures > 0:
                blocks_to_redo.append(mbi)

        return data
