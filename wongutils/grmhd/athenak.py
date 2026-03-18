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

import struct
import numpy as np
from tqdm import tqdm

from wongutils.grmhd.meshblocks import Meshblocks


class AthenaKSnapshot:

    def __init__(self, fname, populate_ghostzones=True, verbose=False,
                 variable_mapping=None):
        """
        Load an AthenaK snapshot file.

        :arg fname: filename of snapshot file to load
        :arg populate_ghostzones: (default=True) whether to populate ghost zones
        :arg verbose: (default=False) whether to print informational messages
        :arg variable_mapping: (default=None) variable list for primitive array
        """

        self.fname = fname

        if fname.endswith('.bin'):
            self.data = self._load_binary(fname)

        self.header = self._parse_header(self.data['header'])
        self._initialize_data(verbose=verbose, variable_mapping=variable_mapping)

        if populate_ghostzones:
            self._populate_ghostzones(verbose=verbose)

    def __repr__(self):
        """
        Return a string representation of the AthenaKSnapshot object.
        """

        # <AthenaKSnapshot: time=12.5, cycle=1200, meshblocks=128, vars=8>
        return f"<AthenaKSnapshot: fname={self.fname}, time={self.data['time']}, " \
               f"cycle={self.data['cycle']}, meshblocks={self.data['n_mbs']}, " \
               f"vars={self.nvars}>"

    def get_primitives_at(self, X, Y, Z, interpolation='linear'):
        """
        Get primitive variables at the specified coordinates.

        The input coordinate arrays must have the same shape.

        :arg X: x1 coordinates
        :arg Y: x2 coordinates
        :arg Z: x3 coordinates
        :arg interpolation: (default='linear') interpolation method to use

        :returns: primitive variables with shape ``X.shape + (self.nvars,)``
        """
        return self.get_at(self.prims, X, Y, Z, interpolation=interpolation)

    def get_at(self, data, X, Y, Z, interpolation='linear'):
        """
        Interpolate meshblock data at the specified coordinates.

        The input coordinate arrays must have the same shape. If the snapshot
        is sliced along one dimension, the corresponding input coordinates are
        ignored and replaced with the slice position before interpolation.

        :arg data: meshblock data to interpolate, with shape
                   ``(nmb, nx1, nx2, nx3, ...)``
        :arg X: x1 coordinates
        :arg Y: x2 coordinates
        :arg Z: x3 coordinates
        :arg interpolation: (default='linear') interpolation method to use

        :returns: interpolated data with shape ``X.shape + data.shape[4:]``
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)
        shape = X.shape

        if Y.shape != shape or Z.shape != shape:
            raise ValueError("X, Y, and Z must have the same shape.")

        if self.slices_position[0] is not None:
            X = np.full_like(X, self.slices_position[0])
        if self.slices_position[1] is not None:
            Y = np.full_like(Y, self.slices_position[1])
        if self.slices_position[2] is not None:
            Z = np.full_like(Z, self.slices_position[2])

        positions = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        interpolated = self.meshblocks.interpolate_data_at(data, positions,
                                                           slice_dim=self.slice_dim,
                                                           interpolation=interpolation)

        if interpolated is None:
            print('No data found at the specified coordinates.')
            return np.full(shape + data.shape[4:], np.nan)

        # reshape the interpolated data to match the input shape
        return interpolated.reshape(shape + data.shape[4:])

    def get_cell_centers(self):
        """
        Get cell-center coordinates for each meshblock.

        The returned arrays follow the same meshblock-first spatial layout as
        ``self.prims[..., 0]`` and include the ghost-zone layer populated by
        the snapshot loader.

        :returns: tuple ``(X, Y, Z)`` of x1, x2, and x3 cell-center
                  coordinate arrays

        """
        nmb, nx1g, nx2g, nx3g = self.prims.shape[:-1]
        nx1 = nx1g - 2
        nx2 = nx2g - 2
        nx3 = nx3g - 2

        x1 = np.empty((nmb, nx1g), dtype=self.mb_geometry.dtype)
        x2 = np.empty((nmb, nx2g), dtype=self.mb_geometry.dtype)
        x3 = np.empty((nmb, nx3g), dtype=self.mb_geometry.dtype)

        for mbi, (x1min, x1max, x2min, x2max, x3min, x3max) in enumerate(self.mb_geometry):
            dx1 = (x1max - x1min) / nx1
            dx2 = (x2max - x2min) / nx2
            dx3 = (x3max - x3min) / nx3

            x1[mbi] = np.linspace(x1min - 0.5 * dx1, x1max + 0.5 * dx1, nx1g)
            x2[mbi] = np.linspace(x2min - 0.5 * dx2, x2max + 0.5 * dx2, nx2g)
            x3[mbi] = np.linspace(x3min - 0.5 * dx3, x3max + 0.5 * dx3, nx3g)

        X = np.broadcast_to(x1[:, :, None, None], (nmb, nx1g, nx2g, nx3g)).copy()
        Y = np.broadcast_to(x2[:, None, :, None], (nmb, nx1g, nx2g, nx3g)).copy()
        Z = np.broadcast_to(x3[:, None, None, :], (nmb, nx1g, nx2g, nx3g)).copy()

        return X, Y, Z

    def _load_binary(self, filename):
        """
        Load AthenaK binary file and return data in a dictionary.

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

    def _parse_header(self, header):
        """Parse the header of an AthenaK snapshot file."""
        header_dict = dict()
        group = None
        for line in [ln.strip() for ln in header]:
            if line.startswith("#"):
                continue
            if line.startswith("<"):
                group = line[1:-1]
                if group not in header:
                    header_dict[group] = {}
                continue
            if group is None:
                continue
            ltoks = line.split('=')
            if len(ltoks) != 2:
                print("Unable to parse header line:", line)
                continue
            value = ltoks[1].strip()
            # try to turn into integer or float
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            header_dict[group][ltoks[0].strip()] = value
        return header_dict

    def _initialize_data(self, verbose=False, variable_mapping=None):

        # ['dens', 'velx', 'vely', 'velz', 'eint', 'bcc1', 'bcc2', 'bcc3'])
        # by convention, athenak uses particular names for the
        # primitive variables. we'll check for those as we try
        # to construct the primtiive array, which will we make
        # of shape (nmb, n3_mb, n2_mb, n1_mb, nvars)

        if variable_mapping is not None:
            self.nvars = len(variable_mapping)
            var1 = np.array(self.data['mb_data'][variable_mapping[0]])
            nmb, nz, ny, nx = var1.shape
            data = np.zeros((nmb, nz+2, ny+2, nx+2, self.nvars), dtype=var1.dtype)
            for vari, var in enumerate(variable_mapping):
                data[:, 1:-1, 1:-1, 1:-1, vari] = np.array(self.data['mb_data'][var])
        else:
            self.nvars = 8
            dens = np.array(self.data['mb_data']['dens'])
            nmb, nz, ny, nx = dens.shape
            data = np.zeros((nmb, nz+2, ny+2, nx+2, self.nvars), dtype=dens.dtype)
            data[:, 1:-1, 1:-1, 1:-1, 0] = dens
            data[:, 1:-1, 1:-1, 1:-1, 1] = np.array(self.data['mb_data']['eint'])
            data[:, 1:-1, 1:-1, 1:-1, 2] = np.array(self.data['mb_data']['velx'])
            data[:, 1:-1, 1:-1, 1:-1, 3] = np.array(self.data['mb_data']['vely'])
            data[:, 1:-1, 1:-1, 1:-1, 4] = np.array(self.data['mb_data']['velz'])
            data[:, 1:-1, 1:-1, 1:-1, 5] = np.array(self.data['mb_data']['bcc1'])
            data[:, 1:-1, 1:-1, 1:-1, 6] = np.array(self.data['mb_data']['bcc2'])
            data[:, 1:-1, 1:-1, 1:-1, 7] = np.array(self.data['mb_data']['bcc3'])

        self.prims = data.transpose((0, 3, 2, 1, 4))

        self.slices = []
        self.slices_position = [None, None, None]
        self.mb_geometry = self.data['mb_geometry']
        self.mb_levels = self.data['mb_logical'][:, 3]

        nx1 = self.data['nx1_mb']
        nx2 = self.data['nx2_mb']
        nx3 = self.data['nx3_mb']

        nx1_out = self.data['nx1_out_mb']
        nx2_out = self.data['nx2_out_mb']
        nx3_out = self.data['nx3_out_mb']

        if nx1_out == 1:
            self.slices.append(1)
            self.slices_position[0] = np.max(self.data['mb_geometry'][:, 0])
        elif nx2_out == 1:
            self.slices.append(2)
            self.slices_position[1] = np.max(self.data['mb_geometry'][:, 2])
        elif nx3_out == 1:
            self.slices.append(3)
            self.slices_position[2] = np.max(self.data['mb_geometry'][:, 4])
        if len(self.slices) > 1:
            raise ValueError("Cannot handle more than one sliced dimension.")

        self.slice_dim = self.slices[0] if len(self.slices) > 0 else None

        # this code triggers when the data in the snapshot file has output
        # meshblocks that are not the expected size. this could happen if,
        # for example, the snapshot file already includes ghost zones.
        # this behavior is currently unsupported.
        if ((1 not in self.slices) and (nx1_out != nx1)) or \
           ((2 not in self.slices) and (nx2_out != nx2)) or \
           ((3 not in self.slices) and (nx3_out != nx3)):
            raise ValueError(
                f"Mismatch in meshblock sizes: {self.data['nx1_out_mb']}, "
                f"{self.data['nx2_out_mb']}, {self.data['nx3_out_mb']} vs "
                f"{nx1}, {nx2}, {nx3}"
            )

        self.meshblocks = Meshblocks(self.mb_geometry, self.mb_levels,
                                     nx1_out, nx2_out, nx3_out)

    def _populate_ghostzones_meshblock(self, mbi):

        nx1 = self.data['nx1_out_mb']
        nx2 = self.data['nx2_out_mb']
        nx3 = self.data['nx3_out_mb']
        geometry = self.mb_geometry[mbi]

        _, x1v = self.meshblocks._get_edges_and_verts(geometry[0], geometry[1], nx1)
        _, x2v = self.meshblocks._get_edges_and_verts(geometry[2], geometry[3], nx2)
        _, x3v = self.meshblocks._get_edges_and_verts(geometry[4], geometry[5], nx3)

        face_indices = [(0, 0), (0, -1), (1, 0), (1, -1), (2, 0), (2, -1)]
        all_positions = []
        face_infos = []

        for axis, idx in face_indices:
            if axis+1 in self.slices:
                continue
            if axis == 0:
                x2g, x3g = np.meshgrid(x2v, x3v, indexing='ij')
                x1g = np.full_like(x2g, x1v[idx])
                positions = np.column_stack((x1g.ravel(), x2g.ravel(), x3g.ravel()))
                shape = x1g.shape
            elif axis == 1:
                x1g, x3g = np.meshgrid(x1v, x3v, indexing='ij')
                x2g = np.full_like(x1g, x2v[idx])
                positions = np.column_stack((x1g.ravel(), x2g.ravel(), x3g.ravel()))
                shape = x1g.shape
            elif axis == 2:
                x1g, x2g = np.meshgrid(x1v, x2v, indexing='ij')
                x3g = np.full_like(x1g, x3v[idx])
                positions = np.column_stack((x1g.ravel(), x2g.ravel(), x3g.ravel()))
                shape = x1g.shape

            all_positions.append(positions)
            face_infos.append((axis, idx, shape, positions.shape[0]))

        all_positions = np.vstack(all_positions)
        interpolated = self.meshblocks.interpolate_data_at(self.prims, all_positions,
                                                           slice_dim=self.slice_dim)

        failures = 0
        if interpolated is None:
            return len(face_infos)

        cursor = 0
        for axis, idx, shape, count in face_infos:

            face_data = interpolated[cursor:cursor + count]
            cursor += count

            if face_data.ndim == 1:
                face_data = face_data.reshape(shape)
            else:
                face_data = face_data.reshape(shape + (face_data.shape[1],))

            slicer = [slice(None)] * 3
            slicer[axis] = idx
            face_slice = tuple(slicer)

            target = self.prims[mbi][face_slice]
            mask = np.isfinite(face_data)
            failures += np.count_nonzero(~mask)
            target[mask] = face_data[mask]
            self.prims[mbi][face_slice] = target

        return failures

    def _populate_ghostzones(self, verbose=False):

        data = self.prims

        for mbi in tqdm(np.argsort(self.mb_levels)):
            self._populate_ghostzones_meshblock(mbi)

        self.prims = data


class AthenaKRestart:
    _PARAMETER_SCAN_LIMIT = 40 * 1024
    _REGION_SIZE_FIELDS = (
        'x1min', 'x2min', 'x3min',
        'x1max', 'x2max', 'x3max',
        'dx1', 'dx2', 'dx3',
    )
    _REGION_INDCS_FIELDS = (
        'ng', 'nx1', 'nx2', 'nx3',
        'is', 'ie', 'js', 'je', 'ks', 'ke',
        'cnx1', 'cnx2', 'cnx3',
        'cis', 'cie', 'cjs', 'cje', 'cks', 'cke',
    )

    def __init__(self, fname):
        """
        Load an AthenaK restart file into byte-preserving sections.

        :arg fname: filename of restart file to load
        """

        self.fname = fname

        if fname.endswith('.rst'):
            self.data = self._load_restart(fname)
        else:
            raise ValueError(f"Unsupported AthenaK restart filename: {fname}")

        self.header = self.data['header_dict']
        self.sections = self.data['sections']
        self.raw_bytes = self.data['raw_bytes']

    def __repr__(self):
        """
        Return a string representation of the AthenaKRestart object.
        """

        fields = [
            f"fname={self.fname}",
            f"time={self.data['time']}",
            f"cycle={self.data['ncycle']}",
            f"meshblocks={self.data['nmb_total']}",
        ]
        if self.data['n_records'] is not None:
            fields.append(f"records={self.data['n_records']}")

        return f"<AthenaKRestart: {', '.join(fields)}>"

    def write(self, filename):
        """
        Write this restart file and update cached serialized sections.

        :arg filename: filename where the restart file should be written
        """

        sections = self._serialize_sections()
        raw = b''.join((
            sections['parameter_dump'],
            sections['mesh_header'],
            sections['meshblock_metadata'],
            sections['internal_state'],
            sections['data_size'],
            sections['payload'],
        ))

        with open(filename, 'wb') as fp:
            fp.write(raw)

        self.sections = sections
        self.raw_bytes = raw
        self.data['sections'] = sections
        self.data['raw_bytes'] = raw
        self.data['parameter_dump'] = sections['parameter_dump']
        self.data['data_size'] = struct.unpack('@Q', sections['data_size'])[0]
        self.data['data_size_raw'] = sections['data_size']
        self.data['payload_raw'] = sections['payload']
        self.data['n_records'] = len(sections['payload']) // self.data['data_size']
        self.data['tail_raw'] = sections['tail']

    def _serialize_sections(self):
        data_size = self._get_serialized_data_size()
        sections = {
            'parameter_dump': self._serialize_parameter_dump(),
            'mesh_header': self._serialize_mesh_header(),
            'meshblock_metadata': self._serialize_meshblock_metadata(),
            'internal_state': self._coerce_bytes(self.data.get('internal_state_raw', b'')),
            'data_size': struct.pack('@Q', data_size),
            'payload': self._serialize_payload(data_size),
        }
        sections['tail'] = b''.join((
            sections['internal_state'],
            sections['data_size'],
            sections['payload'],
        ))
        return sections

    def _load_restart(self, filename):
        with open(filename, 'rb') as fp:
            raw = fp.read()

        filedata = {'raw_bytes': raw}

        parameter_data = self._extract_parameter_dump(raw)
        filedata.update(parameter_data)

        mesh_header = self._load_mesh_header(raw, parameter_data['parameter_dump_end'],
                                             parameter_data['header_dict'])
        filedata.update(mesh_header)

        cursor = mesh_header['mesh_header_end']
        nmb_total = mesh_header['nmb_total']

        logical_locations_size = nmb_total * 4 * np.dtype(np.int32).itemsize
        costs_size = nmb_total * np.dtype(np.float32).itemsize
        section2_end = cursor + logical_locations_size + costs_size
        if section2_end > len(raw):
            raise ValueError('restart file ended while reading meshblock metadata')

        logical_locations = np.frombuffer(raw, dtype=np.int32, count=4 * nmb_total,
                                          offset=cursor).copy().reshape(nmb_total, 4)
        costs = np.frombuffer(raw, dtype=np.float32, count=nmb_total,
                              offset=cursor + logical_locations_size).copy()

        filedata['logical_locations'] = logical_locations
        filedata['costs'] = costs
        filedata['section2_start'] = cursor
        filedata['section2_end'] = section2_end

        payload_data = self._load_payload_layout(raw, section2_end,
                                                 mesh_header['mb_indcs'],
                                                 mesh_header['real_size'],
                                                 nmb_total)
        filedata.update(payload_data)

        filedata['sections'] = {
            'parameter_dump': raw[:parameter_data['parameter_dump_end']],
            'mesh_header': raw[parameter_data['parameter_dump_end']:mesh_header['mesh_header_end']],
            'meshblock_metadata': raw[cursor:section2_end],
            'internal_state': filedata['internal_state_raw'],
            'data_size': filedata['data_size_raw'],
            'payload': filedata['payload_raw'],
            'tail': filedata['tail_raw'],
        }

        return filedata

    def _extract_parameter_dump(self, raw):
        scan = raw[:self._PARAMETER_SCAN_LIMIT]
        marker = scan.find(b'<par_end>')
        if marker < 0:
            raise ValueError('unable to find <par_end> in the restart parameter dump')

        line_end = raw.find(b'\n', marker)
        if line_end < 0:
            raise ValueError('restart parameter dump ended without a newline')

        parameter_dump_end = line_end + 1
        parameter_dump = raw[:parameter_dump_end]
        header_lines = [
            line.decode('utf-8').split('#')[0].strip()
            for line in parameter_dump.splitlines()
        ]
        header_lines = [line for line in header_lines if len(line) > 0 and line != '<par_end>']

        return {
            'parameter_dump': parameter_dump,
            'parameter_dump_end': parameter_dump_end,
            'header_lines': header_lines,
            'header_dict': self._parse_parameter_dump(header_lines),
            'parameter_text': parameter_dump.decode('utf-8'),
        }

    def _parse_parameter_dump(self, header):
        header_dict = {}
        group = None
        for line in [ln.strip() for ln in header]:
            if line.startswith('#'):
                continue
            if line.startswith('<') and line.endswith('>'):
                group = line[1:-1]
                if group not in header_dict:
                    header_dict[group] = {}
                continue
            if group is None:
                continue
            ltoks = line.split('=')
            if len(ltoks) != 2:
                continue
            value = ltoks[1].strip()
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            header_dict[group][ltoks[0].strip()] = value
        return header_dict

    def _serialize_parameter_dump(self):
        parameter_text = self.data.get('parameter_text')
        if parameter_text is None:
            header_lines = self.data.get('header_lines')
            if header_lines is None:
                raise ValueError('unable to serialize restart parameter dump')
            parameter_text = '\n'.join(header_lines)
            if len(parameter_text) > 0 and not parameter_text.endswith('\n'):
                parameter_text += '\n'
            parameter_text += '<par_end>\n'
        elif '<par_end>' not in parameter_text:
            if len(parameter_text) > 0 and not parameter_text.endswith('\n'):
                parameter_text += '\n'
            parameter_text += '<par_end>\n'
        elif not parameter_text.endswith('\n'):
            parameter_text += '\n'

        return parameter_text.encode('utf-8')

    def _load_mesh_header(self, raw, offset, header_dict):
        candidates = []
        for real_dtype in (np.float64, np.float32):
            candidate = self._parse_mesh_header_candidate(raw, offset, real_dtype)
            score = self._score_mesh_header_candidate(candidate, header_dict)
            if score is not None:
                candidate['score'] = score
                candidates.append(candidate)

        if len(candidates) == 0:
            raise ValueError('unable to infer restart floating-point precision')

        best = max(candidates, key=lambda candidate: (candidate['score'],
                                                      candidate['real_size']))
        return best

    def _parse_mesh_header_candidate(self, raw, offset, real_dtype):
        real_dtype = np.dtype(real_dtype)
        real_size = real_dtype.itemsize
        cursor = offset

        mesh_header_size = 8 + 9 * real_size + 19 * 4 + 19 * 4 + 2 * real_size + 4
        if cursor + mesh_header_size > len(raw):
            raise ValueError('restart file ended while reading the mesh header')

        nmb_total, root_level = struct.unpack_from('@ii', raw, cursor)
        cursor += 8

        mesh_size_values = np.frombuffer(raw, dtype=real_dtype, count=9, offset=cursor).copy()
        cursor += 9 * real_size

        mesh_indcs_values = np.frombuffer(raw, dtype=np.int32, count=19, offset=cursor).copy()
        cursor += 19 * np.dtype(np.int32).itemsize

        mb_indcs_values = np.frombuffer(raw, dtype=np.int32, count=19, offset=cursor).copy()
        cursor += 19 * np.dtype(np.int32).itemsize

        time, dt = np.frombuffer(raw, dtype=real_dtype, count=2, offset=cursor).copy()
        cursor += 2 * real_size

        ncycle = struct.unpack_from('@i', raw, cursor)[0]
        cursor += 4

        return {
            'real_dtype': real_dtype,
            'real_size': real_size,
            'nmb_total': nmb_total,
            'root_level': root_level,
            'mesh_size': dict(zip(self._REGION_SIZE_FIELDS, mesh_size_values)),
            'mesh_indcs': dict(zip(self._REGION_INDCS_FIELDS, mesh_indcs_values)),
            'mb_indcs': dict(zip(self._REGION_INDCS_FIELDS, mb_indcs_values)),
            'mesh_size_values': mesh_size_values,
            'mesh_indcs_values': mesh_indcs_values,
            'mb_indcs_values': mb_indcs_values,
            'time': float(time),
            'dt': float(dt),
            'ncycle': ncycle,
            'mesh_header_end': cursor,
        }

    def _serialize_mesh_header(self):
        real_dtype = self._get_real_dtype()

        mesh_size_values = [
            self.data['mesh_size'][field]
            for field in self._REGION_SIZE_FIELDS
        ]
        mesh_indcs_values = [
            self.data['mesh_indcs'][field]
            for field in self._REGION_INDCS_FIELDS
        ]
        mb_indcs_values = [
            self.data['mb_indcs'][field]
            for field in self._REGION_INDCS_FIELDS
        ]

        header = bytearray()
        header.extend(struct.pack('@ii',
                                  int(self.data['nmb_total']),
                                  int(self.data['root_level'])))
        header.extend(np.asarray(mesh_size_values, dtype=real_dtype).tobytes())
        header.extend(np.asarray(mesh_indcs_values, dtype=np.int32).tobytes())
        header.extend(np.asarray(mb_indcs_values, dtype=np.int32).tobytes())
        header.extend(np.asarray([self.data['time'], self.data['dt']],
                                 dtype=real_dtype).tobytes())
        header.extend(struct.pack('@i', int(self.data['ncycle'])))
        return bytes(header)

    def _score_mesh_header_candidate(self, candidate, header_dict):
        if candidate['nmb_total'] <= 0:
            return None
        if candidate['root_level'] < 0 or candidate['ncycle'] < 0:
            return None
        if not np.all(np.isfinite(candidate['mesh_size_values'])):
            return None
        if not np.isfinite(candidate['time']) or not np.isfinite(candidate['dt']):
            return None
        if candidate['mesh_indcs']['ng'] < 0 or candidate['mb_indcs']['ng'] < 0:
            return None

        score = 0
        comparisons = (
            ('mesh', 'x1min', candidate['mesh_size']['x1min'], 'float'),
            ('mesh', 'x1max', candidate['mesh_size']['x1max'], 'float'),
            ('mesh', 'x2min', candidate['mesh_size']['x2min'], 'float'),
            ('mesh', 'x2max', candidate['mesh_size']['x2max'], 'float'),
            ('mesh', 'x3min', candidate['mesh_size']['x3min'], 'float'),
            ('mesh', 'x3max', candidate['mesh_size']['x3max'], 'float'),
            ('mesh', 'nx1', candidate['mesh_indcs']['nx1'], 'int'),
            ('mesh', 'nx2', candidate['mesh_indcs']['nx2'], 'int'),
            ('mesh', 'nx3', candidate['mesh_indcs']['nx3'], 'int'),
            ('mesh', 'nghost', candidate['mesh_indcs']['ng'], 'int'),
            ('meshblock', 'nx1', candidate['mb_indcs']['nx1'], 'int'),
            ('meshblock', 'nx2', candidate['mb_indcs']['nx2'], 'int'),
            ('meshblock', 'nx3', candidate['mb_indcs']['nx3'], 'int'),
        )

        for group, key, value, kind in comparisons:
            expected = header_dict.get(group, {}).get(key)
            if expected is None:
                continue
            if kind == 'int':
                if int(expected) != int(value):
                    return None
            else:
                if not np.isclose(float(expected), float(value), rtol=1.e-5, atol=1.e-7):
                    return None
            score += 1

        return score

    def _load_payload_layout(self, raw, section2_end, mb_indcs, real_size, nmb_total):
        candidate = self._find_data_size_offset(raw, section2_end, mb_indcs, real_size,
                                                nmb_total)
        if candidate is None:
            return {
                'internal_state_raw': raw[section2_end:],
                'data_size_raw': b'',
                'data_size': None,
                'data_size_offset': None,
                'payload_raw': b'',
                'payload_records': tuple(),
                'payload_offsets': np.array([], dtype=np.uint64),
                'n_records': None,
                'tail_raw': raw[section2_end:],
            }

        data_size_offset, data_size, n_records = candidate
        payload_offset = data_size_offset + np.dtype(np.uint64).itemsize
        payload_raw = raw[payload_offset:]
        payload_view = memoryview(payload_raw)

        payload_offsets = np.arange(n_records + 1, dtype=np.uint64) * data_size
        payload_records = tuple(
            payload_view[int(payload_offsets[i]):int(payload_offsets[i + 1])]
            for i in range(n_records)
        )

        return {
            'internal_state_raw': raw[section2_end:data_size_offset],
            'data_size_raw': raw[data_size_offset:payload_offset],
            'data_size': data_size,
            'data_size_offset': data_size_offset,
            'payload_raw': payload_raw,
            'payload_records': payload_records,
            'payload_offsets': payload_offsets,
            'n_records': n_records,
            'tail_raw': raw[section2_end:],
        }

    def _serialize_meshblock_metadata(self):
        logical_locations = np.asarray(self.data['logical_locations'], dtype=np.int32)
        costs = np.asarray(self.data['costs'], dtype=np.float32)

        if logical_locations.shape != (int(self.data['nmb_total']), 4):
            raise ValueError('logical_locations does not match nmb_total')
        if costs.shape != (int(self.data['nmb_total']),):
            raise ValueError('costs does not match nmb_total')

        return logical_locations.tobytes() + costs.tobytes()

    def _find_data_size_offset(self, raw, offset, mb_indcs, real_size, nmb_total):
        min_record_size = self._minimum_record_size(mb_indcs, real_size)
        max_search = min(len(raw) - np.dtype(np.uint64).itemsize, offset + 1024 * 1024)

        candidates = []
        for data_size_offset in range(offset, max_search + 1):
            data_size = struct.unpack_from('@Q', raw, data_size_offset)[0]
            if data_size < min_record_size or data_size % real_size != 0:
                continue

            payload_nbytes = len(raw) - (data_size_offset + np.dtype(np.uint64).itemsize)
            if payload_nbytes < data_size or payload_nbytes % data_size != 0:
                continue

            n_records = payload_nbytes // data_size
            if n_records == 0 or n_records > nmb_total:
                continue

            step3_size = data_size_offset - offset
            score = (3 if n_records == nmb_total else 0) - step3_size
            candidates.append((score, step3_size, data_size_offset, data_size, n_records))

        if len(candidates) == 0:
            return None

        _, _, data_size_offset, data_size, n_records = max(
            candidates,
            key=lambda item: (item[0], -item[1], item[3]),
        )
        return data_size_offset, data_size, n_records

    def _minimum_record_size(self, mb_indcs, real_size):
        ng = mb_indcs['ng']
        nout1 = mb_indcs['nx1'] + 2 * ng
        nout2 = mb_indcs['nx2'] + 2 * ng if mb_indcs['nx2'] > 1 else 1
        nout3 = mb_indcs['nx3'] + 2 * ng if mb_indcs['nx3'] > 1 else 1
        return nout1 * nout2 * nout3 * real_size

    def _serialize_payload(self, data_size):
        payload_records = self.data.get('payload_records', tuple())
        if len(payload_records) > 0:
            payload_parts = []
            for record in payload_records:
                record_bytes = self._coerce_bytes(record)
                if len(record_bytes) != data_size:
                    raise ValueError('restart payload record size is inconsistent')
                payload_parts.append(record_bytes)
            payload = b''.join(payload_parts)
        else:
            payload = self._coerce_bytes(self.data.get('payload_raw', b''))

        if len(payload) == 0:
            raise ValueError('restart payload is empty')
        if len(payload) % data_size != 0:
            raise ValueError('restart payload size is inconsistent with data_size')

        n_records = len(payload) // data_size
        expected_n_records = self.data.get('n_records')
        if expected_n_records is not None and n_records != int(expected_n_records):
            raise ValueError('restart payload record count is inconsistent')

        return payload

    def _get_serialized_data_size(self):
        data_size = self.data.get('data_size')
        if data_size is not None:
            return int(data_size)

        payload_records = self.data.get('payload_records', tuple())
        if len(payload_records) > 0:
            data_size = len(self._coerce_bytes(payload_records[0]))
            for record in payload_records[1:]:
                if len(self._coerce_bytes(record)) != data_size:
                    raise ValueError('restart payload records do not share a common size')
            return data_size

        payload = self._coerce_bytes(self.data.get('payload_raw', b''))
        n_records = self.data.get('n_records')
        if len(payload) == 0 or n_records in [None, 0]:
            raise ValueError('unable to determine restart data_size')
        if len(payload) % int(n_records) != 0:
            raise ValueError('restart payload size is inconsistent with n_records')
        return len(payload) // int(n_records)

    def _get_real_dtype(self):
        real_dtype = self.data.get('real_dtype')
        if real_dtype is not None:
            return np.dtype(real_dtype)

        real_size = int(self.data['real_size'])
        if real_size == 4:
            return np.dtype(np.float32)
        if real_size == 8:
            return np.dtype(np.float64)
        raise ValueError(f'unsupported restart real size: {real_size}')

    def _coerce_bytes(self, payload):
        if isinstance(payload, memoryview):
            return payload.tobytes()
        if isinstance(payload, bytearray):
            return bytes(payload)
        return payload
