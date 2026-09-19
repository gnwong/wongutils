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


class Meshblocks:
    def __init__(self, bounds_data, levels_data, nx1_mb, nx2_mb, nx3_mb):
        self.bounds = np.asarray(bounds_data)
        self.blocks = [(AABB(bounds), i) for i, bounds in enumerate(bounds_data)]
        self.bvh = BVHNode(self.blocks)

        self.nx1_mb = nx1_mb
        self.nx2_mb = nx2_mb
        self.nx3_mb = nx3_mb
        self.mb_levels = levels_data
        self.dxs = (bounds_data[:, [1, 3, 5]] - bounds_data[:, [0, 2, 4]])
        self.dxs = self.dxs / [self.nx1_mb, self.nx2_mb, self.nx3_mb]
        self.xmins = bounds_data[:, [0, 2, 4]] + self.dxs / 2

        self.neighbor_blocks = [None] * len(self.bounds)

    def find_blocks(self, points):
        points = np.asarray(points)
        if points.ndim == 1:
            return self.bvh.find_block(points)
        elif points.ndim == 2 and points.shape[1] == 3:
            return np.array(self.bvh.find_blocks_batch(points))
        else:
            raise ValueError("Invalid shape for points: {}".format(points.shape))

    def _get_edges_and_verts(self, start, end, n):
        """
        Compute edges and vertices for a uniform grid given information
        saved in the geometry on meshblock limits.
        """
        edges = np.linspace(start, end, n + 1)
        dx = edges[1] - edges[0]
        verts = np.linspace(start - dx / 2, end + dx / 2, n + 2)
        return edges, verts

    def find_blocks_for_plane(self, mbi, axis, normal, coord1, coord2):
        """Find blocks for a regular plane adjacent to one meshblock.

        Neighbor candidates are cached on first use so snapshots that do not
        populate ghost zones avoid the all-block overlap search.
        """

        other_axes = [dim for dim in range(3) if dim != axis]
        candidates = self._get_neighbor_blocks(mbi)
        bounds = self.bounds[candidates]

        contains_normal = (normal >= bounds[:, 2*axis])
        contains_normal &= (normal <= bounds[:, 2*axis+1])
        contains_coord1 = (coord1[:, None] >= bounds[:, 2*other_axes[0]])
        contains_coord1 &= (coord1[:, None] <= bounds[:, 2*other_axes[0]+1])
        contains_coord2 = (coord2[:, None] >= bounds[:, 2*other_axes[1]])
        contains_coord2 &= (coord2[:, None] <= bounds[:, 2*other_axes[1]+1])

        contains = contains_coord1[:, None, :] & contains_coord2[None, :, :]
        contains &= contains_normal
        valid = np.any(contains, axis=-1)
        reverse_index = np.argmax(contains[..., ::-1], axis=-1)

        block_ids = np.full(valid.shape, -1, dtype=int)
        block_ids[valid] = candidates[-1-reverse_index[valid]]
        return block_ids.ravel()

    def _get_neighbor_blocks(self, mbi):
        if self.neighbor_blocks[mbi] is None:
            bounds = self.bounds[mbi]
            dx = self.dxs[mbi]
            lower = bounds[::2] - dx / 2
            upper = bounds[1::2] + dx / 2
            overlaps = np.all(self.bounds[:, 1::2] >= lower, axis=1)
            overlaps &= np.all(self.bounds[:, ::2] <= upper, axis=1)
            self.neighbor_blocks[mbi] = np.flatnonzero(overlaps)
        return self.neighbor_blocks[mbi]

    # helper function for nearest neighbor fill
    def _nearest_neighbor_fill(self, d, mesh_ids, ii, dd):
        """
        Assumes ii is in the range [0, nx-1] for each axis (i.e., autocorrects
        for ghost zones).
        """
        x = ii[:, 0] + (dd[:, 0] + 1.5).astype(int)
        y = ii[:, 1] + (dd[:, 1] + 1.5).astype(int)
        z = ii[:, 2] + (dd[:, 2] + 1.5).astype(int)
        return d[mesh_ids, x, y, z]

    # helper function for bilinear interpolation
    def _bilinear_interpolate(self, d, mesh_ids, ii, dd, slice_dim):
        """
        Assumes ii is in the range [0, nx-1] for each axis (i.e., autocorrects
        for ghost zones).
        """

        x = ii[:, 0] + 1
        y = ii[:, 1] + 1
        z = ii[:, 2] + 1
        dx = dd[:, 0][..., None]
        dy = dd[:, 1][..., None]
        dz = dd[:, 2][..., None]

        if slice_dim == 1:
            x = np.ones_like(x)
            c00 = d[mesh_ids, x, y, z]
            c10 = d[mesh_ids, x, y + 1, z]
            c01 = d[mesh_ids, x, y, z + 1]
            c11 = d[mesh_ids, x, y + 1, z + 1]
            da = dy
            db = dz
        elif slice_dim == 2:
            y = np.ones_like(y)
            c00 = d[mesh_ids, x, y, z]
            c10 = d[mesh_ids, x + 1, y, z]
            c01 = d[mesh_ids, x, y, z + 1]
            c11 = d[mesh_ids, x + 1, y, z + 1]
            da = dx
            db = dz
        elif slice_dim == 3:
            z = np.ones_like(z)
            c00 = d[mesh_ids, x, y, z]
            c10 = d[mesh_ids, x + 1, y, z]
            c01 = d[mesh_ids, x, y + 1, z]
            c11 = d[mesh_ids, x + 1, y + 1, z]
            da = dx
            db = dy
        else:
            raise ValueError("Invalid slice_dim: {}".format(slice_dim))

        return (
            c00 * (1 - da) * (1 - db)
            + c10 * da * (1 - db)
            + c01 * (1 - da) * db
            + c11 * da * db
        )

    # helper function for trilinear interpolation
    def _trilinear_interpolate(self, d, mesh_ids, ii, dd,
                               combine_weights=False):
        """
        Assumes ii is in the range [0, nx-1] for each axis (i.e., autocorrects
        for ghost zones).
        """
        x = ii[:, 0] + 1
        y = ii[:, 1] + 1
        z = ii[:, 2] + 1
        dx = dd[:, 0][..., None]
        dy = dd[:, 1][..., None]
        dz = dd[:, 2][..., None]

        c000 = d[mesh_ids, x, y, z]
        c100 = d[mesh_ids, x + 1, y, z]
        c010 = d[mesh_ids, x, y + 1, z]
        c110 = d[mesh_ids, x + 1, y + 1, z]
        c001 = d[mesh_ids, x, y, z + 1]
        c101 = d[mesh_ids, x + 1, y, z + 1]
        c011 = d[mesh_ids, x, y + 1, z + 1]
        c111 = d[mesh_ids, x + 1, y + 1, z + 1]

        if combine_weights:
            mx = 1 - dx
            my = 1 - dy
            mz = 1 - dz
            return (
                c000 * (mx * my * mz)
                + c100 * (dx * my * mz)
                + c010 * (mx * dy * mz)
                + c110 * (dx * dy * mz)
                + c001 * (mx * my * dz)
                + c101 * (dx * my * dz)
                + c011 * (mx * dy * dz)
                + c111 * (dx * dy * dz)
            )

        return (
            c000 * (1 - dx) * (1 - dy) * (1 - dz)
            + c100 * dx * (1 - dy) * (1 - dz)
            + c010 * (1 - dx) * dy * (1 - dz)
            + c110 * dx * dy * (1 - dz)
            + c001 * (1 - dx) * (1 - dy) * dz
            + c101 * dx * (1 - dy) * dz
            + c011 * (1 - dx) * dy * dz
            + c111 * dx * dy * dz
        )

    # helper function to interpolate data at a set of positions
    def interpolate_data_at(self, data, positions, levels_condition=None,
                            comparison_level=None, slice_dim=None,
                            interpolation='linear', block_ids=None,
                            target_level=None):

        # get target meshblocks
        if block_ids is None:
            block_ids = np.array(self.find_blocks(positions))
        else:
            block_ids = np.asarray(block_ids)

        # mask invalid blocks
        is_valid = np.not_equal(block_ids, -1)

        # mask meshblocks that have lower level
        if levels_condition is None:
            level_ok = np.ones_like(block_ids[is_valid].astype(int), dtype=bool)
        elif 'gtreq' in levels_condition:
            level_ok = self.mb_levels[block_ids[is_valid].astype(int)] >= comparison_level
        elif 'lt' in levels_condition:
            level_ok = self.mb_levels[block_ids[is_valid].astype(int)] < comparison_level
        else:
            raise ValueError("Unknown levels_condition: {}".format(levels_condition))

        # combine mask
        valid_mask = np.zeros_like(block_ids, dtype=bool)
        valid_mask[is_valid] = level_ok

        # skip work if no valid points
        if not np.any(valid_mask):
            return None

        # get indices and offsets
        block_ids_valid = block_ids[valid_mask].astype(int)
        positions_valid = positions[valid_mask]
        xi = (positions_valid - self.xmins[block_ids_valid]) / self.dxs[block_ids_valid]
        ii = np.floor(xi).astype(int)
        dd = xi - ii

        if interpolation == 'nearest':
            interpd = self._nearest_neighbor_fill(data, block_ids_valid, ii, dd)
            if interpd.ndim == 1:
                data = np.full(positions.shape[0], np.nan)
            else:
                data = np.full((positions.shape[0], interpd.shape[1]), np.nan)
            data[valid_mask] = interpd
            return data

        elif interpolation in ['linear']:
            if target_level is None:
                same_level = np.zeros(len(block_ids_valid), dtype=bool)
            else:
                same_level = self.mb_levels[block_ids_valid] == target_level

            if np.any(same_level):
                component_shape = data.shape[4:]
                interpd = np.empty((len(block_ids_valid),) + component_shape,
                                   dtype=np.result_type(data.dtype, dd.dtype))
                nearest = np.rint(xi[same_level]).astype(int) + 1
                interpd[same_level] = data[
                    block_ids_valid[same_level],
                    nearest[:, 0], nearest[:, 1], nearest[:, 2]
                ]
            different_level = ~same_level
            if np.any(different_level):
                if slice_dim is not None:
                    values = self._bilinear_interpolate(
                        data, block_ids_valid[different_level], ii[different_level],
                        dd[different_level], slice_dim)
                else:
                    values = self._trilinear_interpolate(
                        data, block_ids_valid[different_level], ii[different_level],
                        dd[different_level])
                if np.any(same_level):
                    interpd[different_level] = values
                else:
                    interpd = values
            if interpd.ndim == 1:
                output_shape = (positions.shape[0],)
            else:
                output_shape = (positions.shape[0], interpd.shape[1])
            output = np.full(output_shape, np.nan)
            output[valid_mask] = interpd
            return output

        raise ValueError("Unknown interpolation method: {}".format(interpolation))


class BalancedMeshblocks(Meshblocks):
    """Leaf-only meshblocks with a logical, 2:1-balanced AMR topology.

    Ghost values are evaluated at the target cell centers. Aligned,
    same-level cells are copied directly, while values across refinement
    boundaries are trilinearly interpolated.
    """

    def __init__(self, bounds_data, levels_data, logical_data,
                 nx1_mb, nx2_mb, nx3_mb):
        super().__init__(bounds_data, levels_data, nx1_mb, nx2_mb, nx3_mb)
        self.logical = np.asarray(logical_data)[:, :3]
        self.is_balanced = self._validate_topology()

        if self.is_balanced:
            self._initialize_logical_lookup()
        if self.is_balanced:
            self._initialize_ghost_indices()

    def _validate_topology(self):
        """Check the assumptions used by the integer-coordinate fast path."""

        sizes = np.array((self.nx1_mb, self.nx2_mb, self.nx3_mb))
        if self.logical.shape != (len(self.bounds), 3):
            return False
        if np.any(self.logical < 0) or np.any(sizes % 2):
            return False

        widths = self.bounds[:, 1::2] - self.bounds[:, ::2]
        scale = np.exp2(self.mb_levels)[:, None]
        if not np.allclose(widths * scale, widths[0] * scale[0]):
            return False

        origins = self.bounds[:, ::2] - self.logical * widths
        if not np.allclose(origins, origins[0]):
            return False

        locations = np.column_stack((self.mb_levels, self.logical))
        if len(np.unique(locations, axis=0)) != len(locations):
            return False

        return self._is_two_to_one_balanced()

    def _is_two_to_one_balanced(self):
        """Check touching logical blocks without an all-pairs bounds scan."""

        blocks_by_level = {}
        for level in np.unique(self.mb_levels):
            at_level = self.logical[self.mb_levels == level]
            blocks_by_level[int(level)] = {tuple(location) for location in at_level}

        levels = sorted(blocks_by_level)
        for fine_level in levels:
            fine_blocks = blocks_by_level[fine_level]
            for coarse_level in levels:
                level_difference = fine_level - coarse_level
                if level_difference < 2:
                    continue

                ratio = 1 << level_difference
                coarse_blocks = blocks_by_level[coarse_level]
                for fine_location in fine_blocks:
                    parent = np.floor_divide(fine_location, ratio)
                    remainder = np.remainder(fine_location, ratio)
                    candidates = []
                    for axis in range(3):
                        axis_candidates = [parent[axis]]
                        if remainder[axis] == 0:
                            axis_candidates.append(parent[axis] - 1)
                        if remainder[axis] == ratio - 1:
                            axis_candidates.append(parent[axis] + 1)
                        candidates.append(axis_candidates)

                    for x1 in candidates[0]:
                        for x2 in candidates[1]:
                            for x3 in candidates[2]:
                                if (x1, x2, x3) in coarse_blocks:
                                    return False

        return True

    def _initialize_logical_lookup(self):
        max_location = int(np.max(self.logical, initial=0))
        self._logical_bits = max(1, max_location.bit_length())
        if 3 * self._logical_bits >= 63:
            self.is_balanced = False
            return

        keys = self._encode_logical(self.logical)
        self._level_lookup = {}
        for level in np.unique(self.mb_levels):
            block_ids = np.flatnonzero(self.mb_levels == level)
            order = np.argsort(keys[block_ids])
            lower = np.min(self.logical[block_ids], axis=0)
            shape = np.ptp(self.logical[block_ids], axis=0) + 1
            if np.prod(shape) <= 4 * len(block_ids):
                lookup = np.full(tuple(shape), -1, dtype=int)
                relative = self.logical[block_ids] - lower
                lookup[tuple(relative.T)] = block_ids
                self._level_lookup[int(level)] = (lower, lookup)
            else:
                self._level_lookup[int(level)] = (
                    keys[block_ids][order], block_ids[order])

    def _initialize_ghost_indices(self):
        sizes = np.array((self.nx1_mb, self.nx2_mb, self.nx3_mb))
        self.ghost_faces = []
        local_indices = []

        for axis, idx in ((0, 0), (0, -1), (1, 0),
                          (1, -1), (2, 0), (2, -1)):
            other_axes = [dim for dim in range(3) if dim != axis]
            coord1, coord2 = np.meshgrid(
                np.arange(-1, sizes[other_axes[0]] + 1),
                np.arange(-1, sizes[other_axes[1]] + 1),
                indexing='ij')
            indices = np.empty((coord1.size, 3), dtype=int)
            indices[:, axis] = -1 if idx == 0 else sizes[axis]
            indices[:, other_axes[0]] = coord1.ravel()
            indices[:, other_axes[1]] = coord2.ravel()
            local_indices.append(indices)
            self.ghost_faces.append(
                (axis, idx, coord1.shape, coord1.size))

        self._ghost_local_indices = np.vstack(local_indices)

    def _encode_logical(self, logical):
        logical = np.asarray(logical, dtype=np.int64)
        return (((logical[:, 0] << self._logical_bits) | logical[:, 1])
                << self._logical_bits) | logical[:, 2]

    def _find_logical_blocks(self, logical, level):
        block_ids = np.full(len(logical), -1, dtype=int)
        lookup = self._level_lookup.get(level)
        if lookup is None:
            return block_ids

        keys, payloads = lookup
        if payloads.ndim == 3:
            relative = logical - keys
            encodable = (
                (relative[:, 0] >= 0)
                & (relative[:, 0] < payloads.shape[0]))
            encodable &= (relative[:, 1] >= 0) & \
                (relative[:, 1] < payloads.shape[1])
            encodable &= (relative[:, 2] >= 0) & \
                (relative[:, 2] < payloads.shape[2])
            target = np.flatnonzero(encodable)
            block_ids[target] = payloads[tuple(relative[encodable].T)]
            return block_ids

        limit = 1 << self._logical_bits
        encodable = (logical[:, 0] >= 0) & (logical[:, 0] < limit)
        encodable &= (logical[:, 1] >= 0) & (logical[:, 1] < limit)
        encodable &= (logical[:, 2] >= 0) & (logical[:, 2] < limit)
        encoded = self._encode_logical(logical[encodable])
        locations = np.searchsorted(keys, encoded)
        found = locations < len(keys)
        found[found] &= keys[locations[found]] == encoded[found]
        target = np.flatnonzero(encodable)[found]
        block_ids[target] = payloads[locations[found]]
        return block_ids

    def interpolate_ghostzones(self, data, mbi):
        """Interpolate one ghost shell using integer logical coordinates."""

        sizes = np.array((self.nx1_mb, self.nx2_mb, self.nx3_mb))
        level = int(self.mb_levels[mbi])
        half_indices = 2 * (
            self.logical[mbi] * sizes + self._ghost_local_indices) + 1
        block_ids = np.full(len(half_indices), -1, dtype=int)

        for level_offset, divisor in ((0, 2 * sizes),
                                      (-1, 4 * sizes),
                                      (1, sizes)):
            unresolved = block_ids == -1
            if not np.any(unresolved):
                break
            logical = np.floor_divide(half_indices[unresolved], divisor)
            block_ids[unresolved] = self._find_logical_blocks(
                logical, level + level_offset)

        valid = block_ids != -1
        if not np.any(valid):
            return None

        valid_positions = np.flatnonzero(valid)
        valid_half_indices = half_indices[valid]
        source_ids = block_ids[valid]
        level_offsets = self.mb_levels[source_ids] - level
        output = np.full(
            (len(block_ids),) + data.shape[4:], np.nan, dtype=data.dtype)
        same_level = level_offsets == 0
        if np.any(same_level):
            indices = (valid_half_indices[same_level] - 1) // 2
            indices -= self.logical[source_ids[same_level]] * sizes
            indices += 1
            output[valid_positions[same_level]] = data[
                source_ids[same_level],
                indices[:, 0], indices[:, 1], indices[:, 2]]

        different_level = ~same_level
        if np.any(different_level):
            different_ids = source_ids[different_level]
            xi = valid_half_indices[different_level] * np.exp2(
                level_offsets[different_level, None] - 1)
            xi = xi - self.logical[different_ids] * sizes - 0.5
            ii = np.floor(xi).astype(int)
            output[valid_positions[different_level]] = self._trilinear_interpolate(
                data, different_ids, ii, xi - ii,
                combine_weights=data.dtype == np.float32)

        return output


class AABB:
    def __init__(self, bounds):
        self.bounds = np.asarray(bounds)

    def contains(self, point):
        x, y, z = point
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax

    def union(self, other):
        b1 = self.bounds
        b2 = other.bounds
        return AABB((
            min(b1[0], b2[0]), max(b1[1], b2[1]),
            min(b1[2], b2[2]), max(b1[3], b2[3]),
            min(b1[4], b2[4]), max(b1[5], b2[5]),
        ))

    def center(self):
        b = self.bounds
        return np.array([
            0.5 * (b[0] + b[1]),
            0.5 * (b[2] + b[3]),
            0.5 * (b[4] + b[5]),
        ])

    def contains_batch(self, pts):
        b = self.bounds
        return ((pts[:, 0] >= b[0]) & (pts[:, 0] <= b[1])
                & (pts[:, 1] >= b[2]) & (pts[:, 1] <= b[3])
                & (pts[:, 2] >= b[4]) & (pts[:, 2] <= b[5]))


class BVHNode:
    def __init__(self, blocks, max_leaf_size=1):
        self.left = None
        self.right = None
        self.blocks = blocks if len(blocks) <= max_leaf_size else None

        self.bounds = blocks[0][0]
        for b in blocks[1:]:
            self.bounds = self.bounds.union(b[0])

        if self.blocks is None:
            centers = np.array([b[0].center() for b in blocks])
            axis = np.argmax(centers.max(0) - centers.min(0))
            blocks.sort(key=lambda b: b[0].center()[axis])
            mid = len(blocks) // 2
            self.left = BVHNode(blocks[:mid], max_leaf_size)
            self.right = BVHNode(blocks[mid:], max_leaf_size)

    def find_block(self, point):
        if not self.bounds.contains(point):
            return None
        if self.blocks is not None:
            for aabb, payload in self.blocks:
                if aabb.contains(point):
                    return payload
            return None
        return self.left.find_block(point) or self.right.find_block(point)

    def find_blocks_batch(self, points):
        points = np.asarray(points)
        results = np.full((points.shape[0],), -1, dtype=int)

        inside = self.bounds.contains_batch(points)
        if not np.any(inside):
            return results

        idxs = np.where(inside)[0]
        subpoints = points[inside]

        if self.blocks is not None:
            for aabb, payload in self.blocks:
                mask = aabb.contains_batch(subpoints)
                results[idxs[mask]] = payload
        else:
            left_results = self.left.find_blocks_batch(subpoints)
            mask = (left_results != -1)
            results[idxs[mask]] = left_results[mask]
            right_results = self.right.find_blocks_batch(subpoints)
            mask = (right_results != -1)
            results[idxs[mask]] = right_results[mask]

        return np.array(results)
