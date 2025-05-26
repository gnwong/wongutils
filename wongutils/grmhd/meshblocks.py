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
        self.blocks = [(AABB(bounds), i) for i, bounds in enumerate(bounds_data)]
        self.bvh = BVHNode(self.blocks)

        self.nx1_mb = nx1_mb
        self.nx2_mb = nx2_mb
        self.nx3_mb = nx3_mb
        self.mb_levels = levels_data
        self.dxs = (bounds_data[:, [1, 3, 5]] - bounds_data[:, [0, 2, 4]])
        self.dxs = self.dxs / [self.nx1_mb, self.nx2_mb, self.nx3_mb]
        self.xmins = bounds_data[:, [0, 2, 4]] + self.dxs / 2

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

    # helper function for trilinear interpolation
    def _trilinear_interpolate(self, d, mesh_ids, ii, dd):
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
                            comparison_level=None):

        # get target meshblocks
        block_ids = np.array(self.find_blocks(positions))

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

        # interpolate, refill, and return
        interpd = self._trilinear_interpolate(data, block_ids_valid, ii, dd)
        if interpd.ndim == 1:
            data = np.full(positions.shape[0], np.nan)
        else:
            data = np.full((positions.shape[0], interpd.shape[1]), np.nan)
        data[valid_mask] = interpd

        return data


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
