"""Occupancy grid over the floor — pure, no sim import. [v2 Phase B]

Obstacles are axis-aligned rectangles (machines, the table, static obstacles, and — in Phase C — other
robots). A cell is occupied if its center falls inside any rectangle inflated by the robot radius, so a
robot whose *center* stays on free cells keeps its whole body clear of the real obstacle.

Grid convention: ``grid[row, col]`` with ``row`` along +y and ``col`` along +x; ``True`` = occupied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Cell = tuple[int, int]  # (row, col)


@dataclass(frozen=True)
class Rect:
    """An axis-aligned obstacle footprint centered at ``(cx, cy)`` with half-extents ``(hx, hy)``."""

    cx: float
    cy: float
    hx: float
    hy: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return abs(x - self.cx) <= self.hx + margin and abs(y - self.cy) <= self.hy + margin


@dataclass
class OccupancyGrid:
    """A boolean occupancy grid with world↔cell mapping over ``bounds`` = ``(xmin, ymin, xmax, ymax)``."""

    bounds: tuple[float, float, float, float]
    resolution: float
    grid: np.ndarray  # (nrows, ncols) bool, True = occupied

    @property
    def nrows(self) -> int:
        return self.grid.shape[0]

    @property
    def ncols(self) -> int:
        return self.grid.shape[1]

    def world_to_cell(self, x: float, y: float) -> Cell:
        xmin, ymin, _, _ = self.bounds
        col = int((x - xmin) / self.resolution)
        row = int((y - ymin) / self.resolution)
        return (row, col)

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        xmin, ymin, _, _ = self.bounds
        return (xmin + (col + 0.5) * self.resolution, ymin + (row + 0.5) * self.resolution)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.nrows and 0 <= col < self.ncols

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and not self.grid[row, col]

    def segment_free(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """True if the straight world segment ``a→b`` stays on free cells (sampled at half-resolution)."""
        steps = max(1, int(math.dist(a, b) / (self.resolution * 0.5)))
        for t in np.linspace(0.0, 1.0, steps + 1):
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            if not self.is_free(*self.world_to_cell(x, y)):
                return False
        return True


def build_grid(
    rects: list[Rect],
    bounds: tuple[float, float, float, float],
    *,
    resolution: float = 0.1,
    inflation: float = 0.0,
) -> OccupancyGrid:
    """Rasterize ``rects`` (each inflated by ``inflation``) into an :class:`OccupancyGrid`."""
    xmin, ymin, xmax, ymax = bounds
    ncols = max(1, math.ceil((xmax - xmin) / resolution))
    nrows = max(1, math.ceil((ymax - ymin) / resolution))
    xs = xmin + (np.arange(ncols) + 0.5) * resolution
    ys = ymin + (np.arange(nrows) + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys)  # (nrows, ncols)
    grid = np.zeros((nrows, ncols), dtype=bool)
    for r in rects:
        grid |= (np.abs(gx - r.cx) <= r.hx + inflation) & (np.abs(gy - r.cy) <= r.hy + inflation)
    return OccupancyGrid(bounds, resolution, grid)
