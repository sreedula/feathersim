"""A* global path planning on an occupancy grid — pure, no sim import. [v2 Phase B]

8-connected A* with an octile heuristic; diagonal moves are blocked when they'd cut the corner of an
occupied cell. :func:`plan_path` wraps the cell search with world↔cell conversion and a line-of-sight
shortcut pass so the followed path is direct rather than staircased.
"""

from __future__ import annotations

import heapq
import math

from feathersim.planning.occupancy import Cell, OccupancyGrid

_SQRT2 = math.sqrt(2.0)
# (drow, dcol, cost); diagonals last so ties prefer straight moves.
_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2)]


def _octile(a: Cell, b: Cell) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (_SQRT2 - 2.0) * min(dr, dc)


def astar(grid: OccupancyGrid, start: Cell, goal: Cell) -> list[Cell] | None:
    """Shortest 8-connected path of cells from ``start`` to ``goal``, or ``None`` if unreachable.

    Returns ``None`` if either endpoint is occupied/out of bounds or no path exists.
    """
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return None
    if start == goal:
        return [start]

    open_heap: list[tuple[float, Cell]] = [(0.0, start)]
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current)
        r, c = current
        for dr, dc, cost in _MOVES:
            nr, nc = r + dr, c + dc
            if not grid.is_free(nr, nc):
                continue
            if dr != 0 and dc != 0:  # no diagonal corner-cutting through an occupied orthogonal cell
                if not grid.is_free(r + dr, c) or not grid.is_free(r, c + dc):
                    continue
            tentative = g_score[current] + cost
            if tentative < g_score.get((nr, nc), math.inf):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative
                heapq.heappush(open_heap, (tentative + _octile((nr, nc), goal), (nr, nc)))
    return None


def _reconstruct(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _shortcut(grid: OccupancyGrid, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Greedy line-of-sight smoothing: keep the farthest reachable point from each anchor."""
    if len(points) <= 2:
        return points
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not grid.segment_free(points[i], points[j]):
            j -= 1
        out.append(points[j])
        i = j
    return out


def plan_path(
    grid: OccupancyGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> list[tuple[float, float]] | None:
    """Plan a smoothed world-space waypoint path from ``start_xy`` to ``goal_xy`` (``None`` if blocked)."""
    cells = astar(grid, grid.world_to_cell(*start_xy), grid.world_to_cell(*goal_xy))
    if cells is None:
        return None
    points = [grid.cell_to_world(*c) for c in cells]
    points[0], points[-1] = start_xy, goal_xy  # snap endpoints to the exact requested poses
    return _shortcut(grid, points)
