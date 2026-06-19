"""v2 Phase B: path planning + obstacle avoidance.

The occupancy grid and A* are pure and tested directly. The waypoint follower drives the kinematic base
(no rendering needed), so the "reaches the goal without intersecting obstacles" check runs headless too.
"""

import math

import numpy as np
import pytest

from feathersim.planning import build_grid, plan_path
from feathersim.planning.astar import astar
from feathersim.planning.occupancy import OccupancyGrid, Rect
from feathersim.sdk.robot import Robot, SkillError
from feathersim.sim.world import _OBSTACLE_HALF, _OBSTACLE_POSITIONS, ROBOT_RADIUS, World


def _open_grid(n: int = 10) -> OccupancyGrid:
    return OccupancyGrid((0.0, 0.0, float(n), float(n)), 1.0, np.zeros((n, n), dtype=bool))


# --- occupancy grid ------------------------------------------------------------------------------


def test_build_grid_marks_and_inflates_obstacle():
    grid = build_grid([Rect(0.0, 0.0, 0.2, 0.2)], (-1.0, -1.0, 1.0, 1.0), resolution=0.1, inflation=0.2)
    assert not grid.is_free(*grid.world_to_cell(0.0, 0.0))      # obstacle center occupied
    assert not grid.is_free(*grid.world_to_cell(0.35, 0.0))     # within the inflated band (0.2+0.2)
    assert grid.is_free(*grid.world_to_cell(0.9, 0.9))          # far corner free


def test_segment_free_detects_obstacle_crossing():
    grid = build_grid([Rect(0.0, 0.0, 0.3, 0.3)], (-1.0, -1.0, 1.0, 1.0), resolution=0.1)
    assert not grid.segment_free((-0.9, 0.0), (0.9, 0.0))   # straight through the obstacle
    assert grid.segment_free((-0.9, -0.9), (-0.9, 0.9))     # clear of it


# --- A* ------------------------------------------------------------------------------------------


def test_astar_finds_path_in_open_grid():
    grid = _open_grid(10)
    path = astar(grid, (0, 0), (9, 9))
    assert path is not None and path[0] == (0, 0) and path[-1] == (9, 9)


def test_astar_returns_none_when_goal_walled_off():
    g = np.zeros((10, 10), dtype=bool)
    g[:, 5] = True  # full vertical wall — no gap
    grid = OccupancyGrid((0.0, 0.0, 10.0, 10.0), 1.0, g)
    assert astar(grid, (0, 0), (0, 9)) is None


def test_astar_routes_around_a_barrier():
    g = np.zeros((10, 10), dtype=bool)
    g[0:8, 5] = True  # wall with a gap at the top (rows 8,9 open)
    grid = OccupancyGrid((0.0, 0.0, 10.0, 10.0), 1.0, g)
    path = astar(grid, (0, 0), (0, 9))
    assert path is not None
    assert all(not g[r, c] for r, c in path)        # never steps on an occupied cell
    assert any(r >= 8 for r, _ in path)             # detoured up through the gap


def test_astar_rejects_occupied_endpoints():
    g = np.zeros((5, 5), dtype=bool)
    g[2, 2] = True
    grid = OccupancyGrid((0.0, 0.0, 5.0, 5.0), 1.0, g)
    assert astar(grid, (2, 2), (4, 4)) is None      # start occupied
    assert astar(grid, (0, 0), (2, 2)) is None      # goal occupied


def test_plan_path_snaps_endpoints_and_stays_free():
    grid = build_grid([Rect(0.0, 0.0, 0.3, 0.3)], (-2.0, -2.0, 2.0, 2.0), resolution=0.1, inflation=0.2)
    start, goal = (-1.5, 0.0), (1.5, 0.0)
    path = plan_path(grid, start, goal)
    assert path is not None
    assert path[0] == start and path[-1] == goal                       # exact endpoints
    assert all(grid.segment_free(path[i], path[i + 1]) for i in range(len(path) - 1))


# --- world integration ---------------------------------------------------------------------------


def test_world_obstacle_count_validated():
    with pytest.raises(ValueError):
        World(n_machines=3, seed=0, n_obstacles=99)


def test_world_grid_includes_machines_table_and_obstacles():
    world = World(n_machines=3, seed=0, n_obstacles=2)
    rects = world.obstacle_rects()
    assert len(rects) == 3 + 1 + 2  # machines + table + obstacles
    grid = world.occupancy_grid()
    for ox, oy in _OBSTACLE_POSITIONS:
        assert not grid.is_free(*grid.world_to_cell(ox, oy))


def test_every_tending_pose_is_reachable_on_the_grid():
    """Self-documents the geometry coupling (APPROACH_DISTANCE / machine half-extent / ROBOT_RADIUS):
    every fixture's tending pose must land on a free cell, or planned tends silently fail."""
    world = World(n_machines=3, seed=0, n_obstacles=2)
    robot = Robot(world, plan=True)
    grid = world.occupancy_grid()
    for fixture in ("machine_0", "machine_1", "machine_2", "table"):
        x, y, _ = robot.tending_pose(fixture)
        assert grid.is_free(*grid.world_to_cell(x, y)), fixture


def _dist_to_rect(x: float, y: float, r: Rect) -> float:
    return math.hypot(max(abs(x - r.cx) - r.hx, 0.0), max(abs(y - r.cy) - r.hy, 0.0))


_MACHINES = ("machine_0", "machine_1", "machine_2")
# Every leg the autonomy loop can drive: tend() = move_to(machine)→move_to("table") (table↔machine,
# both directions), the first tend from the origin start, and a machine→machine leg if a perception
# false positive parks the robot at a machine before it reselects. All are checked so a future geometry
# change can't silently regress any of them (sampling only one masked a real body hit last round).
_DRIVEN_LEGS = (
    [(m, "table") for m in _MACHINES]
    + [("table", m) for m in _MACHINES]
    + [("origin", m) for m in _MACHINES]
    + [("machine_0", "machine_2")]
)


@pytest.mark.parametrize("start, goal", _DRIVEN_LEGS)
def test_planned_legs_reach_goal_without_body_hitting_obstacles(start, goal):
    """For *every* leg the loop drives, the planned route reaches the goal and the robot *body* (not
    just its centre) clears every obstacle — the follower bows into the inflation band on turns."""
    world = World(n_machines=3, seed=0, n_obstacles=2)
    robot = Robot(world, plan=True)
    start_pose = (0.0, 0.0, 0.0) if start == "origin" else robot.tending_pose(start)
    world.set_base_pose(*start_pose)

    traj: list[tuple[float, float]] = []
    inner = world.step
    world.step = lambda: (inner(), traj.append(world.robot_pose()[:2]))[0]  # type: ignore[assignment]
    robot.move_to(goal)

    obstacles = [Rect(ox, oy, _OBSTACLE_HALF, _OBSTACLE_HALF) for ox, oy in _OBSTACLE_POSITIONS]
    clearance = min(_dist_to_rect(x, y, r) for x, y in traj for r in obstacles)
    # clearance = centre-to-raw-obstacle, so > ROBOT_RADIUS proves the whole body cleared (no centre-only check).
    assert clearance > ROBOT_RADIUS, f"{start}->{goal}: body clearance {clearance:.3f} ≤ radius {ROBOT_RADIUS}"
    gx, gy, _ = robot.tending_pose(goal)
    rx, ry, _ = robot.pose
    assert math.hypot(rx - gx, ry - gy) < 0.06


def test_planning_disabled_by_default():
    assert Robot(World(n_machines=1, seed=0))._grid is None


def test_planned_move_to_unreachable_goal_raises():
    world = World(n_machines=3, seed=0, n_obstacles=2)
    robot = Robot(world, plan=True)
    with pytest.raises(SkillError, match="no path"):
        robot.move_to((_OBSTACLE_POSITIONS[0][0], _OBSTACLE_POSITIONS[0][1], 0.0))  # inside an obstacle
