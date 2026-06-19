"""Drive the sim base along a planned waypoint path. [v2 Phase B]

Reuses the Phase-2 :func:`drive_to_pose` controller leg-by-leg: intermediate waypoints are reached on
position only (heading left free, but aimed along travel so the holonomic base looks natural), and the
final waypoint is reached on the full target pose. The base stays holonomic — it strafes between
waypoints without needing to rotate first.
"""

from __future__ import annotations

import math

from feathersim.control.go_to_pose import (
    BaseDriver,
    DriveResult,
    PoseGains,
    PoseTolerance,
    drive_to_pose,
    pose_error,
)
from feathersim.kinematics.holonomic import MecanumGeometry

Waypoint = tuple[float, float]


def follow_path(
    world: BaseDriver,
    waypoints: list[Waypoint],
    final_yaw: float,
    *,
    gains: PoseGains = PoseGains(),
    geom: MecanumGeometry = MecanumGeometry(),
    waypoint_tolerance: float = 0.04,
    final_tolerance: PoseTolerance = PoseTolerance(),
    max_steps_per_leg: int = 3000,
) -> DriveResult:
    """Drive through ``waypoints`` (last one is the goal, reached at ``final_yaw``).

    Returns the :class:`DriveResult` of the final leg; if any leg fails to arrive, returns that leg's
    (un-reached) result early so the caller can surface the failure.
    """
    # Loose heading tolerance for intermediate waypoints so the base doesn't stall aligning its yaw.
    cruise_tol = PoseTolerance(position=waypoint_tolerance, heading=math.pi)
    result: DriveResult | None = None
    for k, (wx, wy) in enumerate(waypoints):
        if k == len(waypoints) - 1:
            target, tol = (wx, wy, final_yaw), final_tolerance
        else:
            nx, ny = waypoints[k + 1]
            heading = math.atan2(ny - wy, nx - wx)  # face the next leg
            target, tol = (wx, wy, heading), cruise_tol
        # Skip a waypoint we're already on (e.g. the start point) to avoid a wasted leg.
        if k < len(waypoints) - 1 and pose_error(world.robot_pose(), target)[0] <= waypoint_tolerance:
            continue
        result = drive_to_pose(
            world, target, gains=gains, tolerance=tol, geom=geom, max_steps=max_steps_per_leg
        )
        if not result.reached:
            return result
    if result is None:  # single-waypoint path already satisfied — drive the final pose explicitly
        wx, wy = waypoints[-1]
        result = drive_to_pose(
            world, (wx, wy, final_yaw), gains=gains, tolerance=final_tolerance,
            geom=geom, max_steps=max_steps_per_leg,
        )
    return result
