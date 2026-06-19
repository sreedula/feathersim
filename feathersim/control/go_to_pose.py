"""Go-to-pose controller for the holonomic base. [Phase 2]

A proportional controller turns a world-frame pose error into a body-frame twist (the pure,
unit-testable :func:`velocity_command`), and :func:`drive_to_pose` closes the loop in sim —
routing the commanded twist through the mecanum IK then FK each tick so the wheel kinematics is
genuinely load-bearing — until the base is within tolerance, then halts it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from feathersim.kinematics.holonomic import (
    MecanumGeometry,
    body_to_wheels,
    wheels_to_body,
)

Pose = tuple[float, float, float]  # (x, y, yaw)


class BaseDriver(Protocol):
    """The slice of the sim world :func:`drive_to_pose` needs — kept structural so this module
    stays sim-agnostic (no MuJoCo import). :class:`feathersim.sim.world.World` satisfies it."""

    def robot_pose(self) -> Pose: ...
    def command_base_velocity(self, vx: float, vy: float, omega: float) -> None: ...
    def stop_base(self) -> None: ...
    def step(self) -> None: ...


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]`` (the boundary sign at ±pi is float-rounding dependent)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class PoseGains:
    """Proportional gains and per-axis speed caps for the go-to-pose controller."""

    kp_linear: float = 2.0
    kp_angular: float = 3.0
    max_linear: float = 1.0   # m/s, applied per body axis
    max_angular: float = 2.0  # rad/s


@dataclass(frozen=True)
class PoseTolerance:
    """Arrival thresholds: euclidean position (m) and absolute heading (rad)."""

    position: float = 0.02
    heading: float = 0.02


@dataclass(frozen=True)
class DriveResult:
    """Outcome of a :func:`drive_to_pose` run."""

    reached: bool
    steps: int
    pose: Pose
    position_error: float
    heading_error: float


def pose_error(pose: Pose, target: Pose) -> tuple[float, float]:
    """Return ``(distance, |heading error|)`` between ``pose`` and ``target``."""
    x, y, yaw = pose
    tx, ty, tyaw = target
    distance = math.hypot(tx - x, ty - y)
    heading = abs(wrap_to_pi(tyaw - yaw))
    return distance, heading


def goal_in_body_frame(pose: Pose, target: Pose) -> tuple[float, float, float]:
    """The target expressed in the robot's body frame: ``(forward, left, dyaw)``.

    This is the full observation the go-to-pose control law acts on — both the P-controller and the
    Phase-D learned policy map exactly this 3-vector to a twist, so it's the shared BC observation.
    """
    x, y, yaw = pose
    tx, ty, tyaw = target
    dx, dy = tx - x, ty - y
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_to_pi(tyaw - yaw))  # R(-yaw)·error, heading error


def velocity_command(pose: Pose, target: Pose, gains: PoseGains = PoseGains()) -> Pose:
    """Pure P-control: world-frame pose error → clamped body-frame twist ``(vx, vy, omega)``.

    The translational error is rotated from world frame into the robot's body frame (x-forward,
    y-left) so commanding ``vx``/``vy`` drives that error to zero regardless of heading.
    """
    err_forward, err_left, dyaw = goal_in_body_frame(pose, target)
    vx = _clamp(gains.kp_linear * err_forward, gains.max_linear)
    vy = _clamp(gains.kp_linear * err_left, gains.max_linear)
    omega = _clamp(gains.kp_angular * dyaw, gains.max_angular)
    return (vx, vy, omega)


def drive_to_pose(
    world: BaseDriver,
    target: Pose,
    *,
    gains: PoseGains = PoseGains(),
    tolerance: PoseTolerance = PoseTolerance(),
    geom: MecanumGeometry = MecanumGeometry(),
    max_steps: int = 5000,
    velocity_fn: "Callable[[Pose, Pose, PoseGains], Pose]" = velocity_command,
) -> DriveResult:
    """Drive the sim base to ``target`` (x, y, yaw); halt on arrival or after ``max_steps``.

    Each tick: compute the body twist (via ``velocity_fn`` — the P-controller by default, or the Phase-D
    learned policy), push it through the mecanum inverse then forward kinematics (exact round-trip),
    command the base, and step the world.
    """
    for step in range(max_steps):
        pose = world.robot_pose()
        distance, heading = pose_error(pose, target)
        if distance <= tolerance.position and heading <= tolerance.heading:
            world.stop_base()
            return DriveResult(True, step, pose, distance, heading)

        vx, vy, omega = velocity_fn(pose, target, gains)
        wheels = body_to_wheels(vx, vy, omega, geom)      # IK: twist → wheel speeds
        rvx, rvy, romega = wheels_to_body(wheels, geom)   # FK: wheel speeds → twist (== input)
        world.command_base_velocity(rvx, rvy, romega)
        world.step()

    pose = world.robot_pose()
    distance, heading = pose_error(pose, target)
    reached = distance <= tolerance.position and heading <= tolerance.heading
    world.stop_base()
    return DriveResult(reached, max_steps, pose, distance, heading)
