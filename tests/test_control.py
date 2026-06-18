"""Phase 2: go-to-pose control — pure velocity_command + closed-loop drive in sim."""

import math

import pytest

from feathersim.control.go_to_pose import (
    PoseGains,
    PoseTolerance,
    drive_to_pose,
    pose_error,
    velocity_command,
    wrap_to_pi,
)
from feathersim.sim.world import World


# --- pure pieces -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "angle,expected",
    [(0.0, 0.0), (math.pi, -math.pi), (3 * math.pi, -math.pi), (-3 * math.pi / 2, math.pi / 2)],
)
def test_wrap_to_pi(angle, expected):
    assert wrap_to_pi(angle) == pytest.approx(expected)


def test_velocity_command_zero_at_target():
    vx, vy, omega = velocity_command((1.0, 2.0, 0.5), (1.0, 2.0, 0.5))
    assert (vx, vy, omega) == pytest.approx((0.0, 0.0, 0.0))


def test_velocity_command_uses_body_frame():
    # Robot faces world +y (yaw=pi/2). A target at world +x is to the robot's RIGHT, so the error
    # must show up as strafe (vy<0), not forward motion (vx~0).
    vx, vy, omega = velocity_command((0.0, 0.0, math.pi / 2), (1.0, 0.0, math.pi / 2))
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy < 0.0
    assert omega == pytest.approx(0.0, abs=1e-9)


def test_velocity_command_saturates():
    gains = PoseGains(kp_linear=2.0, max_linear=1.0, kp_angular=3.0, max_angular=2.0)
    vx, vy, omega = velocity_command((0.0, 0.0, 0.0), (100.0, -100.0, 3.0), gains)
    assert vx == pytest.approx(1.0)
    assert vy == pytest.approx(-1.0)
    assert omega == pytest.approx(2.0)


def test_pose_error():
    dist, head = pose_error((0.0, 0.0, 0.0), (3.0, 4.0, math.pi / 2))
    assert dist == pytest.approx(5.0)
    assert head == pytest.approx(math.pi / 2)


# --- closed-loop drive in sim ------------------------------------------------------------

def test_drive_reaches_pose_translate_strafe_rotate():
    world = World(n_machines=3, seed=0)
    target = (1.0, 0.5, 0.8)
    tol = PoseTolerance()
    result = drive_to_pose(world, target, tolerance=tol)
    assert result.reached
    assert result.steps < 5000
    assert result.position_error <= tol.position
    assert result.heading_error <= tol.heading


def test_drive_reaches_heading_only_change():
    world = World(n_machines=2, seed=1)
    result = drive_to_pose(world, (0.0, 0.0, -2.5))
    assert result.reached
    assert result.position_error <= PoseTolerance().position
    assert result.heading_error <= PoseTolerance().heading


def test_base_halts_after_arrival():
    world = World(n_machines=2, seed=0)
    result = drive_to_pose(world, (0.8, -0.6, 1.0))
    assert result.reached
    settled = world.robot_pose()
    # With the base halted, stepping further must not move it.
    for _ in range(100):
        world.step()
    assert world.robot_pose() == pytest.approx(settled, abs=1e-9)


def test_drive_reports_failure_when_out_of_steps():
    world = World(n_machines=1, seed=0)
    result = drive_to_pose(world, (3.0, 3.0, 0.0), max_steps=5)
    assert not result.reached
    assert result.steps == 5
    assert result.position_error > PoseTolerance().position
