"""Phase 3: skill SDK — pre/postconditions and end-to-end tending composition."""

import math

import pytest

from feathersim.sdk import APPROACH_DISTANCE, Robot, SkillError
from feathersim.sim.machine import MachineState
from feathersim.sim.world import MACHINE_Y, TABLE_XY, World


def _park_at(robot: Robot, fixture: str) -> None:
    """Teleport the base to a fixture's tending pose (fast setup, no driving)."""
    robot.world.set_base_pose(*robot.tending_pose(fixture))


# --- pose geometry -----------------------------------------------------------------------

def test_tending_pose_machine_faces_machine():
    world = World(n_machines=3, seed=0)
    robot = Robot(world)
    mx, my = world.fixtures["machine_0"]
    x, y, yaw = robot.tending_pose("machine_0")
    assert (x, y) == pytest.approx((mx, MACHINE_Y - APPROACH_DISTANCE))
    assert yaw == pytest.approx(math.pi / 2)  # facing +y toward the machine


def test_tending_pose_table_faces_table():
    robot = Robot(World(n_machines=2, seed=0))
    x, y, yaw = robot.tending_pose("table")
    assert (x, y) == pytest.approx((TABLE_XY[0], TABLE_XY[1] + APPROACH_DISTANCE))
    assert yaw == pytest.approx(-math.pi / 2)  # facing −y toward the table


def test_unknown_fixture_raises():
    robot = Robot(World(n_machines=1, seed=0))
    with pytest.raises(SkillError):
        robot.tending_pose("machine_9")


# --- move_to -----------------------------------------------------------------------------

def test_move_to_fixture_reaches_tending_pose():
    world = World(n_machines=3, seed=0)
    robot = Robot(world)
    robot.move_to("machine_1")
    assert robot.pose == pytest.approx(robot.tending_pose("machine_1"), abs=0.05)


def test_move_to_explicit_pose():
    robot = Robot(World(n_machines=1, seed=0))
    robot.move_to((0.5, -0.3, 0.4))
    assert robot.pose == pytest.approx((0.5, -0.3, 0.4), abs=0.05)


def test_move_to_off_contract_target_raises_skill_error():
    # A list isn't a pose tuple and isn't a fixture name → clean SkillError, not a raw TypeError.
    robot = Robot(World(n_machines=1, seed=0))
    with pytest.raises(SkillError):
        robot.move_to([0.5, -0.3, 0.4])  # type: ignore[arg-type]


# --- read accessors ----------------------------------------------------------------------

def test_state_accessors():
    world = World(n_machines=2, seed=0)
    robot = Robot(world)
    assert robot.machine_state("machine_0") is MachineState.IDLE
    assert robot.parts_done("machine_0") == 0
    with pytest.raises(SkillError):
        robot.machine_state("nope")


# --- pick preconditions ------------------------------------------------------------------

def test_pick_requires_being_parked():
    world = World(n_machines=2, seed=0)
    robot = Robot(world)
    world.machines[0].state = MachineState.DONE  # done, but robot is at the origin
    with pytest.raises(SkillError, match="not parked"):
        robot.pick("machine_0")


def test_pick_requires_done_machine():
    world = World(n_machines=2, seed=0)
    robot = Robot(world)
    _park_at(robot, "machine_0")               # parked, but machine still idle
    with pytest.raises(SkillError, match="not done"):
        robot.pick("machine_0")


def test_pick_unloads_and_machine_reloads():
    world = World(n_machines=2, seed=0)
    robot = Robot(world)
    _park_at(robot, "machine_0")
    world.machines[0].state = MachineState.DONE
    part = robot.pick("machine_0")
    assert robot.holding == part == "part_machine_0_0"
    assert world.machines[0].state is MachineState.IDLE   # reloaded, will resume cycling
    assert world.machines[0].parts_done == 1


def test_pick_rejects_double_hold():
    world = World(n_machines=2, seed=0)
    robot = Robot(world)
    _park_at(robot, "machine_0")
    world.machines[0].state = MachineState.DONE
    robot.pick("machine_0")
    world.machines[1].state = MachineState.DONE
    _park_at(robot, "machine_1")
    with pytest.raises(SkillError, match="already holding"):
        robot.pick("machine_1")


# --- place preconditions -----------------------------------------------------------------

def test_place_requires_holding():
    robot = Robot(World(n_machines=1, seed=0))
    _park_at(robot, "table")
    with pytest.raises(SkillError, match="not holding"):
        robot.place("table")


def test_place_requires_being_parked():
    world = World(n_machines=1, seed=0)
    robot = Robot(world)
    robot.holding = "part_x"  # holding, but sitting at the origin (not at the table)
    with pytest.raises(SkillError, match="not parked"):
        robot.place("table")


def test_place_deposits_part():
    world = World(n_machines=1, seed=0)
    robot = Robot(world)
    robot.holding = "part_x"
    _park_at(robot, "table")
    deposited = robot.place("table")
    assert deposited == "part_x"
    assert robot.holding is None
    assert robot.delivered == ["part_x"]


def test_place_accepts_explicit_pose_target():
    world = World(n_machines=1, seed=0)
    robot = Robot(world)
    drop = (0.3, -0.3, 0.0)
    world.set_base_pose(*drop)
    robot.holding = "part_x"
    assert robot.place(drop) == "part_x"  # explicit pose target, parallel to move_to's pose path
    assert robot.delivered == ["part_x"]


# --- end-to-end --------------------------------------------------------------------------

def test_tend_one_machine_end_to_end():
    world = World(n_machines=3, seed=0)
    robot = Robot(world)
    robot.wait_until_done("machine_0")
    part = robot.tend("machine_0")
    assert part == "part_machine_0_0"
    assert robot.holding is None
    assert robot.delivered == [part]
    assert world.machines[0].parts_done == 1
    # The robot finished parked at the table, machine_0 cycling again.
    assert robot.pose == pytest.approx(robot.tending_pose("table"), abs=0.05)
    assert world.machines[0].state in (MachineState.IDLE, MachineState.RUNNING)


def test_wait_until_done_times_out():
    robot = Robot(World(n_machines=1, seed=0))
    with pytest.raises(SkillError, match="not done"):
        robot.wait_until_done("machine_0", timeout_s=0.001)
