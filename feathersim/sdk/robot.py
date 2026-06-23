"""Developer-facing skill SDK: a ``Robot`` facade over the sim. [Phase 3]

Hides joints, MJCF, kinematics, and controller wiring behind four skills — ``move_to``, ``pick``,
``place``, ``tend`` — plus a ``wait_until_done`` helper. Because the robot is a mobile base with no
arm, parts are modeled as *logical* handoffs: ``pick`` unloads a ``done`` machine (which then
auto-reloads and resumes its cycle) and ``place`` deposits the part onto the output table. Every
skill checks its preconditions and raises :class:`SkillError` rather than doing something nonsensical.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from feathersim.control.go_to_pose import (
    PoseGains,
    PoseTolerance,
    drive_to_pose,
    pose_error,
    velocity_command,
)
from feathersim.kinematics.holonomic import MecanumGeometry
from feathersim.planning import follow_path, plan_path
from feathersim.sim.machine import MachineState

if TYPE_CHECKING:
    from feathersim.sim.world import World

Pose = tuple[float, float, float]

# How far in front of a fixture the base parks to service it (machine/robot half-extents + clearance).
APPROACH_DISTANCE = 0.6
# "At the fixture" slack — looser than the drive tolerance so arrival reliably counts as arrived.
_AT_POSITION = 0.06
_AT_HEADING = 0.15


class SkillError(RuntimeError):
    """Base error for a skill that cannot proceed (precondition unmet, or navigation failed)."""


class PreconditionError(SkillError):
    """A skill's precondition on world/robot state is unmet (e.g. picking from a machine that isn't
    ``done``, or not parked at the target). Distinct from a navigation failure so callers — notably
    the autonomy loop recovering from a perception false positive — can tell "the world wasn't ready"
    apart from "the robot couldn't get there" and only swallow the former."""


class Robot:
    """High-level robot API. Wrap a :class:`feathersim.sim.world.World` and call skills on it."""

    def __init__(
        self,
        world: World,
        *,
        robot_id: int = 0,
        gains: PoseGains = PoseGains(),
        tolerance: PoseTolerance = PoseTolerance(),
        geom: MecanumGeometry = MecanumGeometry(),
        plan: bool = False,
        controller=velocity_command,
        animate_arm: bool = False,
    ) -> None:
        self.world = world
        self.robot_id = robot_id              # which base in the world this facade drives
        self._driver = world.driver(robot_id)
        self.gains = gains
        self.tolerance = tolerance
        self.geom = geom
        # With animate_arm=True, pick/place visibly swing the arm into the machine / over the table and
        # back (stepping the world while the parked base holds still) — for the rendered single-robot demo.
        # Default off so the SDK's pick/place stay instant logical handoffs (the fleet animates the arm in
        # its own state machine, so its robots keep this off to avoid double-animating).
        self._animate_arm = animate_arm
        # The velocity law move_to drives with — the P-controller by default, or a Phase-D learned policy.
        self._controller = controller
        self.holding: str | None = None      # name of the part currently carried, or None
        self.delivered: list[str] = []        # parts placed on the table, in order
        # With plan=True, move_to routes around the world's obstacles via A* instead of driving straight.
        # The grid is built ONCE here from the world's static obstacles; it does not track moving bodies
        # (the Phase-C fleet adds other robots as dynamic obstacles itself, replanning per tick).
        self._grid = world.occupancy_grid() if plan else None

    # --- introspection -------------------------------------------------------------------

    @property
    def pose(self) -> Pose:
        """Current base pose ``(x, y, yaw)``."""
        return self.world.robot_pose(self.robot_id)

    def machine_state(self, machine: str) -> MachineState:
        """Ground-truth state of ``machine`` (for scripting/inspection; the autonomy loop will use
        perception instead)."""
        return self._machine(machine).state

    def parts_done(self, machine: str) -> int:
        """How many finished parts ``machine`` has had unloaded."""
        return self._machine(machine).parts_done

    def tending_pose(self, fixture: str) -> Pose:
        """The pose at which the base services ``fixture`` (a machine name or ``"table"``)."""
        fx, fy = self._fixture_xy(fixture)
        # Approach from the origin side. Fixtures are laid out off the x-axis (machines +y, table −y);
        # a fixture exactly at y=0 has no "origin side", so fall back to approaching from −y.
        side = math.copysign(1.0, fy) if fy != 0.0 else 1.0
        stand = (fx, fy - side * APPROACH_DISTANCE)
        yaw = math.atan2(fy - stand[1], fx - stand[0])       # face the fixture
        return (stand[0], stand[1], yaw)

    # --- skills --------------------------------------------------------------------------

    def move_to(self, target: str | Pose) -> None:
        """Drive to ``target`` — a fixture name, ``"table"``, or an explicit ``(x, y, yaw)`` pose.

        With planning enabled (``Robot(..., plan=True)``) the route is an A* path around the world's
        obstacles; otherwise it drives straight (v1 behavior)."""
        goal = target if _is_pose(target) else self.tending_pose(target)  # type: ignore[arg-type]
        if self._grid is not None:
            waypoints = plan_path(self._grid, self.pose[:2], (goal[0], goal[1]))
            if waypoints is None:
                raise SkillError(f"no path to {target!r}")
            result = follow_path(
                self._driver, waypoints, goal[2], gains=self.gains, geom=self.geom,
                final_tolerance=self.tolerance, velocity_fn=self._controller,
            )
            if not result.reached:
                raise SkillError(
                    f"could not follow planned path to {target!r}: stalled {result.position_error:.3f}m "
                    f"off a waypoint"
                )
        else:
            result = drive_to_pose(
                self._driver, goal, gains=self.gains, tolerance=self.tolerance, geom=self.geom,
                velocity_fn=self._controller,
            )
            if not result.reached:
                raise SkillError(f"could not reach {target!r}: {result.position_error:.3f}m off")

    def pick(self, machine: str) -> str:
        """Unload the finished part from ``machine`` and carry it.

        Preconditions: not already holding; parked at the machine; the machine is ``done``. The
        machine then resets to ``idle`` (auto-reloaded with fresh stock) and resumes cycling.
        Returns the part's name.
        """
        if self.holding is not None:
            raise PreconditionError(f"already holding {self.holding!r}")
        m = self._machine(machine)
        if not self._at(self.tending_pose(machine)):
            raise PreconditionError(f"not parked at {machine!r}; call move_to first")
        if m.state is not MachineState.DONE:
            raise PreconditionError(f"{machine!r} is {m.state.value}, not done")
        self._reach_arm()                            # swing the arm into the machine (if animating)
        part = f"part_{machine}_{m.parts_done}"
        m.reset(self.world.time)  # unload finished part; machine reloads and restarts
        self.holding = part
        self.world.set_carried(self.robot_id, True)  # the part now rides this robot's gripper
        self._retract_arm()                          # retract to the carry pose
        return part

    def place(self, target: str | Pose = "table") -> str:
        """Deposit the carried part at ``target`` (default the output table).

        Precondition: parked at ``target`` and holding a part. Returns the deposited part's name.
        """
        if self.holding is None:
            raise PreconditionError("not holding a part")
        if not self._at(self.tending_pose(target) if isinstance(target, str) else target):
            raise PreconditionError(f"not parked at {target!r}; call move_to first")
        self._reach_arm()                             # extend the arm over the table (if animating)
        part, self.holding = self.holding, None
        self.delivered.append(part)
        self.world.set_carried(self.robot_id, False)  # part leaves the gripper...
        self.world.deposit_part()                     # ...and lands on the output-table stack
        self._retract_arm()
        return part

    def tend(self, machine: str) -> str:
        """Tend one machine end-to-end: go to it, unload the finished part, carry it to the table,
        and place it. The machine must already be ``done`` (use :meth:`wait_until_done`). Returns
        the delivered part's name."""
        self.move_to(machine)
        self.pick(machine)
        self.move_to("table")
        return self.place("table")

    def wait_until_done(self, machine: str, *, timeout_s: float = 60.0) -> None:
        """Step the world until ``machine`` reaches ``done``, or raise on timeout."""
        m = self._machine(machine)
        deadline = self.world.time + timeout_s
        while m.state is not MachineState.DONE:
            if self.world.time >= deadline:
                raise SkillError(f"{machine!r} not done within {timeout_s:.0f}s (state={m.state.value})")
            self.world.step()

    # --- internals -----------------------------------------------------------------------

    def _reach_arm(self) -> None:
        """Animate the arm out to the grasp/place pose (no-op unless ``animate_arm``)."""
        if not self._animate_arm:
            return
        from feathersim.sim.world import ARM_REACH
        self._slew_arm(ARM_REACH)

    def _retract_arm(self) -> None:
        """Animate the arm back to the tucked carry pose (no-op unless ``animate_arm``)."""
        if not self._animate_arm:
            return
        from feathersim.sim.world import ARM_REST
        self._slew_arm(ARM_REST)

    def _slew_arm(self, pose, *, max_steps: int = 140) -> None:
        # The parked base holds still (gravcomp arm doesn't perturb it); stepping advances sim time and
        # ticks the machine FSMs, which is fine — the machine just reset and is cycling again.
        self.world.set_arm_target(self.robot_id, pose)
        for _ in range(max_steps):
            if self.world.arm_at(self.robot_id, pose):
                break
            self.world.step()

    def _machine(self, name: str):
        for m in self.world.machines:
            if m.name == name:
                return m
        raise SkillError(f"unknown machine {name!r}")

    def _fixture_xy(self, fixture: str) -> tuple[float, float]:
        try:
            return self.world.fixtures[fixture]
        except (KeyError, TypeError):  # unknown name, or an off-contract unhashable target
            raise SkillError(f"unknown fixture {fixture!r}") from None

    def _at(self, pose: Pose) -> bool:
        distance, heading = pose_error(self.pose, pose)
        return distance <= _AT_POSITION and heading <= _AT_HEADING


def _is_pose(target) -> bool:
    return (
        isinstance(target, tuple)
        and len(target) == 3
        and all(isinstance(v, (int, float)) for v in target)
    )
