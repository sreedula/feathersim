"""The multi-robot fleet executor — tick-based, collision-avoiding, unattended. [v2 Phase C]

The blocking Phase-5 `run_autonomy` can't drive N robots that share one `mj_step`, so the fleet is
re-expressed as a tick loop: every tick each robot advances its own state machine (`select → to_machine →
pick → to_table → place`) by one step, then the world steps once. Coordination:

- **Task allocation** via :class:`FleetManager` — a machine is locked to one robot, never double-booked.
- **Collision avoidance** has two layers: each robot **plans around every other robot** (treated as an
  inflated obstacle, replanned periodically), and a **symmetric contact backstop** (:func:`_would_collide`)
  stops any robot whose predicted next step would land within a body-clearance of another — so no robot,
  including the highest-priority one, can drive into another. Trade-off: unlike a strict-priority scheme,
  this offers no structural no-deadlock guarantee, so two robots in a tight cell can occasionally wedge;
  that's bounded by ``max_sim_seconds`` and surfaced as ``FleetReport.completed == False``.
- **Per-robot perception**: each robot reads the machine cameras through its *own* randomized sensor
  (Phase-A `corrupt_image`), so robots can genuinely disagree; the robust model usually still gets it right.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from feathersim.control.go_to_pose import PoseGains, pose_error, velocity_command
from feathersim.fleet.manager import FleetManager
from feathersim.fleet.scheduling import Strategy
from feathersim.kinematics.holonomic import MecanumGeometry, body_to_wheels, wheels_to_body
from feathersim.perception.infer import PerceivedState
from feathersim.planning import plan_path
from feathersim.planning.occupancy import Rect
from feathersim.sdk.robot import PreconditionError, Robot
from feathersim.sim.machine import MachineState
from feathersim.sim.world import ARM_REACH, ARM_REST, ROBOT_RADIUS, TIMESTEP, World

# Arrival slack — kept strictly *tighter* than the SDK's `_at` "parked" tolerance (0.06 / 0.15) so a robot
# the SM calls "arrived" reliably passes pick/place's precondition with boundary margin (rather than landing
# exactly on the SDK threshold). The arm itself is inert (gravcomp + qvel-zeroed), so it adds no drift.
_ARRIVE_POS, _ARRIVE_HEADING = 0.035, 0.08
# Two bodies overlap if their centres are closer than the sum of radii.
_BODY_CLEARANCE = 2.0 * ROBOT_RADIUS
_WAYPOINT_TOL = 0.12          # advance to the next stored waypoint within this distance
_REPLAN_INTERVAL = 0.2        # sim seconds between path replans (to react to robots that have moved)
_ROBOT_MARGIN = 0.08          # extra half-extent when treating another robot as an obstacle (planned gap)
_MIN_SEP = _BODY_CLEARANCE + 0.04   # symmetric backstop: predicted next position must clear this
_LOOKAHEAD = 0.08             # how far ahead (m) to predict the next position for the backstop
_SLOT_SPACING = 0.5           # x spacing of per-robot delivery slots — > body diameter so neighbours clear


def _table_slot(world: World, k: int) -> tuple[float, float, float]:
    """A per-robot delivery pose: a distinct x-slot along the output table so robots don't all converge
    on one point (the main multi-robot contention). Facing the table (−y)."""
    tx, ty = world.fixtures["table"]
    offset = (k - (world.n_robots - 1) / 2.0) * _SLOT_SPACING
    return (tx + offset, ty + 0.6, -math.pi / 2.0)

PerceiveFn = Callable[[int], dict[str, PerceivedState]]


@dataclass
class FleetReport:
    """Outcome of a fleet run — throughput, per-robot/-machine counts, and the closest robots ever came."""

    strategy: str
    parts_delivered: int
    completed: bool          # True if the part target was met (False = stopped at the sim-time budget)
    sim_seconds: float
    per_robot: dict[int, int]
    per_machine: dict[str, int]
    min_robot_separation: float
    events: list[tuple] = field(default_factory=list)

    @property
    def throughput_per_min(self) -> float:
        return 0.0 if self.sim_seconds <= 0 else self.parts_delivered * 60.0 / self.sim_seconds

    @property
    def collided(self) -> bool:
        """True if two robot *bodies* ever overlapped (centres closer than 2·radius)."""
        return self.min_robot_separation < _BODY_CLEARANCE


def make_perceive_fn(world, renderer, perception, randomizer, *, seed: int = 0) -> PerceiveFn:
    """Build a per-robot perception reader: render each machine, corrupt it with robot ``k``'s own
    sensor noise, and run the model. Each robot gets an independent RNG, so their readings can differ."""
    rngs = [np.random.default_rng(seed * 1000 + k) for k in range(world.n_robots)]

    def perceive(robot_id: int) -> dict[str, PerceivedState]:
        world.sync_visuals()
        out: dict[str, PerceivedState] = {}
        for i in range(world.n_machines):
            image = randomizer.corrupt_image(world.render_machine(renderer, i), rngs[robot_id])
            out[f"machine_{i}"] = perception.read(image)
        return out

    return perceive


def _min_separation(world: World) -> float:
    pts = world.robot_positions()
    if len(pts) < 2:
        return math.inf
    return min(math.dist(pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts)))


def _plan_leg(world: World, k: int, goal_xy: tuple[float, float]) -> list[tuple[float, float]] | None:
    """Plan robot ``k``'s path to ``goal_xy`` treating every *other* robot as an obstacle at its current
    position (so robots route around each other, keeping ≥2·radius apart)."""
    half = ROBOT_RADIUS + _ROBOT_MARGIN
    extras = [
        Rect(*world.robot_pose(j)[:2], half, half) for j in range(world.n_robots) if j != k
    ]
    return plan_path(world.occupancy_grid(extra_obstacles=extras), world.robot_pose(k)[:2], goal_xy)


def _would_collide(world: World, k: int, target_xy: tuple[float, float]) -> bool:
    """True if robot ``k``'s *predicted next position* (a short step toward ``target_xy``) would land
    within ``_MIN_SEP`` of any other robot. Symmetric and direction-agnostic — it catches a robot
    sliding tangentially past a neighbour, not just one driving head-on — so no robot, including the
    top-priority one, can move into another's body.

    It can't deadlock in the open: each robot's planned path routes *around* the others, so its predicted
    step normally clears them; this only fires on the transient where a stale path aims too close, and
    clears on the next replan."""
    px, py = world.robot_pose(k)[:2]
    dx, dy = target_xy[0] - px, target_xy[1] - py
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return False
    step = min(_LOOKAHEAD, dist)
    nxt = (px + dx / dist * step, py + dy / dist * step)
    return any(
        math.dist(nxt, world.robot_pose(j)[:2]) < _MIN_SEP
        for j in range(world.n_robots) if j != k
    )


class FleetController:
    """The fleet tick engine: one ``step()`` advances every robot's state machine and the world once.

    Holds all per-robot state (phase, assigned machine, planned path, last perception) so both the
    blocking :func:`run_fleet` and the dashboard can drive it and read its live state. ``velocity_fn`` is
    swappable at runtime — the rule-vs-learned controller toggle.
    """

    def __init__(
        self, world: World, perceive_fn: PerceiveFn, *, strategy: Strategy, strategy_name: str = "",
        robots: list[Robot] | None = None, gains: PoseGains = PoseGains(),
        geom: MecanumGeometry = MecanumGeometry(), select_interval: float = 0.3,
        velocity_fn=velocity_command,
        on_event: Callable[[int, str, str, float], None] | None = None,
    ) -> None:
        self.world, self.perceive_fn = world, perceive_fn
        self.strategy_name, self.gains, self.geom = strategy_name, gains, geom
        self.select_interval, self.velocity_fn, self.on_event = select_interval, velocity_fn, on_event
        n = world.n_robots
        self.robots = robots or [Robot(world, robot_id=k, gains=gains, geom=geom) for k in range(n)]
        self.manager = FleetManager({m.name: world.fixtures[m.name] for m in world.machines}, strategy)
        self.phase = ["select"] * n
        self.target: list[str | None] = [None] * n
        self.goal: list[tuple[float, float, float] | None] = [None] * n
        self.path: list[list[tuple[float, float]] | None] = [None] * n
        self.wp = [1] * n
        self.replan_at = [0.0] * n
        self.next_select = [0.0] * n
        self.last_readings: list[dict | None] = [None] * n  # per-robot last perception (for the dashboard)
        self.per_robot = {k: 0 for k in range(n)}
        self.per_machine = {m.name: 0 for m in world.machines}
        self.events: list[tuple] = []
        self.min_sep = math.inf

    @property
    def delivered(self) -> int:
        return sum(self.per_robot.values())

    def _start_leg(self, k: int, goal_pose: tuple[float, float, float]) -> None:
        self.goal[k], self.path[k], self.wp[k], self.replan_at[k] = goal_pose, None, 1, 0.0

    def _drive(self, k: int) -> None:
        """One stored-path follower tick: (re)plan when stale, advance waypoints, stop on the contact
        backstop if the next step would touch another robot. Drives via the (swappable) ``velocity_fn``."""
        world = self.world
        pose = world.robot_pose(k)
        if self.path[k] is None or world.time >= self.replan_at[k]:
            fresh = _plan_leg(world, k, self.goal[k][:2])
            if fresh is not None:
                self.path[k], self.wp[k] = fresh, 1
            self.replan_at[k] = world.time + _REPLAN_INTERVAL
        if self.path[k] is None:
            world.stop_base(robot=k)
            return
        pth = self.path[k]
        while self.wp[k] < len(pth) - 1 and math.dist(pose[:2], pth[self.wp[k]]) <= _WAYPOINT_TOL:
            self.wp[k] += 1
        if self.wp[k] >= len(pth) - 1:
            tgt = self.goal[k]
        else:
            wx, wy = pth[self.wp[k]]
            tgt = (wx, wy, math.atan2(wy - pose[1], wx - pose[0]))
        if _would_collide(world, k, tgt[:2]):
            world.stop_base(robot=k)
            return
        vx, vy, omega = self.velocity_fn(pose, tgt, self.gains)
        rvx, rvy, romega = wheels_to_body(body_to_wheels(vx, vy, omega, self.geom), self.geom)
        world.command_base_velocity(rvx, rvy, romega, robot=k)

    def _advance(self, k: int) -> None:
        world, robot = self.world, self.robots[k]
        ph, pose = self.phase[k], world.robot_pose(k)
        if ph == "select":
            world.stop_base(robot=k)
            if world.time < self.next_select[k]:
                return
            self.next_select[k] = world.time + self.select_interval
            readings = self.perceive_fn(k)
            self.last_readings[k] = readings
            done = [name for name, s in readings.items() if s.machine_state is MachineState.DONE]
            self.manager.observe(done, world.time)
            machine = self.manager.assign(k, pose[:2], done)
            if machine is not None:
                self.target[k], self.phase[k] = machine, "to_machine"
                self._start_leg(k, robot.tending_pose(machine))
        elif ph in ("to_machine", "to_table"):
            dist, head = pose_error(pose, self.goal[k])
            if dist <= _ARRIVE_POS and head <= _ARRIVE_HEADING:
                world.stop_base(robot=k)
                self.phase[k] = "pick" if ph == "to_machine" else "place"
            else:
                self._drive(k)
        elif ph == "pick":
            world.stop_base(robot=k)                      # hold parked (zero any residual drive velocity)
            world.set_arm_target(k, ARM_REACH)            # reach into the machine
            if not world.arm_at(k, ARM_REACH):
                return
            try:
                robot.pick(self.target[k])                # grasp (part now rides the gripper)
            except PreconditionError:
                self.manager.release(self.target[k])
                world.set_arm_target(k, ARM_REST)
                self.phase[k] = "select"
                return
            self.manager.release(self.target[k])
            world.set_arm_target(k, ARM_REST)             # retract to carry pose
            self.phase[k] = "pick_lift"
        elif ph == "pick_lift":
            world.stop_base(robot=k)
            if world.arm_at(k, ARM_REST):                 # arm tucked → drive off
                self.phase[k] = "to_table"
                self._start_leg(k, _table_slot(world, k))
        elif ph == "place":
            world.stop_base(robot=k)
            world.set_arm_target(k, ARM_REACH)            # extend over the table
            if not world.arm_at(k, ARM_REACH):
                return
            part = robot.place(self.goal[k])              # release (lands on the stack)
            self.per_robot[k] += 1
            self.per_machine[self.target[k]] += 1
            self.events.append((k, self.target[k], part, round(world.time, 2)))
            if self.on_event is not None:
                self.on_event(k, self.target[k], part, world.time)
            world.set_arm_target(k, ARM_REST)
            self.phase[k] = "place_lift"
        elif ph == "place_lift":
            world.stop_base(robot=k)
            if world.arm_at(k, ARM_REST):
                self.phase[k] = "select"

    def step(self) -> None:
        """Advance every robot one tick, step the world once, and update the closest-approach metric."""
        for k in range(self.world.n_robots):
            self._advance(k)
        self.world.step()
        self.min_sep = min(self.min_sep, _min_separation(self.world))

    def report(self, target_parts: int) -> FleetReport:
        return FleetReport(
            strategy=self.strategy_name, parts_delivered=self.delivered,
            completed=self.delivered >= target_parts, sim_seconds=self.world.time,
            per_robot=dict(self.per_robot), per_machine=dict(self.per_machine),
            min_robot_separation=self.min_sep, events=list(self.events),
        )


def run_fleet(
    world: World,
    perceive_fn: PerceiveFn,
    *,
    strategy: Strategy,
    strategy_name: str = "",
    target_parts: int = 6,
    robots: list[Robot] | None = None,
    gains: PoseGains = PoseGains(),
    geom: MecanumGeometry = MecanumGeometry(),
    max_sim_seconds: float = 600.0,
    select_interval: float = 0.3,
    on_event: Callable[[int, str, str, float], None] | None = None,
) -> FleetReport:
    """Run the fleet unattended until ``target_parts`` are delivered (or the sim-time budget elapses)."""
    ctrl = FleetController(
        world, perceive_fn, strategy=strategy, strategy_name=strategy_name, robots=robots,
        gains=gains, geom=geom, select_interval=select_interval, on_event=on_event,
    )
    while ctrl.delivered < target_parts and world.time < max_sim_seconds:
        ctrl.step()
    return ctrl.report(target_parts)
