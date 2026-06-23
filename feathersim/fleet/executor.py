"""The multi-robot fleet executor — tick-based, collision-avoiding, unattended. [v2 Phase C]

The blocking Phase-5 `run_autonomy` can't drive N robots that share one `mj_step`, so the fleet is
re-expressed as a tick loop: every tick each robot advances its own state machine (`select → to_machine →
pick → to_table → place`) by one step, then the world steps once. Coordination:

- **Task allocation** via :class:`FleetManager` — a machine is locked to one robot, never double-booked.
- **Collision avoidance** is two clean layers: **A*** plans each leg around the *static* world (machines,
  table, pillars), and **ORCA** (:mod:`feathersim.fleet.orca`, reciprocal velocity obstacles) makes the
  per-tick velocity collision-free against the *other robots*. ORCA is smooth, reciprocal, and — unlike the
  old symmetric backstop — deadlock-free; a small symmetry-breaking bias on the preferred velocity resolves
  perfectly-symmetric encounters. Because planning ignores the movable robots, there is also no planning
  deadly-embrace (a robot is never left path-less by another robot parking on its goal).
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
from feathersim.fleet.orca import ORCAAgent, new_velocity
from feathersim.fleet.scheduling import Strategy
from feathersim.kinematics.holonomic import MecanumGeometry, body_to_wheels, wheels_to_body
from feathersim.perception.infer import PerceivedState
from feathersim.planning import plan_path
from feathersim.sdk.robot import PreconditionError, Robot
from feathersim.sim.machine import MachineState
from feathersim.sim.world import ARM_REACH, ARM_REST, ROBOT_RADIUS, TIMESTEP, World

# Arrival slack — kept strictly *tighter* than the SDK's `_at` "parked" tolerance (0.06 / 0.15) so a robot
# the SM calls "arrived" reliably passes pick/place's precondition with boundary margin (rather than landing
# exactly on the SDK threshold). The arm itself is inert (gravcomp + qvel-zeroed), so it adds no drift.
_ARRIVE_POS, _ARRIVE_HEADING = 0.05, 0.08
_BODY_CLEARANCE = 2.0 * ROBOT_RADIUS   # two bodies overlap if centres are closer than the sum of radii (0.4)
_WAYPOINT_TOL = 0.12          # advance to the next stored waypoint within this distance
_SLOT_SPACING = 0.55          # x spacing of per-robot delivery slots — just past 2·_ORCA_RADIUS (0.54), so
                              # two robots at adjacent slots are mutually reachable (tight: 0.01 m of slack;
                              # if _ORCA_RADIUS grows, widen this too or adjacent slots become unreachable)
# ORCA (robot↔robot avoidance). The avoidance disc is inflated past the true body radius so ORCA keeps
# *centres* ≥ 2·_ORCA_RADIUS = 0.54 apart → bodies always clear the 0.40 contact threshold with margin.
_ORCA_RADIUS = ROBOT_RADIUS + 0.07
_ORCA_TAU = 2.0               # avoidance time horizon (s) — how far ahead ORCA looks
_ORCA_BIAS = 0.06             # small fixed rotation (rad) of the preferred velocity; breaks perfectly
                              # symmetric encounters that pure ORCA would stall on (and swirls clusters)


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
    """Plan robot ``k``'s path to ``goal_xy`` around the **static** world only (machines, table, pillars).

    Other robots are deliberately *not* obstacles here — ORCA handles them reactively at velocity level. So
    a path always exists (a robot is never left ``path=None`` because a peer parked on its route), which is
    exactly what removes the planning deadly-embrace the old robot-as-obstacle planner could hit."""
    return plan_path(world.occupancy_grid(), world.robot_pose(k)[:2], goal_xy)


def _orca_neighbors(ctrl: FleetController, k: int) -> list[ORCAAgent]:
    """Every other robot as an ORCA disc. A robot that is itself driving (running ORCA) is a *reciprocal*
    peer (50/50 avoidance); a parked one is a non-reciprocal static obstacle the mover fully avoids."""
    world = ctrl.world
    return [
        ORCAAgent(world.robot_pose(j)[:2], ctrl.vel[j], _ORCA_RADIUS, reciprocal=ctrl._driving(j))
        for j in range(world.n_robots) if j != k
    ]


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
        self.vel: list[tuple[float, float]] = [(0.0, 0.0)] * n  # each robot's world-frame velocity (for ORCA)
        self.next_select = [0.0] * n
        self.last_readings: list[dict | None] = [None] * n  # per-robot last perception (for the dashboard)
        self.per_robot = dict.fromkeys(range(n), 0)
        self.per_machine = {m.name: 0 for m in world.machines}
        self.events: list[tuple] = []
        self.min_sep = math.inf

    @property
    def delivered(self) -> int:
        return sum(self.per_robot.values())

    def _driving(self, k: int) -> bool:
        """True while robot ``k`` is in a transit phase (so it's actively running ORCA — a reciprocal peer)."""
        return self.phase[k] in ("to_machine", "to_table")

    def _start_leg(self, k: int, goal_pose: tuple[float, float, float]) -> None:
        self.goal[k], self.path[k], self.wp[k] = goal_pose, None, 1

    def _drive(self, k: int) -> None:
        """One transit tick: follow the A* path's current waypoint as a *preferred* velocity, then let ORCA
        make it collision-free against the other robots. The (swappable) ``velocity_fn`` — hand-coded or the
        learned BC policy — still produces the preferred motion and the heading; ORCA only adjusts where the
        robot actually goes to avoid peers. Drives the base through the mecanum IK→FK (load-bearing)."""
        world = self.world
        pose = world.robot_pose(k)
        if self.path[k] is None:
            self.path[k], self.wp[k] = _plan_leg(world, k, self.goal[k][:2]), 1
        if self.path[k] is None:
            world.stop_base(robot=k)
            self.vel[k] = (0.0, 0.0)
            return
        pth = self.path[k]
        while self.wp[k] < len(pth) - 1 and math.dist(pose[:2], pth[self.wp[k]]) <= _WAYPOINT_TOL:
            self.wp[k] += 1
        wx, wy = self.goal[k][:2] if self.wp[k] >= len(pth) - 1 else pth[self.wp[k]]
        target = (wx, wy, self.goal[k][2])

        # Preferred body twist from the controller → preferred world velocity (with a symmetry-breaking bias).
        vx_b, vy_b, omega = self.velocity_fn(pose, target, self.gains)
        yaw = pose[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        pref_world = (cy * vx_b - sy * vy_b, sy * vx_b + cy * vy_b)
        # Per-robot bias (distinct rotation per id), not a shared constant: a shared constant rotates a
        # perfectly symmetric ring rigidly (it would orbit forever); distinct rotations break the ring.
        bias = _ORCA_BIAS * (1.0 + 0.5 * k)
        cb, sb = math.cos(bias), math.sin(bias)
        pref_world = (pref_world[0] * cb - pref_world[1] * sb, pref_world[0] * sb + pref_world[1] * cb)

        # ORCA → collision-free world velocity → back to a body twist (heading kept from the controller).
        me = ORCAAgent(pose[:2], self.vel[k], _ORCA_RADIUS)
        v = new_velocity(me, _orca_neighbors(self, k), pref_world, self.gains.max_linear, _ORCA_TAU, TIMESTEP)
        self.vel[k] = v
        bvx, bvy = cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]
        rvx, rvy, romega = wheels_to_body(body_to_wheels(bvx, bvy, omega, self.geom), self.geom)
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
        for k in range(self.world.n_robots):     # a parked robot is a stationary ORCA obstacle (vel 0)
            if not self._driving(k):
                self.vel[k] = (0.0, 0.0)
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
