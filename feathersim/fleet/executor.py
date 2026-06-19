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
from feathersim.sim.world import ROBOT_RADIUS, TIMESTEP, World

# Arrival slack (match the SDK's "parked" tolerance so pick/place preconditions pass on arrival).
_ARRIVE_POS, _ARRIVE_HEADING = 0.06, 0.15
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
    n = world.n_robots
    robots = robots or [Robot(world, robot_id=k, gains=gains, geom=geom) for k in range(n)]
    manager = FleetManager({m.name: world.fixtures[m.name] for m in world.machines}, strategy)

    phase = ["select"] * n
    target: list[str | None] = [None] * n
    goal: list[tuple[float, float, float] | None] = [None] * n
    path: list[list[tuple[float, float]] | None] = [None] * n
    wp = [1] * n
    replan_at = [0.0] * n
    next_select = [0.0] * n
    per_robot = {k: 0 for k in range(n)}
    per_machine = {m.name: 0 for m in world.machines}
    events: list[tuple] = []
    min_sep = math.inf

    def start_leg(k: int, goal_pose: tuple[float, float, float]) -> None:
        goal[k], path[k], wp[k], replan_at[k] = goal_pose, None, 1, 0.0  # plan lazily on first drive tick

    def drive(k: int) -> None:
        """One stored-path follower tick: (re)plan when stale, advance waypoints, stop on the symmetric
        contact backstop if the next step would touch another robot."""
        pose = world.robot_pose(k)
        if path[k] is None or world.time >= replan_at[k]:
            fresh = _plan_leg(world, k, goal[k][:2])
            if fresh is not None:
                path[k], wp[k] = fresh, 1
            replan_at[k] = world.time + _REPLAN_INTERVAL
        if path[k] is None:
            world.stop_base(robot=k)  # no route yet — hold
            return
        pth = path[k]
        while wp[k] < len(pth) - 1 and math.dist(pose[:2], pth[wp[k]]) <= _WAYPOINT_TOL:
            wp[k] += 1
        if wp[k] >= len(pth) - 1:
            tgt = goal[k]  # final leg — settle on the full goal pose (correct heading for pick/place)
        else:
            wx, wy = pth[wp[k]]
            tgt = (wx, wy, math.atan2(wy - pose[1], wx - pose[0]))
        if _would_collide(world, k, tgt[:2]):
            world.stop_base(robot=k)  # contact backstop — predicted step would touch another robot
            return
        vx, vy, omega = velocity_command(pose, tgt, gains)
        rvx, rvy, romega = wheels_to_body(body_to_wheels(vx, vy, omega, geom), geom)
        world.command_base_velocity(rvx, rvy, romega, robot=k)

    def advance(k: int) -> None:
        ph, robot, pose = phase[k], robots[k], world.robot_pose(k)
        if ph == "select":
            world.stop_base(robot=k)
            if world.time < next_select[k]:
                return
            next_select[k] = world.time + select_interval
            readings = perceive_fn(k)
            done = [name for name, s in readings.items() if s.machine_state is MachineState.DONE]
            manager.observe(done, world.time)
            machine = manager.assign(k, pose[:2], done)
            if machine is not None:
                target[k], phase[k] = machine, "to_machine"
                start_leg(k, robot.tending_pose(machine))
        elif ph in ("to_machine", "to_table"):
            dist, head = pose_error(pose, goal[k])
            if dist <= _ARRIVE_POS and head <= _ARRIVE_HEADING:
                world.stop_base(robot=k)
                phase[k] = "pick" if ph == "to_machine" else "place"
            else:
                drive(k)
        elif ph == "pick":
            try:
                robot.pick(target[k])
            except PreconditionError:
                manager.release(target[k])  # perception false positive — give the machine back
                phase[k] = "select"
                return
            manager.release(target[k])  # part is unloaded; machine is free to be re-tended by anyone
            phase[k] = "to_table"
            start_leg(k, _table_slot(world, k))
        elif ph == "place":
            part = robot.place(goal[k])  # deposit at this robot's staggered table slot
            per_robot[k] += 1
            per_machine[target[k]] += 1
            events.append((k, target[k], part, round(world.time, 2)))
            if on_event is not None:
                on_event(k, target[k], part, world.time)
            phase[k] = "select"

    while sum(per_robot.values()) < target_parts and world.time < max_sim_seconds:
        for k in range(n):
            advance(k)
        world.step()
        min_sep = min(min_sep, _min_separation(world))

    delivered = sum(per_robot.values())
    return FleetReport(
        strategy=strategy_name,
        parts_delivered=delivered,
        completed=delivered >= target_parts,
        sim_seconds=world.time,
        per_robot=per_robot,
        per_machine=per_machine,
        min_robot_separation=min_sep,
        events=events,
    )
