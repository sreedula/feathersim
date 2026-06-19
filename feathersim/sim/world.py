"""MuJoCo sim world: floor, holonomic robot base, parts table, 2–3 machines. [Phase 1]

Builds an MJCF model programmatically (so machine count is configurable), wraps ``MjModel`` /
``MjData``, and ticks the pure machine FSMs each physics step. Headless and deterministic given a
seed: the seed assigns each machine its idle/cycle durations, so same seed → identical state trace.

The robot base has planar x/y/yaw joints. Phase 2 drives them via :meth:`World.command_base_velocity`
(kinematic velocity control: write the base joint velocities, then step). Unactuated, the base stays
put.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from feathersim.planning.occupancy import OccupancyGrid, Rect, build_grid
from feathersim.sim.machine import Machine, MachineState

TIMESTEP = 0.01  # sim seconds per step (100 Hz) — fine resolution, fast headless stepping

MACHINE_Y = 1.5          # machines sit in a row at this y, doors facing the origin (−y)
TABLE_XY = (0.0, -1.5)   # output/parts table, opposite the machines

# Distinct machine colors (RGB 0–1) so renders/print output are easy to tell apart.
_MACHINE_COLORS = [(0.85, 0.25, 0.25), (0.25, 0.65, 0.30), (0.90, 0.70, 0.20)]

# Status-light color per machine state — the visual cue the perception model learns to read.
STATE_LIGHT = {
    MachineState.IDLE: (0.55, 0.55, 0.58),     # gray  — idle/empty
    MachineState.RUNNING: (0.95, 0.65, 0.10),  # amber — machining
    MachineState.DONE: (0.15, 0.85, 0.25),     # green — finished part ready
}
_PART_RGBA = (0.15, 0.35, 0.95, 1.0)  # vivid blue — high contrast vs floor, door & machines (alpha 0 hides)

# Domain-randomization occluder (v2 Phase A): a small box in front of the status light, hidden by
# default (alpha 0). The dataset generator places & shows it to partially block the light.
_OCCLUDER_BASE = (0.0, -0.34, 0.44)   # local pos in the machine body, between camera and status light
_OCCLUDER_RGB = (0.18, 0.18, 0.20)    # neutral dark — clearly an occluder, never mistaken for a state color

ROBOT_RADIUS = 0.2                        # base cylinder radius — the occupancy-grid inflation
# Extra inflation for the static pillars beyond the robot radius. The waypoint follower drives a
# P-controlled curve toward each waypoint (only the straight *segments* are checked free), so it bows
# into the inflation band on turns; this margin, with the tightened follower waypoint tolerance, keeps
# real body clearance above the robot radius on *every* leg the loop drives — the central machine_1↔table
# corridor bows worst at ~0.29 m vs the 0.2 m radius (see tests/test_planning.py). Applied only to
# obstacles, not machines/table — those are inflated by the radius alone so tending poses stay reachable.
OBSTACLE_CLEARANCE = 0.08
GRID_BOUNDS = (-2.0, -2.0, 2.0, 2.0)      # (xmin, ymin, xmax, ymax) the planner rasterizes over
# Static obstacles (v2 Phase B): pillars on the table↔machine_2 and table↔machine_0 diagonals so the
# planner must route around them, kept clear of every tending pose (body clearance is then governed by
# grid inflation, not by a goal sitting next to a pillar) and spaced so the central x≈0 corridor to
# machine_1 stays open. Non-colliding (kinematic base) — avoidance is planning-based.
_OBSTACLE_POSITIONS = [(0.62, 0.0), (-0.62, 0.0)]
_OBSTACLE_HALF = 0.22

# Multi-robot fleet (v2 Phase C). robot_0 is blue (matches v1); extras get distinct colors.
_ROBOT_COLORS = [(0.20, 0.50, 0.90), (0.92, 0.45, 0.15), (0.45, 0.80, 0.30)]

# Arm shoulder-pitch poses (v3): the forearm points +x at angle 0. REST tucks it up (carry); REACH swings
# it forward+down to grasp from a machine bed or place on the table. Animated kinematically in step().
ARM_REST, ARM_REACH = -1.4, -0.15
ARM_RATE = 4.5  # rad/s the arm slews toward its target


def _robot_starts(n_robots: int) -> list[tuple[float, float]]:
    """Start positions for the bases. One robot starts at the origin (v1); a fleet spreads along x on the
    table side at y=-0.85 — clear of the pillars at (±0.62, 0) *and* their grid inflation, so every base
    starts on a free cell (a start inside an inflated obstacle would make its first plan fail)."""
    if n_robots == 1:
        return [(0.0, 0.0)]
    xs = np.linspace(-0.45, 0.45, n_robots)
    return [(float(x), -0.85) for x in xs]


def _robot_mjcf(n_robots: int) -> str:
    """MJCF for ``n_robots`` holonomic bases, each with its own planar joints, color, and heading marker.

    Every body is homed at the origin so the slide-joint ``qpos`` *is* the world position; start poses
    are written into ``qpos`` in ``World.__post_init__`` (see :func:`_robot_starts`)."""
    out = []
    for k in range(n_robots):
        r, g, b = _ROBOT_COLORS[k % len(_ROBOT_COLORS)]
        out.append(
            f"""
    <body name="robot_{k}" pos="0 0 0.15">
      <joint name="base_x_{k}" type="slide" axis="1 0 0"/>
      <joint name="base_y_{k}" type="slide" axis="0 1 0"/>
      <joint name="base_yaw_{k}" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.2 0.12" material="robotmat_{k}"/>
      <geom type="cylinder" pos="0 0 0.12" size="0.13 0.03" material="robotmat_{k}"/>
      <geom name="heading_{k}" type="box" pos="0.16 0 0.12" size="0.06 0.05 0.02"
            rgba="0.97 0.85 0.15 1"/>
      <camera name="robotcam_{k}" pos="-0.02 0 0.42" xyaxes="0 -1 0 0.34 0 0.94" fovy="74"/>
      <geom type="box" pos="0 0 0.19" size="0.05 0.07 0.05" contype="0" conaffinity="0" material="armmat"/>
      <body name="arm_{k}" pos="0 0 0.23" gravcomp="1">
        <joint name="arm_{k}" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.30 0 0" size="0.032" contype="0" conaffinity="0" material="armmat"/>
        <geom type="box" pos="0.33 0 0" size="0.03 0.055 0.04" contype="0" conaffinity="0" material="grippermat"/>
        <geom name="carried_{k}" type="box" pos="0.37 0 0" size="0.055 0.055 0.05"
              contype="0" conaffinity="0" rgba="0.15 0.35 0.95 0"/>
      </body>
    </body>"""
        )
    return "".join(out)


def _machine_positions(n: int) -> list[tuple[float, float]]:
    """Lay ``n`` machines in a row along x at ``MACHINE_Y``, facing the robot at the origin."""
    if n == 1:
        return [(0.0, MACHINE_Y)]
    xs = np.linspace(-1.0, 1.0, n)
    return [(float(x), MACHINE_Y) for x in xs]


def _rgba_str(rgb_or_rgba: tuple[float, ...]) -> str:
    """Format an RGB(A) tuple as an MJCF ``rgba`` attribute value (alpha defaults to 1)."""
    vals = tuple(rgb_or_rgba) + (() if len(rgb_or_rgba) == 4 else (1.0,))
    return " ".join(f"{v:g}" for v in vals)


def _obstacle_mjcf(n_obstacles: int) -> str:
    """MJCF for ``n_obstacles`` static pillars (non-colliding visual markers the planner routes around)."""
    out = []
    for i, (ox, oy) in enumerate(_OBSTACLE_POSITIONS[:n_obstacles]):
        out.append(
            f"""
    <body name="obstacle_{i}" pos="{ox} {oy} 0.35">
      <geom type="box" size="{_OBSTACLE_HALF} {_OBSTACLE_HALF} 0.35"
            contype="0" conaffinity="0" material="pillarmat"/>
    </body>"""
        )
    return "".join(out)


def _assets_mjcf(n_machines: int, n_robots: int) -> str:
    """Cinematic visual setup: shadows + a gradient skybox, a textured floor, and glossy per-body
    materials. Purely visual (the status-light/part colors that perception reads are unchanged)."""
    mats = []
    for i in range(n_machines):
        r, g, b = _MACHINE_COLORS[i % len(_MACHINE_COLORS)]
        mats.append(f'<material name="machmat_{i}" rgba="{r} {g} {b} 1" specular="0.5" shininess="0.55" reflectance="0.12"/>')
    for k in range(n_robots):
        r, g, b = _ROBOT_COLORS[k % len(_ROBOT_COLORS)]
        mats.append(f'<material name="robotmat_{k}" rgba="{r} {g} {b} 1" specular="0.75" shininess="0.85" reflectance="0.2"/>')
    materials = "\n    ".join(mats)
    return f"""
  <visual>
    <global offwidth="1280" offheight="1280"/>
    <headlight diffuse="0.3 0.3 0.33" ambient="0.33 0.33 0.35" specular="0.2 0.2 0.2"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map force="0.1" zfar="40"/>
    <rgba haze="0.10 0.13 0.18 1"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.34 0.45 0.58" rgb2="0.03 0.05 0.09" width="512" height="512"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.15 0.17 0.21" rgb2="0.21 0.23 0.27" width="512" height="512"/>
    <material name="floormat" texture="floortex" texrepeat="14 14" specular="0.2" shininess="0.3" reflectance="0.12"/>
    <material name="tablemat" rgba="0.50 0.35 0.20 1" specular="0.3" shininess="0.4"/>
    <material name="doormat" rgba="0.09 0.09 0.12 1" specular="0.6" shininess="0.85"/>
    <material name="pillarmat" rgba="0.90 0.42 0.13 1" specular="0.3" shininess="0.5"/>
    <material name="armmat" rgba="0.78 0.80 0.84 1" specular="0.85" shininess="0.9" reflectance="0.25"/>
    <material name="grippermat" rgba="0.18 0.19 0.22 1" specular="0.7" shininess="0.85"/>
    {materials}
  </asset>"""


# Output-table stack slots (relative to the table body): a 4×3 grid that fills as parts are delivered.
STACK_SLOTS = [(x, y) for y in (-0.15, 0.0, 0.15) for x in (-0.3, -0.1, 0.1, 0.3)]


def _stack_mjcf() -> str:
    """Delivered-part slots on the output table (alpha 0 until ``set_delivered_count`` reveals them)."""
    return "".join(
        f"""
      <geom name="stack_{j}" type="box" pos="{sx} {sy} 0.26" size="0.06 0.06 0.05"
            contype="0" conaffinity="0" rgba="0.15 0.35 0.95 0"/>"""
        for j, (sx, sy) in enumerate(STACK_SLOTS)
    )


def build_mjcf(n_machines: int, n_obstacles: int = 0, n_robots: int = 1) -> str:
    """Return an MJCF string for a world with ``n_machines`` machines, ``n_obstacles`` pillars, ``n_robots`` bases."""
    light_rgba = _rgba_str(STATE_LIGHT[MachineState.IDLE])  # initial color; mutated per state at runtime
    part_rgba = _rgba_str(_PART_RGBA)
    bodies = []
    for i, (mx, my) in enumerate(_machine_positions(n_machines)):
        r, g, b = _MACHINE_COLORS[i % len(_MACHINE_COLORS)]
        bodies.append(
            f"""
    <body name="machine_{i}" pos="{mx} {my} 0.3">
      <geom type="box" size="0.3 0.3 0.3" material="machmat_{i}"/>
      <geom name="door_{i}" type="box" pos="0 -0.31 -0.05" size="0.25 0.02 0.22" material="doormat"/>
      <geom name="light_{i}" type="sphere" pos="0 -0.22 0.44" size="0.13"
            contype="0" conaffinity="0" rgba="{light_rgba}"/>
      <geom name="part_{i}" type="box" pos="0 -0.42 0.02" size="0.13 0.13 0.11"
            contype="0" conaffinity="0" rgba="{part_rgba}"/>
      <geom name="occluder_{i}" type="box" pos="{_OCCLUDER_BASE[0]} {_OCCLUDER_BASE[1]} {_OCCLUDER_BASE[2]}"
            size="0.05 0.02 0.05" contype="0" conaffinity="0"
            rgba="{_OCCLUDER_RGB[0]} {_OCCLUDER_RGB[1]} {_OCCLUDER_RGB[2]} 0"/>
    </body>"""
        )
    return f"""
<mujoco model="feathersim">
  <option timestep="{TIMESTEP}" gravity="0 0 -9.81"/>{_assets_mjcf(n_machines, n_robots)}
  <worldbody>
    <light pos="1.5 1.0 5" dir="-0.25 -0.2 -1" directional="true" diffuse="0.65 0.65 0.68"
           specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="-2.5 -1.5 3.5" dir="0.4 0.25 -1" diffuse="0.22 0.24 0.30" castshadow="false"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="floormat"/>{_robot_mjcf(n_robots)}
    <body name="table" pos="{TABLE_XY[0]} {TABLE_XY[1]} 0.2">
      <geom type="box" size="0.5 0.3 0.2" material="tablemat"/>{_stack_mjcf()}
    </body>{"".join(bodies)}{_obstacle_mjcf(n_obstacles)}
  </worldbody>
</mujoco>"""


@dataclass
class World:
    """Headless MuJoCo world that also owns the machines' pure FSMs.

    ``n_machines`` must be 1–3. ``seed`` makes the machine timings (and thus the whole state
    trace) reproducible.
    """

    n_machines: int = 3
    seed: int = 0
    n_obstacles: int = 0
    n_robots: int = 1
    model: mujoco.MjModel = field(init=False, repr=False)
    data: mujoco.MjData = field(init=False, repr=False)
    machines: list[Machine] = field(init=False)
    fixtures: dict[str, tuple[float, float]] = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.n_machines <= 3:
            raise ValueError(f"n_machines must be 1–3, got {self.n_machines}")
        if not 0 <= self.n_obstacles <= len(_OBSTACLE_POSITIONS):
            raise ValueError(f"n_obstacles must be 0–{len(_OBSTACLE_POSITIONS)}, got {self.n_obstacles}")
        if not 1 <= self.n_robots <= len(_ROBOT_COLORS):
            raise ValueError(f"n_robots must be 1–{len(_ROBOT_COLORS)}, got {self.n_robots}")
        self.model = mujoco.MjModel.from_xml_string(
            build_mjcf(self.n_machines, self.n_obstacles, self.n_robots)
        )
        self.data = mujoco.MjData(self.model)

        # Per-robot qpos / qvel(dof) addresses for the base joints — looked up by name, no layout assumption.
        self._pose_adr: list[list[int]] = []
        self._dof_adr: list[list[int]] = []
        for k in range(self.n_robots):
            jids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"base_{a}_{k}")
                for a in ("x", "y", "yaw")
            ]
            self._pose_adr.append([int(self.model.jnt_qposadr[j]) for j in jids])
            self._dof_adr.append([int(self.model.jnt_dofadr[j]) for j in jids])

        # geom ids for each machine's status light and bed part (mutated to render visual state).
        self._light_gid = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"light_{i}")
            for i in range(self.n_machines)
        ]
        self._part_gid = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"part_{i}")
            for i in range(self.n_machines)
        ]
        self._occluder_gid = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"occluder_{i}")
            for i in range(self.n_machines)
        ]
        # Per-robot arm shoulder joint (qpos/dof addresses) + slew targets, animated kinematically.
        arm_jids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"arm_{k}")
            for k in range(self.n_robots)
        ]
        self._arm_pose_adr = [int(self.model.jnt_qposadr[j]) for j in arm_jids]
        self._arm_dof_adr = [int(self.model.jnt_dofadr[j]) for j in arm_jids]
        self._arm_target = [ARM_REST] * self.n_robots

        # Part-transport visuals: a carried-part geom per robot (rides its gripper) + table stack slots.
        self._carried_gid = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"carried_{k}")
            for k in range(self.n_robots)
        ]
        self._stack_gid = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"stack_{j}")
            for j in range(len(STACK_SLOTS))
        ]
        self.delivered_total = 0

        # Authored cinematic key light (light 0) — DR jitters *relative* to this and reset_scene restores
        # it, so perception sees a clean baseline equal to the feed's lighting (no train/serve gap).
        self._light0_pos = self.model.light_pos[0].copy()
        self._light0_diffuse = self.model.light_diffuse[0].copy()

        # Seed-driven machine timings: deterministic, but varied so machines don't finish in lockstep.
        rng = np.random.default_rng(self.seed)
        self.machines = [
            Machine(
                name=f"machine_{i}",
                idle_time=round(float(rng.uniform(1.0, 2.5)), 3),
                cycle_time=round(float(rng.uniform(3.0, 6.0)), 3),
            )
            for i in range(self.n_machines)
        ]

        # Ground-truth world positions of the fixtures the SDK navigates to.
        self.fixtures = {m.name: pos for m, pos in zip(self.machines, _machine_positions(self.n_machines))}
        self.fixtures["table"] = TABLE_XY

        # Bodies are homed at the origin; write each base's start position into qpos so qpos == world pos.
        for k, (sx, sy) in enumerate(_robot_starts(self.n_robots)):
            ax, ay, _ = self._pose_adr[k]
            self.data.qpos[ax] = sx
            self.data.qpos[ay] = sy
        for k in range(self.n_robots):           # arms start tucked in the carry pose
            self.data.qpos[self._arm_pose_adr[k]] = ARM_REST

        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        """Current sim time in seconds."""
        return float(self.data.time)

    def step(self) -> None:
        """Advance physics one timestep, slew each arm toward its target, and tick every machine FSM.

        The FSM clock is read *after* the step, so the first update sees ``time == timestep``,
        not 0 — immaterial since ``timestep`` (0.01s) ≪ any machine's idle/cycle time.
        """
        # Arm joints are kinematic: slew qpos toward target and zero qvel before the step. The qvel-zeroing
        # is load-bearing (with the arm body's gravcomp="1") for the arm being dynamically inert — it can't
        # accumulate velocity or perturb the base, so the unactuated base stays exactly put.
        for k in range(self.n_robots):
            adr = self._arm_pose_adr[k]
            delta = self._arm_target[k] - self.data.qpos[adr]
            self.data.qpos[adr] += max(-ARM_RATE * TIMESTEP, min(ARM_RATE * TIMESTEP, delta))
            self.data.qvel[self._arm_dof_adr[k]] = 0.0
        mujoco.mj_step(self.model, self.data)
        for m in self.machines:
            m.update(self.time)

    def robot_pose(self, robot: int = 0) -> tuple[float, float, float]:
        """Return base ``robot``'s pose ``(x, y, yaw)`` from its planar joints."""
        x, y, yaw = (float(self.data.qpos[a]) for a in self._pose_adr[robot])
        return (x, y, yaw)

    def robot_positions(self) -> list[tuple[float, float]]:
        """Every base's ``(x, y)`` position — used for inter-robot collision avoidance and telemetry."""
        return [self.robot_pose(k)[:2] for k in range(self.n_robots)]

    def set_base_pose(self, x: float, y: float, yaw: float, *, robot: int = 0) -> None:
        """Teleport base ``robot`` to ``(x, y, yaw)`` and halt it (for resets / test setup)."""
        ax, ay, ayaw = self._pose_adr[robot]
        self.data.qpos[ax] = x
        self.data.qpos[ay] = y
        self.data.qpos[ayaw] = yaw
        self.stop_base(robot=robot)
        mujoco.mj_forward(self.model, self.data)

    def command_base_velocity(self, vx: float, vy: float, omega: float, *, robot: int = 0) -> None:
        """Command base ``robot`` a body-frame twist (x-forward, y-left, omega CCW).

        Converts to world-frame joint velocities using the current yaw and writes them to the base
        DOFs; the next :meth:`step` integrates the motion. Kinematic control — it does not respect
        contacts, so it's for free-space navigation only.
        """
        _, _, yaw = self.robot_pose(robot)
        c, s = math.cos(yaw), math.sin(yaw)
        world_vx = c * vx - s * vy
        world_vy = s * vx + c * vy
        ax, ay, ayaw = self._dof_adr[robot]
        self.data.qvel[ax] = world_vx
        self.data.qvel[ay] = world_vy
        self.data.qvel[ayaw] = omega

    def stop_base(self, *, robot: int = 0) -> None:
        """Zero base ``robot``'s joint velocities (halt)."""
        for a in self._dof_adr[robot]:
            self.data.qvel[a] = 0.0

    def driver(self, robot: int = 0) -> "_RobotDriver":
        """A single-base :class:`~feathersim.control.go_to_pose.BaseDriver` view bound to ``robot``,
        so the go-to-pose controller and waypoint follower can drive any one base in a fleet."""
        return _RobotDriver(self, robot)

    def states(self) -> dict[str, MachineState]:
        """Ground-truth machine states keyed by name (the perception labels in Phase 4)."""
        return {m.name: m.state for m in self.machines}

    # --- visuals & rendering (perception) ------------------------------------------------

    def set_machine_visual(self, i: int, state: MachineState, part_present: bool) -> None:
        """Set machine ``i``'s status-light color (from ``state``) and bed-part visibility.

        Mutates ``model.geom_rgba`` so the next render reflects it. Used by the dataset generator to
        place the scene in known ground-truth configs, and by :meth:`sync_visuals` for live render.
        """
        self.model.geom_rgba[self._light_gid[i], :3] = STATE_LIGHT[state]
        self.model.geom_rgba[self._part_gid[i], 3] = 1.0 if part_present else 0.0

    def sync_visuals(self) -> None:
        """Drive every machine's visuals from its live FSM state (part present while running/done).

        Note the asymmetry vs. the dataset: at serve time part-present is tied to state here, so live
        configs (e.g. IDLE-with-part) are a *subset* of what the dataset trains on — harmless, and the
        decorrelated training keeps the part head from collapsing into the state head.
        """
        for i, m in enumerate(self.machines):
            self.set_machine_visual(i, m.state, m.state in (MachineState.RUNNING, MachineState.DONE))

    # --- domain randomization (v2 Phase A) ------------------------------------------------

    def randomize_lighting(self, offset_xy: tuple[float, float], diffuse_scale: float) -> None:
        """Jitter the key light *relative to its authored pose*: shift it by ``offset_xy`` and scale its
        diffuse by ``diffuse_scale``. ``((0, 0), 1.0)`` is the authored cinematic light."""
        self.model.light_pos[0, 0] = self._light0_pos[0] + offset_xy[0]
        self.model.light_pos[0, 1] = self._light0_pos[1] + offset_xy[1]
        self.model.light_diffuse[0] = self._light0_diffuse * diffuse_scale

    def set_occluder(self, i: int, *, present: bool, dx: float = 0.0, dz: float = 0.0,
                     size: float = 0.05) -> None:
        """Place machine ``i``'s occluder box (offset ``dx``/``dz`` from its base, half-extent ``size``)
        and show/hide it via alpha. Used by the dataset generator to partially block the status light."""
        gid = self._occluder_gid[i]
        bx, by, bz = _OCCLUDER_BASE
        self.model.geom_pos[gid] = (bx + dx, by, bz + dz)
        self.model.geom_size[gid] = (size, 0.02, size)
        self.model.geom_rgba[gid, 3] = 1.0 if present else 0.0

    def set_arm_target(self, robot: int, angle: float) -> None:
        """Command ``robot``'s arm to slew toward ``angle`` (``ARM_REST`` carry / ``ARM_REACH`` grasp)."""
        self._arm_target[robot] = angle

    def arm_at(self, robot: int, angle: float, tol: float = 0.06) -> bool:
        """True once ``robot``'s arm has reached ``angle`` — the SM gates pick/place on this."""
        return abs(float(self.data.qpos[self._arm_pose_adr[robot]]) - angle) <= tol

    def set_carried(self, robot: int, present: bool) -> None:
        """Show/hide the part riding ``robot``'s gripper (a robot carries a part between pick and place)."""
        self.model.geom_rgba[self._carried_gid[robot], 3] = 1.0 if present else 0.0

    def deposit_part(self) -> None:
        """Reveal the next slot in the output-table stack (a delivered part lands on the table)."""
        if self.delivered_total < len(self._stack_gid):
            self.model.geom_rgba[self._stack_gid[self.delivered_total], 3] = 1.0
        self.delivered_total += 1

    def reset_scene(self) -> None:
        """Restore the authored cinematic key light and hide every occluder — the clean render conditions."""
        self.randomize_lighting((0.0, 0.0), 1.0)
        for i in range(self.n_machines):
            self.set_occluder(i, present=False)

    # --- path planning (v2 Phase B) -------------------------------------------------------

    def _footprints(self, obstacle_margin: float) -> list[Rect]:
        rects = [Rect(mx, my, 0.3, 0.3) for mx, my in _machine_positions(self.n_machines)]
        rects.append(Rect(TABLE_XY[0], TABLE_XY[1], 0.5, 0.3))
        rects += [
            Rect(ox, oy, _OBSTACLE_HALF + obstacle_margin, _OBSTACLE_HALF + obstacle_margin)
            for ox, oy in _OBSTACLE_POSITIONS[: self.n_obstacles]
        ]
        return rects

    def obstacle_rects(self) -> list[Rect]:
        """True obstacle footprints (machines, output table, static pillars) — no clearance margin."""
        return self._footprints(0.0)

    def occupancy_grid(
        self, *, resolution: float = 0.1, inflation: float = ROBOT_RADIUS,
        extra_obstacles: "tuple[Rect, ...] | list[Rect]" = (),
    ) -> OccupancyGrid:
        """Build an occupancy grid of the floor: machines/table inflated by ``inflation`` (the robot
        radius), static pillars by ``inflation + OBSTACLE_CLEARANCE`` (extra room for the follower's
        curve). ``extra_obstacles`` (e.g. other robots, in Phase C) are added as dynamic footprints."""
        rects = self._footprints(OBSTACLE_CLEARANCE) + list(extra_obstacles)
        return build_grid(rects, GRID_BOUNDS, resolution=resolution, inflation=inflation)

    def machine_camera(self, i: int, *, azimuth: float = 90.0, elevation: float = -12.0,
                       distance: float = 1.6) -> mujoco.MjvCamera:
        """A free camera framing machine ``i`` from the front (its door/light side)."""
        mx, my = self.fixtures[f"machine_{i}"]
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (mx, my - 0.1, 0.35)
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.distance = distance
        return cam

    def overview_camera(self, *, azimuth: float = 90.0, elevation: float = -40.0,
                        distance: float = 5.5) -> mujoco.MjvCamera:
        """A free camera framing the whole cell (machines, table, robot) for the dashboard feed."""
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (0.0, 0.0, 0.2)
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.distance = distance
        return cam

    def render(self, renderer: "mujoco.Renderer", camera: "mujoco.MjvCamera") -> np.ndarray:
        """Render the scene from ``camera`` as an ``(H, W, 3)`` uint8 RGB frame."""
        renderer.update_scene(self.data, camera)
        return renderer.render()

    def render_machine(self, renderer: "mujoco.Renderer", i: int,
                       camera: "mujoco.MjvCamera | None" = None) -> np.ndarray:
        """Render machine ``i``'s close-up as an ``(H, W, 3)`` uint8 RGB frame."""
        return self.render(renderer, camera if camera is not None else self.machine_camera(i))


@dataclass
class _RobotDriver:
    """A single-base ``BaseDriver`` view of one robot in a (possibly multi-robot) :class:`World`.

    Lets :func:`drive_to_pose` / :func:`follow_path` (which assume one base) drive base ``robot`` by
    delegating the four driver methods to the indexed :class:`World` API. ``step`` advances the whole
    world (all bases move on the shared ``mj_step``)."""

    world: World
    robot: int = 0

    def robot_pose(self) -> tuple[float, float, float]:
        return self.world.robot_pose(self.robot)

    def command_base_velocity(self, vx: float, vy: float, omega: float) -> None:
        self.world.command_base_velocity(vx, vy, omega, robot=self.robot)

    def stop_base(self) -> None:
        self.world.stop_base(robot=self.robot)

    def step(self) -> None:
        self.world.step()
