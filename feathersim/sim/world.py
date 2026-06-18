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


def build_mjcf(n_machines: int) -> str:
    """Return an MJCF string for a world with ``n_machines`` machine bodies."""
    light_rgba = _rgba_str(STATE_LIGHT[MachineState.IDLE])  # initial color; mutated per state at runtime
    part_rgba = _rgba_str(_PART_RGBA)
    bodies = []
    for i, (mx, my) in enumerate(_machine_positions(n_machines)):
        r, g, b = _MACHINE_COLORS[i % len(_MACHINE_COLORS)]
        bodies.append(
            f"""
    <body name="machine_{i}" pos="{mx} {my} 0.3">
      <geom type="box" size="0.3 0.3 0.3" rgba="{r} {g} {b} 1"/>
      <geom name="door_{i}" type="box" pos="0 -0.31 -0.05" size="0.25 0.02 0.22"
            rgba="0.12 0.12 0.14 1"/>
      <geom name="light_{i}" type="sphere" pos="0 -0.22 0.44" size="0.13"
            contype="0" conaffinity="0" rgba="{light_rgba}"/>
      <geom name="part_{i}" type="box" pos="0 -0.42 0.02" size="0.13 0.13 0.11"
            contype="0" conaffinity="0" rgba="{part_rgba}"/>
    </body>"""
        )
    return f"""
<mujoco model="feathersim">
  <option timestep="{TIMESTEP}" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.82 0.82 0.86 1"/>
    <body name="robot" pos="0 0 0.15">
      <joint name="base_x" type="slide" axis="1 0 0"/>
      <joint name="base_y" type="slide" axis="0 1 0"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.2 0.12" rgba="0.20 0.50 0.90 1"/>
      <geom name="heading" type="box" pos="0.16 0 0.12" size="0.06 0.05 0.02"
            rgba="0.95 0.85 0.20 1"/>
    </body>
    <body name="table" pos="{TABLE_XY[0]} {TABLE_XY[1]} 0.2">
      <geom type="box" size="0.5 0.3 0.2" rgba="0.60 0.42 0.24 1"/>
    </body>{"".join(bodies)}
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
    model: mujoco.MjModel = field(init=False, repr=False)
    data: mujoco.MjData = field(init=False, repr=False)
    machines: list[Machine] = field(init=False)
    fixtures: dict[str, tuple[float, float]] = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.n_machines <= 3:
            raise ValueError(f"n_machines must be 1–3, got {self.n_machines}")
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.n_machines))
        self.data = mujoco.MjData(self.model)

        # qpos / qvel (dof) addresses for the base joints — looked up by name, no layout assumption.
        base_joints = ("base_x", "base_y", "base_yaw")
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in base_joints]
        self._pose_adr = [int(self.model.jnt_qposadr[j]) for j in jids]
        self._dof_adr = [int(self.model.jnt_dofadr[j]) for j in jids]

        # geom ids for each machine's status light and bed part (mutated to render visual state).
        self._light_gid = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"light_{i}")
            for i in range(self.n_machines)
        ]
        self._part_gid = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"part_{i}")
            for i in range(self.n_machines)
        ]

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

        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        """Current sim time in seconds."""
        return float(self.data.time)

    def step(self) -> None:
        """Advance physics one timestep and tick every machine FSM to the new sim time.

        The FSM clock is read *after* the step, so the first update sees ``time == timestep``,
        not 0 — immaterial since ``timestep`` (0.01s) ≪ any machine's idle/cycle time.
        """
        mujoco.mj_step(self.model, self.data)
        for m in self.machines:
            m.update(self.time)

    def robot_pose(self) -> tuple[float, float, float]:
        """Return the base pose ``(x, y, yaw)`` from the planar joints."""
        x, y, yaw = (float(self.data.qpos[a]) for a in self._pose_adr)
        return (x, y, yaw)

    def set_base_pose(self, x: float, y: float, yaw: float) -> None:
        """Teleport the base to ``(x, y, yaw)`` and halt it (for resets / test setup)."""
        ax, ay, ayaw = self._pose_adr
        self.data.qpos[ax] = x
        self.data.qpos[ay] = y
        self.data.qpos[ayaw] = yaw
        self.stop_base()
        mujoco.mj_forward(self.model, self.data)

    def command_base_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Command a body-frame twist (x-forward, y-left, omega CCW).

        Converts to world-frame joint velocities using the current yaw and writes them to the base
        DOFs; the next :meth:`step` integrates the motion. Kinematic control — it does not respect
        contacts, so it's for free-space navigation only.
        """
        _, _, yaw = self.robot_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        world_vx = c * vx - s * vy
        world_vy = s * vx + c * vy
        ax, ay, ayaw = self._dof_adr
        self.data.qvel[ax] = world_vx
        self.data.qvel[ay] = world_vy
        self.data.qvel[ayaw] = omega

    def stop_base(self) -> None:
        """Zero the base joint velocities (halt)."""
        for a in self._dof_adr:
            self.data.qvel[a] = 0.0

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
