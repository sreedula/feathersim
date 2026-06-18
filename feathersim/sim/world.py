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

# Distinct machine colors (RGB 0–1) so renders/print output are easy to tell apart.
_MACHINE_COLORS = [(0.85, 0.25, 0.25), (0.25, 0.65, 0.30), (0.90, 0.70, 0.20)]


def _machine_positions(n: int) -> list[tuple[float, float]]:
    """Lay ``n`` machines in a row along x at y=+1.5, facing the robot at the origin."""
    if n == 1:
        return [(0.0, 1.5)]
    xs = np.linspace(-1.0, 1.0, n)
    return [(float(x), 1.5) for x in xs]


def build_mjcf(n_machines: int) -> str:
    """Return an MJCF string for a world with ``n_machines`` machine bodies."""
    bodies = []
    for i, (mx, my) in enumerate(_machine_positions(n_machines)):
        r, g, b = _MACHINE_COLORS[i % len(_MACHINE_COLORS)]
        bodies.append(
            f"""
    <body name="machine_{i}" pos="{mx} {my} 0.3">
      <geom type="box" size="0.3 0.3 0.3" rgba="{r} {g} {b} 1"/>
      <geom name="door_{i}" type="box" pos="0 -0.31 -0.05" size="0.25 0.02 0.22"
            rgba="0.12 0.12 0.14 1"/>
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
    <body name="table" pos="0 -1.5 0.2">
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
