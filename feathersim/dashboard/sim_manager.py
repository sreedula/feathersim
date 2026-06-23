"""The dashboard's simulation engine: one threaded sim, autonomy you can preempt. [Phase 6]

MuJoCo's ``MjModel``/``MjData`` aren't thread-safe, so *all* stepping, perception, and rendering happen
on a single background thread owned by :class:`SimManager`. HTTP handlers never touch the sim — the
thread publishes a telemetry snapshot and the latest camera JPEG under a lock, and handlers read those.

Autonomy is re-expressed here as a **tick-based state machine** (``select → to_machine → pick → to_table
→ place``) rather than the blocking Phase-5 :func:`run_autonomy`, so a teleop command can seize control
*mid-skill*: each tick the thread either advances one autonomy step or applies the operator's twist,
depending on the current mode. The machine state survives the manual interlude, so releasing the controls
resumes the skill in progress. Selection still consumes only perception (never ground truth).
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
from dataclasses import dataclass

import mujoco
from PIL import Image

from feathersim.control.go_to_pose import PoseGains, pose_error, velocity_command
from feathersim.kinematics.holonomic import MecanumGeometry, body_to_wheels, wheels_to_body
from feathersim.perception.dataset import IMAGE_SIZE
from feathersim.perception.infer import Perception
from feathersim.sdk.robot import PreconditionError, Robot
from feathersim.sim.machine import MachineState
from feathersim.sim.world import TIMESTEP, World

_log = logging.getLogger(__name__)

AUTO, MANUAL = "auto", "manual"
FEED_SIZE = 480  # overview-feed render resolution (px); perception still renders 64px machine crops

# Arrival slack for the tick-based driver — matches the SDK's "parked" tolerance so a tick-driven
# approach satisfies pick/place preconditions just like the blocking driver would.
_ARRIVE_POS, _ARRIVE_HEADING = 0.06, 0.15

_SKILL_TEXT = {
    "select": "scanning machines",
    "to_machine": "navigating to {target}",
    "pick": "unloading {target}",
    "to_table": "delivering to table",
    "place": "placing part",
}


@dataclass
class Teleop:
    """A body-frame twist command from the operator (x-forward, y-left, CCW)."""

    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


class SimManager:
    """Owns the sim and runs it on a background thread; exposes thread-safe snapshots + teleop control."""

    def __init__(
        self,
        *,
        n_machines: int = 3,
        seed: int = 0,
        perception: Perception | None = None,
        render: bool = True,
        speed: float = 1.0,
        min_confidence: float = 0.0,
        steps_per_publish: int = 5,
        gains: PoseGains = PoseGains(),
        geom: MecanumGeometry = MecanumGeometry(),
    ) -> None:
        self.world = World(n_machines=n_machines, seed=seed)
        self.robot = Robot(self.world, gains=gains, geom=geom)
        # Default to the trained model; tests inject a stand-in (and render=False) to run headless.
        if perception is None:
            from feathersim.perception.train import load_or_train_model

            perception = Perception(load_or_train_model())
        self.perception = perception
        self.gains = gains
        self.geom = geom
        self.speed = speed
        self.min_confidence = min_confidence
        self.steps_per_publish = steps_per_publish

        # Renderers are created on the *sim thread* in _run (GL contexts are thread-affine on macOS —
        # a context made on one thread renders nothing from another), not here on the caller's thread.
        self._render = render
        self._perc_renderer: mujoco.Renderer | None = None
        self._feed_renderer: mujoco.Renderer | None = None

        # Autonomy state machine.
        self._phase = "select"
        self._target: tuple[float, float, float] | None = None
        self._target_machine: str | None = None
        self._done_since: dict[str, float] = {}
        self._last_readings: dict | None = None

        # Cross-thread shared state (guarded by _lock): operator inputs + published outputs.
        self._lock = threading.Lock()
        self._mode = AUTO
        self._teleop = Teleop()
        self._snapshot: dict = self._build_telemetry()
        self._frame: bytes | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background sim thread (idempotent)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="feathersim-sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to halt and join it (the thread releases its own renderers)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                # Wedged in a slow render/perception call — its GL renderers can't be closed from
                # here (context is thread-affine), so warn loudly rather than report a clean shutdown.
                _log.warning("sim thread did not stop within 5s; renderers may leak")
            self._thread = None

    def is_running(self) -> bool:
        """True while the sim thread is active — lets the MJPEG stream end cleanly on shutdown."""
        return self._thread is not None and not self._stop.is_set()

    def _run(self) -> None:
        if self._render:  # build the GL renderers on this thread so their contexts live here
            self._perc_renderer = mujoco.Renderer(self.world.model, IMAGE_SIZE, IMAGE_SIZE)
            self._feed_renderer = mujoco.Renderer(self.world.model, FEED_SIZE, FEED_SIZE)
        try:
            while not self._stop.is_set():
                start = time.perf_counter()
                for _ in range(self.steps_per_publish):
                    self.tick()
                self._publish()
                # Pace to (roughly) ``speed`` × real time so the browser feed is watchable.
                budget = self.steps_per_publish * TIMESTEP / self.speed
                time.sleep(max(0.0, budget - (time.perf_counter() - start)))
        finally:
            for r in (self._perc_renderer, self._feed_renderer):
                if r is not None:
                    r.close()
            self._perc_renderer = self._feed_renderer = None

    # --- operator control (called from HTTP threads) -------------------------------------

    def set_teleop(self, vx: float, vy: float, omega: float) -> None:
        """Apply an operator twist; any teleop command seizes manual control (preempts autonomy)."""
        vx, vy = _clamp_vec(vx, vy, self.gains.max_linear)  # clamp the resultant, not each axis
        with self._lock:
            self._mode = MANUAL
            self._teleop = Teleop(vx, vy, _clamp(omega, self.gains.max_angular))

    def set_mode(self, mode: str) -> None:
        """Switch between ``"auto"`` (resume autonomy) and ``"manual"`` (hold position under operator)."""
        if mode not in (AUTO, MANUAL):
            raise ValueError(f"mode must be {AUTO!r} or {MANUAL!r}, got {mode!r}")
        with self._lock:
            self._mode = mode
            self._teleop = Teleop()  # zero the twist on any explicit mode change

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def telemetry(self) -> dict:
        """Latest published telemetry snapshot (thread-safe copy)."""
        with self._lock:
            return dict(self._snapshot)

    def frame(self) -> bytes | None:
        """Latest published camera JPEG, or ``None`` if rendering is disabled / not yet produced."""
        with self._lock:
            return self._frame

    # --- the per-tick control loop -------------------------------------------------------

    def tick(self) -> None:
        """Advance the sim one control step under the current mode (the unit of preemption)."""
        with self._lock:
            mode, teleop = self._mode, self._teleop
        if mode == MANUAL:
            self._command_body_twist(teleop.vx, teleop.vy, teleop.omega)
            self.world.step()
        else:
            self._autonomy_step()

    def _autonomy_step(self) -> None:
        phase = self._phase
        if phase == "select":
            machine = self._select_machine()
            if machine is None:
                self._command_body_twist(0.0, 0.0, 0.0)
                self.world.step()  # idle in place; let the machines keep cycling
                return
            self._target_machine = machine
            self._target = self.robot.tending_pose(machine)
            self._phase = "to_machine"
        elif phase in ("to_machine", "to_table"):
            pose = self.world.robot_pose()
            distance, heading = pose_error(pose, self._target)
            if distance <= _ARRIVE_POS and heading <= _ARRIVE_HEADING:
                self.world.stop_base()
                self._phase = "pick" if phase == "to_machine" else "place"
                return
            vx, vy, omega = velocity_command(pose, self._target, self.gains)
            self._command_body_twist(vx, vy, omega)
            self.world.step()
        elif phase == "pick":
            try:
                self.robot.pick(self._target_machine)
            except PreconditionError:
                self._phase = "select"  # perception false positive — re-perceive
                return
            self._target = self.robot.tending_pose("table")
            self._phase = "to_table"
        elif phase == "place":
            self.robot.place("table")
            self._phase = "select"

    def _select_machine(self) -> str | None:
        """Perceive every machine and return the longest-waiting perceived-``done`` one (oldest-first)."""
        readings = self.perception.perceive(self.world, self._perc_renderer)
        self._last_readings = readings
        now = self.world.time
        perceived_done = {
            n for n, r in readings.items()
            if r.machine_state is MachineState.DONE and r.confidence >= self.min_confidence
        }
        self._done_since = {n: t for n, t in self._done_since.items() if n in perceived_done}
        for n in perceived_done:
            self._done_since.setdefault(n, now)
        if not perceived_done:
            return None
        return min(perceived_done, key=lambda n: (self._done_since[n], n))

    def _command_body_twist(self, vx: float, vy: float, omega: float) -> None:
        """Route a body twist through the mecanum IK→FK round-trip, then command the base."""
        wheels = body_to_wheels(vx, vy, omega, self.geom)
        rvx, rvy, romega = wheels_to_body(wheels, self.geom)
        self.world.command_base_velocity(rvx, rvy, romega)

    # --- publishing ----------------------------------------------------------------------

    def _publish(self) -> None:
        telemetry = self._build_telemetry()
        frame = self._render_feed() if self._feed_renderer is not None else None
        with self._lock:
            self._snapshot = telemetry
            if frame is not None:
                self._frame = frame

    def _skill_text(self) -> str:
        return _SKILL_TEXT[self._phase].format(target=self._target_machine)

    def _build_telemetry(self) -> dict:
        """Snapshot the sim for the dashboard. Runs on the sim thread (safe direct world access)."""
        with self._lock:
            mode = self._mode
        x, y, yaw = self.world.robot_pose()
        sim_time = self.world.time
        machines = []
        for m in self.world.machines:
            perceived = (self._last_readings or {}).get(m.name)
            machines.append(
                {
                    "name": m.name,
                    "state": m.state.value,  # ground truth
                    "perceived": perceived.machine_state.value if perceived else None,
                    "confidence": round(perceived.confidence, 3) if perceived else None,
                    "parts_done": m.parts_done,
                }
            )
        delivered = len(self.robot.delivered)
        return {
            "sim_time": round(sim_time, 2),
            "mode": mode,
            "skill": "manual override" if mode == MANUAL else self._skill_text(),
            "holding": self.robot.holding,
            "delivered": delivered,
            # Throughput is over *total* sim time, so manual-override time counts against the rate.
            "throughput_per_min": round(delivered * 60.0 / sim_time, 2) if sim_time > 0 else 0.0,
            "robot": {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)},
            "machines": machines,
        }

    def _render_feed(self) -> bytes:
        rgb = self.world.render(self._feed_renderer, self.world.overview_camera())
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
        return buf.getvalue()


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _clamp_vec(vx: float, vy: float, limit: float) -> tuple[float, float]:
    """Scale ``(vx, vy)`` down so its magnitude never exceeds ``limit`` (preserves direction)."""
    mag = math.hypot(vx, vy)
    if mag > limit and mag > 0.0:
        scale = limit / mag
        return vx * scale, vy * scale
    return vx, vy
