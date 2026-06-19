"""Command-center engine: the fleet running live, with overlays + interactive controls. [v2 Phase E]

Wraps the Phase-C :class:`FleetController` on a background thread (one sim, all stepping/rendering here),
and publishes — under a lock — a rich telemetry snapshot (per-robot phase/target, per-machine true *and*
perceived state, task assignments, live perception accuracy) plus a top-down **schematic** JPEG with each
robot's planned path overlaid. Two live controls: a **controller toggle** (hand-coded ↔ learned policy)
and a **perception-difficulty slider** that scales domain randomization and visibly moves accuracy.
"""

from __future__ import annotations

import io
import math
import threading
import time
from collections import deque

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from feathersim.control.go_to_pose import velocity_command
from feathersim.fleet import FleetController, longest_waiting
from feathersim.perception.dataset import IMAGE_SIZE
from feathersim.perception.infer import Perception
from feathersim.perception.randomize import DomainRandomizer, apply_scene
from feathersim.perception.train import load_or_train_clean_model, load_or_train_model
from feathersim.policy.policy import PolicyController
from feathersim.policy.train import load_or_train_policy
from feathersim.sim.world import (
    GRID_BOUNDS,
    ROBOT_RADIUS,
    STATE_LIGHT,
    TIMESTEP,
    _OBSTACLE_HALF,
    _OBSTACLE_POSITIONS,
    _ROBOT_COLORS,
    TABLE_XY,
    World,
)

SCHEMATIC_SIZE = 460
FEED_SIZE = 560  # the cinematic 3D overview feed resolution
CAM_SIZE = 220   # per-robot onboard camera resolution (composited into a strip)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(int(255 * c) for c in rgb)


class FleetSimManager:
    """Owns the fleet sim and runs it on a thread; exposes thread-safe telemetry, a schematic, + controls."""

    def __init__(
        self, *, n_machines: int = 3, n_robots: int = 3, n_obstacles: int = 0, seed: int = 0,
        render: bool = True, speed: float = 1.0, steps_per_publish: int = 5, difficulty: float = 0.4,
    ) -> None:
        self.world = World(n_machines=n_machines, seed=seed, n_obstacles=n_obstacles, n_robots=n_robots)
        self.perception = Perception(load_or_train_model())             # robust model — drives the fleet
        self.clean_perception = Perception(load_or_train_clean_model())  # clean baseline — for the slider
        self.policy = PolicyController(load_or_train_policy())
        self.difficulty = difficulty            # live perception-difficulty slider (0=clean, 1=full DR)
        self.controller_name = "rule"           # "rule" (hand-coded) or "learned" (BC policy)
        self.speed, self.steps_per_publish = speed, steps_per_publish
        self._rngs = [np.random.default_rng(seed * 100 + k) for k in range(n_robots)]
        # Recent perceived==true booleans → live accuracy, for the deployed robust model and the clean one.
        self._acc = deque(maxlen=n_machines * 40)
        self._acc_clean = deque(maxlen=n_machines * 40)
        self._render = render
        self._perc_renderer: mujoco.Renderer | None = None  # built on the sim thread (GL affinity)
        self._feed_renderer: mujoco.Renderer | None = None   # 3D overview feed (also sim-thread)
        self._cam_renderer: mujoco.Renderer | None = None    # per-robot onboard cameras (sim-thread)
        self._overview_cam = self.world.overview_camera(distance=6.2, elevation=-32.0)

        self.ctrl = FleetController(
            self.world, self._perceive, strategy=longest_waiting, strategy_name="longest_waiting",
            velocity_fn=velocity_command,
        )

        self._lock = threading.Lock()
        self._snapshot: dict = self._build_telemetry()
        self._frame: bytes | None = None        # schematic JPEG
        self._frame3d: bytes | None = None       # cinematic 3D overview JPEG
        self._frame_cams: bytes | None = None    # per-robot onboard-camera strip JPEG
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- perception (called by the controller on the sim thread) -------------------------

    def _perceive(self, robot_id: int) -> dict:
        self.world.sync_visuals()
        randomizer = DomainRandomizer.at_difficulty(self.difficulty)
        # Apply the full scene-stage DR (randomized lighting + status-light occluders) before rendering,
        # then sensor-corrupt the crop — otherwise only noise/blur reaches the model and the slider barely bites.
        apply_scene(self.world, randomizer.sample_scene(self._rngs[robot_id], self.world.n_machines))
        out = {}
        for i, machine in enumerate(self.world.machines):
            image = randomizer.corrupt_image(
                self.world.render_machine(self._perc_renderer, i), self._rngs[robot_id]
            )
            reading = self.perception.read(image)          # robust model — the fleet acts on this
            out[machine.name] = reading
            self._acc.append(reading.machine_state is machine.state)
            self._acc_clean.append(self.clean_perception.read(image).machine_state is machine.state)
        return out

    # --- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="feathersim-fleet", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and not self._stop.is_set()

    def _run(self) -> None:
        if self._render:
            self._perc_renderer = mujoco.Renderer(self.world.model, IMAGE_SIZE, IMAGE_SIZE)
            self._feed_renderer = mujoco.Renderer(self.world.model, FEED_SIZE, FEED_SIZE)
            self._cam_renderer = mujoco.Renderer(self.world.model, CAM_SIZE, CAM_SIZE)
        try:
            while not self._stop.is_set():
                start = time.perf_counter()
                for _ in range(self.steps_per_publish):
                    self.ctrl.step()
                self._publish()
                budget = self.steps_per_publish * TIMESTEP / self.speed
                time.sleep(max(0.0, budget - (time.perf_counter() - start)))
        finally:
            for r in (self._perc_renderer, self._feed_renderer, self._cam_renderer):
                if r is not None:
                    r.close()
            self._perc_renderer = self._feed_renderer = self._cam_renderer = None

    # --- controls (HTTP threads) ---------------------------------------------------------

    def set_controller(self, name: str) -> None:
        """Toggle the drive controller: ``"rule"`` (hand-coded) or ``"learned"`` (BC policy).

        The single attribute write the sim thread reads is atomic under the GIL — intentionally lock-free.
        """
        if name not in ("rule", "learned"):
            raise ValueError(f"controller must be 'rule' or 'learned', got {name!r}")
        self.controller_name = name
        self.ctrl.velocity_fn = velocity_command if name == "rule" else self.policy

    def set_difficulty(self, value: float) -> None:
        """Set perception difficulty in [0, 1] (scales the live DR). Single atomic write, lock-free (GIL)."""
        self.difficulty = max(0.0, min(1.0, float(value)))

    def telemetry(self) -> dict:
        with self._lock:
            return dict(self._snapshot)

    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def frame3d(self) -> bytes | None:
        with self._lock:
            return self._frame3d

    def frame_cams(self) -> bytes | None:
        with self._lock:
            return self._frame_cams

    # --- publishing ----------------------------------------------------------------------

    def _publish(self) -> None:
        telemetry = self._build_telemetry()
        frame = self._render_schematic()
        frame3d = self._render_3d() if self._feed_renderer is not None else None  # also resets the scene clean
        frame_cams = self._render_robotcams() if self._cam_renderer is not None else None
        with self._lock:
            self._snapshot, self._frame = telemetry, frame
            if frame3d is not None:
                self._frame3d = frame3d
            if frame_cams is not None:
                self._frame_cams = frame_cams

    def _render_robotcams(self) -> bytes:
        """Composite each robot's forward onboard camera into a labelled horizontal strip (the scene is
        already clean here — `_render_3d` ran reset_scene + sync_visuals just before)."""
        strip = Image.new("RGB", (CAM_SIZE * self.world.n_robots, CAM_SIZE), (12, 14, 18))
        draw = ImageDraw.Draw(strip)
        for k in range(self.world.n_robots):
            self._cam_renderer.update_scene(self.world.data, f"robotcam_{k}")
            strip.paste(Image.fromarray(self._cam_renderer.render()), (k * CAM_SIZE, 0))
            draw.rectangle([k * CAM_SIZE, 0, (k + 1) * CAM_SIZE - 1, CAM_SIZE - 1], outline=_hex(_ROBOT_COLORS[k]), width=3)
            draw.text((k * CAM_SIZE + 8, 6), f"robot {k}", fill=_hex(_ROBOT_COLORS[k]))
        buf = io.BytesIO()
        strip.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def _render_3d(self) -> bytes:
        # Restore the clean cinematic scene (a perception read may have left it randomized) and paint the
        # status lights to live machine state, then render the overview.
        self.world.reset_scene()
        self.world.sync_visuals()
        rgb = self.world.render(self._feed_renderer, self._overview_cam)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=82)
        return buf.getvalue()

    def _build_telemetry(self) -> dict:
        world, ctrl = self.world, self.ctrl
        sim_time = world.time
        machines = []
        for i, m in enumerate(world.machines):
            perceived = {}
            for k in range(world.n_robots):
                reading = ctrl.last_readings[k]
                if reading and m.name in reading:
                    perceived[k] = reading[m.name].machine_state.value
            machines.append({
                "name": m.name, "state": m.state.value, "parts_done": m.parts_done,
                "assigned_to": ctrl.manager.assignments.get(m.name), "perceived_by": perceived,
            })
        robots = []
        for k in range(world.n_robots):
            x, y, yaw = world.robot_pose(k)
            robots.append({
                "id": k, "color": _hex(_ROBOT_COLORS[k]), "phase": ctrl.phase[k],
                "target": ctrl.target[k], "pose": {"x": round(x, 2), "y": round(y, 2), "yaw": round(yaw, 2)},
            })
        accuracy = round(float(np.mean(self._acc)), 3) if self._acc else None
        clean_accuracy = round(float(np.mean(self._acc_clean)), 3) if self._acc_clean else None
        return {
            "sim_time": round(sim_time, 1),
            "delivered": ctrl.delivered,
            "throughput_per_min": round(ctrl.delivered * 60.0 / sim_time, 1) if sim_time > 0 else 0.0,
            "controller": self.controller_name,
            "difficulty": round(self.difficulty, 2),
            "perception_accuracy": accuracy,
            "clean_accuracy": clean_accuracy,
            "min_robot_separation": round(ctrl.min_sep, 2) if ctrl.min_sep != math.inf else None,
            "robots": robots,
            "machines": machines,
        }

    # --- schematic (top-down, no GL) -----------------------------------------------------

    def _px(self, x: float, y: float) -> tuple[int, int]:
        xmin, ymin, xmax, ymax = GRID_BOUNDS
        s = SCHEMATIC_SIZE
        return (int((x - xmin) / (xmax - xmin) * s), int((1 - (y - ymin) / (ymax - ymin)) * s))

    def _rect(self, cx: float, cy: float, hx: float, hy: float) -> list[int]:
        x0, y0 = self._px(cx - hx, cy + hy)
        x1, y1 = self._px(cx + hx, cy - hy)
        return [x0, y0, x1, y1]

    def _render_schematic(self) -> bytes:
        buf = io.BytesIO()
        self._schematic_image().save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _schematic_image(self) -> Image.Image:
        """The top-down cell as a PIL image: table, pillars, machines (true-state colored), planned paths,
        and robots. Pure drawing — no GL — so it also backs the README fleet GIF."""
        world, ctrl = self.world, self.ctrl
        img = Image.new("RGB", (SCHEMATIC_SIZE, SCHEMATIC_SIZE), (16, 19, 24))
        d = ImageDraw.Draw(img)

        d.rectangle(self._rect(*TABLE_XY, 0.5, 0.3), fill=(96, 67, 38), outline=(140, 100, 60))  # table
        for ox, oy in _OBSTACLE_POSITIONS[: world.n_obstacles]:                                  # pillars
            d.rectangle(self._rect(ox, oy, _OBSTACLE_HALF, _OBSTACLE_HALF), fill=(190, 90, 30))
        for m in world.machines:                                                                 # machines
            mx, my = world.fixtures[m.name]
            color = tuple(int(255 * c) for c in STATE_LIGHT[m.state])  # TRUE state color
            d.rectangle(self._rect(mx, my, 0.3, 0.3), fill=color, outline=(230, 230, 230))
            d.text((self._px(mx, my)[0] - 22, self._px(mx, my)[1] - 6), f"{m.name[-1]}:{m.parts_done}",
                   fill=(20, 20, 20))

        for k in range(world.n_robots):                                                          # planned paths
            path = ctrl.path[k]
            if path and len(path) > 1:
                d.line([self._px(x, y) for x, y in path], fill=_hex(_ROBOT_COLORS[k]), width=2)

        for k in range(world.n_robots):                                                          # robots
            x, y, yaw = world.robot_pose(k)
            color = tuple(int(255 * c) for c in _ROBOT_COLORS[k])
            r = int(ROBOT_RADIUS / (GRID_BOUNDS[2] - GRID_BOUNDS[0]) * SCHEMATIC_SIZE)
            cx, cy = self._px(x, y)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(255, 255, 255))
            hx, hy = self._px(x + 0.25 * math.cos(yaw), y + 0.25 * math.sin(yaw))
            d.line([cx, cy, hx, hy], fill=(255, 255, 255), width=2)
            d.text((cx - 3, cy - 6), str(k), fill=(0, 0, 0))

        return img
