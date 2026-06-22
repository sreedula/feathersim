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
from feathersim.fleet import STRATEGIES, FleetController, longest_waiting
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
FEED_SIZE = 720  # the cinematic 3D overview feed resolution (≤ the 1280 offscreen buffer)
CAM_SIZE = 256   # per-robot onboard camera resolution (composited into a strip)
HUD_CROP = 150   # the perception-HUD cell renders each machine crop at this size
HUD_HEADER = 30
HUD_PAD = 10
HUD_FOOTER = 62  # cell space below the crop for prediction + confidence bar + verdict
TRAIL_LEN = 48   # how many recent positions each robot's tactical-map trajectory trail keeps


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(int(255 * c) for c in rgb)


class FleetSimManager:
    """Owns the fleet sim and runs it on a thread; exposes thread-safe telemetry, a schematic, + controls."""

    def __init__(
        self, *, n_machines: int = 4, n_robots: int = 4, n_obstacles: int = 0, seed: int = 0,
        render: bool = True, speed: float = 1.0, steps_per_publish: int = 5, difficulty: float = 0.4,
    ) -> None:
        self.world = World(n_machines=n_machines, seed=seed, n_obstacles=n_obstacles, n_robots=n_robots)
        self.perception = Perception(load_or_train_model())             # robust model — drives the fleet
        self.clean_perception = Perception(load_or_train_clean_model())  # clean baseline — for the slider
        self.policy = PolicyController(load_or_train_policy())
        self.difficulty = difficulty            # live perception-difficulty slider (0=clean, 1=full DR)
        self.controller_name = "rule"           # "rule" (hand-coded) or "learned" (BC policy)
        self.strategy_name = "longest_waiting"  # live scheduling strategy (see STRATEGIES)
        self.speed, self.steps_per_publish = speed, steps_per_publish
        self._rngs = [np.random.default_rng(seed * 100 + k) for k in range(n_robots)]
        # Recent perceived==true booleans → live accuracy, for the deployed robust model and the clean one.
        self._acc = deque(maxlen=n_machines * 40)
        self._acc_clean = deque(maxlen=n_machines * 40)
        self._render = render
        self._perc_renderer: mujoco.Renderer | None = None  # built on the sim thread (GL affinity)
        self._feed_renderer: mujoco.Renderer | None = None   # 3D overview feed (also sim-thread)
        self._cam_renderer: mujoco.Renderer | None = None    # per-robot onboard cameras (sim-thread)
        self._overview_cam = self.world.overview_camera(distance=7.0, elevation=-23.0, azimuth=92.0)

        self.ctrl = FleetController(
            self.world, self._perceive, strategy=longest_waiting, strategy_name="longest_waiting",
            velocity_fn=velocity_command,
        )

        self._lock = threading.Lock()
        self._snapshot: dict = self._build_telemetry()
        self._frame: bytes | None = None        # schematic JPEG
        self._frame3d: bytes | None = None       # cinematic 3D overview JPEG
        self._frame_cams: bytes | None = None    # per-robot onboard-camera strip JPEG
        self._frame_hud: bytes | None = None     # perception HUD: what the model saw + inferred
        # Latest per-machine perception, captured in _perceive for the HUD (the DR-corrupted crop the
        # robust model actually received + its prediction/confidence + agreement with ground truth).
        self._hud_data: list[dict | None] = [None] * self.world.n_machines
        # Recent (x, y) per robot for the tactical map's fading trajectory trail — the actual ORCA-curved
        # path travelled (distinct from the straight A* planned line).
        self._trails: list[deque] = [deque(maxlen=TRAIL_LEN) for _ in range(self.world.n_robots)]
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
            # Stash what the model saw + inferred for the perception HUD (latest read wins; same sim thread).
            self._hud_data[i] = {
                "crop": np.clip(image, 0, 255).astype(np.uint8),
                "pred": reading.machine_state, "conf": float(reading.confidence),
                "true": machine.state, "correct": reading.machine_state is machine.state,
                "robot": robot_id,
            }
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

    def set_strategy(self, name: str) -> None:
        """Swap the live scheduling strategy (longest_waiting / nearest_done / balanced). The manager reads
        ``strategy`` on the sim thread; this single attribute write is atomic under the GIL (lock-free)."""
        if name not in STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(STRATEGIES)}, got {name!r}")
        self.strategy_name = name
        self.ctrl.manager.strategy = STRATEGIES[name]

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

    def frame_hud(self) -> bytes | None:
        with self._lock:
            return self._frame_hud

    # --- publishing ----------------------------------------------------------------------

    def _publish(self) -> None:
        for k in range(self.world.n_robots):     # extend each robot's trajectory trail
            self._trails[k].append(self.world.robot_pose(k)[:2])
        telemetry = self._build_telemetry()
        frame = self._render_schematic()
        frame_hud = self._render_perception_hud()   # PIL-only (uses captured crops); no GL needed
        frame3d = self._render_3d() if self._feed_renderer is not None else None  # also resets the scene clean
        frame_cams = self._render_robotcams() if self._cam_renderer is not None else None
        with self._lock:
            self._snapshot, self._frame = telemetry, frame
            if frame_hud is not None:
                self._frame_hud = frame_hud
            if frame3d is not None:
                self._frame3d = frame3d
            if frame_cams is not None:
                self._frame_cams = frame_cams

    def _render_perception_hud(self) -> bytes | None:
        """The headline 'see what the robot sees AND thinks' panel: for each machine, the DR-corrupted
        crop the robust model actually received, the state it inferred + confidence, and whether that
        matches ground truth. As the difficulty slider rises, the crops visibly degrade and wrong/low-
        confidence reads appear — the model's belief shown next to the exact pixels that produced it."""
        if not any(d is not None for d in self._hud_data):
            return None
        n = self.world.n_machines
        cell_w = HUD_CROP + 2 * HUD_PAD
        cell_h = HUD_HEADER + HUD_CROP + HUD_FOOTER
        img = Image.new("RGB", (cell_w * n, cell_h), (13, 15, 19))
        draw = ImageDraw.Draw(img)
        for i in range(n):
            ox = i * cell_w
            draw.rectangle([ox, 0, ox + cell_w - 1, HUD_HEADER - 1], fill=(22, 26, 32))
            draw.text((ox + HUD_PAD, 9), f"MACHINE {i}", fill=(212, 218, 226))
            d = self._hud_data[i]
            if d is None:
                draw.text((ox + HUD_PAD, HUD_HEADER + HUD_CROP // 2), "awaiting read", fill=(120, 126, 134))
                continue
            crop = Image.fromarray(d["crop"]).resize((HUD_CROP, HUD_CROP), Image.BILINEAR)
            img.paste(crop, (ox + HUD_PAD, HUD_HEADER))
            draw.rectangle([ox + HUD_PAD, HUD_HEADER, ox + HUD_PAD + HUD_CROP - 1, HUD_HEADER + HUD_CROP - 1], outline=(44, 50, 58))
            draw.text((ox + cell_w - HUD_PAD - 42, 9), f"via r{d['robot']}", fill=_hex(_ROBOT_COLORS[d["robot"]]))
            pcol = _hex(STATE_LIGHT[d["pred"]])
            py = HUD_HEADER + HUD_CROP + 6
            draw.ellipse([ox + HUD_PAD, py + 1, ox + HUD_PAD + 11, py + 12], fill=pcol)
            draw.text((ox + HUD_PAD + 17, py + 1), f"sees: {d['pred'].value}", fill=pcol)
            cy = py + 20                                   # confidence bar
            draw.rectangle([ox + HUD_PAD, cy, ox + HUD_PAD + HUD_CROP, cy + 8], fill=(33, 38, 45))
            draw.rectangle([ox + HUD_PAD, cy, ox + HUD_PAD + int(HUD_CROP * d["conf"]), cy + 8], fill=(68, 147, 248))
            draw.text((ox + HUD_PAD + HUD_CROP - 34, cy - 2), f"{int(d['conf'] * 100)}%", fill=(184, 190, 198))
            vy = cy + 13                                   # verdict vs ground truth
            if d["correct"]:
                draw.text((ox + HUD_PAD, vy), "OK  matches truth", fill=(46, 204, 85))
            else:
                draw.text((ox + HUD_PAD, vy), f"X  wrong (is {d['true'].value})", fill=(248, 81, 73))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=86)
        return buf.getvalue()

    _PHASE_LABEL = {
        "select": "scanning", "to_machine": "en route", "pick": "grasping",
        "pick_lift": "lifting", "to_table": "delivering", "place": "placing", "place_lift": "lifting",
    }

    def _render_robotcams(self) -> bytes:
        """Composite each robot's forward onboard camera into a strip with a small robot-HUD overlay —
        a centre crosshair + a live status bar (current phase + target). The scene is already clean here
        (`_render_3d` ran reset_scene + sync_visuals just before)."""
        strip = Image.new("RGB", (CAM_SIZE * self.world.n_robots, CAM_SIZE), (12, 14, 18))
        draw = ImageDraw.Draw(strip)
        for k in range(self.world.n_robots):
            self._cam_renderer.update_scene(self.world.data, f"robotcam_{k}")
            strip.paste(Image.fromarray(self._cam_renderer.render()), (k * CAM_SIZE, 0))
            ox = k * CAM_SIZE
            color = _hex(_ROBOT_COLORS[k])
            cx, cy = ox + CAM_SIZE // 2, CAM_SIZE // 2     # reticle
            draw.line([cx - 9, cy, cx + 9, cy], fill=color, width=1)
            draw.line([cx, cy - 9, cx, cy + 9], fill=color, width=1)
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=color, width=1)
            draw.rectangle([ox, 0, ox + CAM_SIZE - 1, CAM_SIZE - 1], outline=color, width=3)
            draw.text((ox + 8, 6), f"robot {k}", fill=color)
            phase = self._PHASE_LABEL.get(self.ctrl.phase[k], self.ctrl.phase[k])  # live status bar
            tgt = self.ctrl.target[k]
            status = f"{phase}" + (f"  {tgt}" if tgt and self.ctrl.phase[k] != "select" else "")
            draw.rectangle([ox, CAM_SIZE - 22, ox + CAM_SIZE - 1, CAM_SIZE - 1], fill=(10, 12, 16))
            draw.text((ox + 8, CAM_SIZE - 18), status, fill=(210, 216, 224))
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
        Image.fromarray(rgb).save(buf, format="JPEG", quality=88)
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
            vx, vy = ctrl.vel[k]                          # ORCA's commanded world velocity (0 when parked)
            robots.append({
                "id": k, "color": _hex(_ROBOT_COLORS[k]), "phase": ctrl.phase[k],
                "target": ctrl.target[k], "pose": {"x": round(x, 2), "y": round(y, 2), "yaw": round(yaw, 2)},
                "speed": round(math.hypot(vx, vy), 2),
            })
        accuracy = round(float(np.mean(self._acc)), 3) if self._acc else None
        clean_accuracy = round(float(np.mean(self._acc_clean)), 3) if self._acc_clean else None
        # Recent deliveries, newest first — the live mission log (robot k delivered <part> from <machine>).
        recent_events = [
            {"robot": k, "machine": mname, "part": part, "t": round(t, 1)}
            for (k, mname, part, t) in ctrl.events[-9:][::-1]
        ]
        return {
            "sim_time": round(sim_time, 1),
            "delivered": ctrl.delivered,
            "throughput_per_min": round(ctrl.delivered * 60.0 / sim_time, 1) if sim_time > 0 else 0.0,
            "controller": self.controller_name,
            "strategy": self.strategy_name,
            "strategies": sorted(STRATEGIES),
            "difficulty": round(self.difficulty, 2),
            "perception_accuracy": accuracy,
            "clean_accuracy": clean_accuracy,
            "min_robot_separation": round(ctrl.min_sep, 2) if ctrl.min_sep != math.inf else None,
            "robots": robots,
            "machines": machines,
            "recent_events": recent_events,
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

        for k in range(world.n_robots):                                                          # trajectory trails
            trail = self._trails[k] if k < len(self._trails) else ()
            pts = list(trail)
            base = _ROBOT_COLORS[k]
            for i in range(1, len(pts)):
                frac = i / len(pts)                          # fade in toward the newest segment
                col = tuple(int(255 * c * (0.18 + 0.5 * frac)) for c in base)
                d.line([self._px(*pts[i - 1]), self._px(*pts[i])], fill=col, width=2)

        for k in range(world.n_robots):                                                          # planned paths
            path = ctrl.path[k]
            if path and len(path) > 1:
                d.line([self._px(x, y) for x, y in path], fill=_hex(_ROBOT_COLORS[k]), width=1)

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
