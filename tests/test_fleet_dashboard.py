"""v2 Phase E: the command-center dashboard.

Construction, the controls, telemetry shape, and the top-down schematic are all headless (no GL — the
schematic is drawn with PIL, not rendered). The live perception accuracy + the routes need the manager's
render thread and are GL-guarded.
"""

import mujoco
import pytest
from fastapi.testclient import TestClient

from feathersim.control.go_to_pose import velocity_command
from feathersim.dashboard.fleet_manager import CAM_SIZE, HUD_CROP, HUD_FOOTER, HUD_HEADER, HUD_PAD, FleetSimManager
from feathersim.dashboard.fleet_server import create_app
from feathersim.perception.randomize import DomainRandomizer
from feathersim.sim.world import World


# --- difficulty scaling (pure) -------------------------------------------------------------------


def test_difficulty_zero_is_clean_and_one_is_full():
    clean = DomainRandomizer.at_difficulty(0.0)
    assert clean.noise_sigma == (0.0, 0.0) and clean.occluder_prob == 0.0 and clean.blur_prob == 0.0
    full = DomainRandomizer.at_difficulty(1.0)
    assert full.occluder_prob > 0 and full.noise_sigma[1] > 0 and full.blur_prob > 0
    assert DomainRandomizer.at_difficulty(5.0).occluder_prob == DomainRandomizer.at_difficulty(1.0).occluder_prob


# --- manager: controls, telemetry, schematic (headless) ------------------------------------------


@pytest.fixture(scope="module")
def manager():
    # Constructing loads the (committed-metrics) models from disk — no GL needed until the render thread.
    return FleetSimManager(render=False, n_robots=3)


def test_controller_toggle_swaps_velocity_fn(manager):
    manager.set_controller("learned")
    assert manager.controller_name == "learned" and manager.ctrl.velocity_fn is manager.policy
    manager.set_controller("rule")
    assert manager.controller_name == "rule" and manager.ctrl.velocity_fn is velocity_command


def test_controller_toggle_rejects_unknown(manager):
    with pytest.raises(ValueError):
        manager.set_controller("teleop")


def test_difficulty_clamps_to_unit_interval(manager):
    manager.set_difficulty(1.7)
    assert manager.difficulty == 1.0
    manager.set_difficulty(-0.3)
    assert manager.difficulty == 0.0
    manager.set_difficulty(0.5)


def test_telemetry_has_command_center_fields(manager):
    t = manager._build_telemetry()
    assert set(t) >= {
        "sim_time", "delivered", "throughput_per_min", "controller", "difficulty",
        "perception_accuracy", "clean_accuracy", "robots", "machines", "recent_events",
    }
    assert isinstance(t["recent_events"], list)
    assert len(t["robots"]) == 3 and len(t["machines"]) == manager.world.n_machines
    assert set(t["machines"][0]) >= {"name", "state", "parts_done", "assigned_to", "perceived_by"}
    assert set(t["robots"][0]) >= {"id", "color", "phase", "target", "pose", "speed"}
    assert all(r["speed"] >= 0.0 for r in t["robots"])


def test_schematic_renders_a_jpeg_without_gl(manager):
    frame = manager._render_schematic()  # PIL top-down, no rendering
    assert frame[:2] == b"\xff\xd8" and len(frame) > 500


# --- live run + routes (need GL) -----------------------------------------------------------------


def _rendering_available() -> bool:
    try:
        w = World(n_machines=1, seed=0)
        r = mujoco.Renderer(w.model, 32, 32)
        r.update_scene(w.data, w.machine_camera(0))
        r.render()
        r.close()
        return True
    except Exception:
        return False


rendering = pytest.mark.skipif(
    not _rendering_available(),
    reason="MuJoCo rendering unavailable (set MUJOCO_GL=egl/osmesa on a headless host)",
)


@rendering
def test_difficulty_slider_robust_beats_clean():
    """The headline interactive element: under high difficulty the deployed robust model holds while the
    clean baseline degrades. Driven deterministically over many reads (not the live thread, to avoid
    timing flake)."""
    import numpy as np

    from feathersim.sim.machine import MachineState

    mgr = FleetSimManager(render=False, n_robots=3, difficulty=1.0)
    mgr._perc_renderer = mujoco.Renderer(mgr.world.model, 64, 64)
    states = [MachineState.IDLE, MachineState.RUNNING, MachineState.DONE]
    for i, m in enumerate(mgr.world.machines):
        m.state = states[i % 3]
    try:
        for _ in range(60):
            for k in range(mgr.world.n_robots):
                mgr._perceive(k)
    finally:
        mgr._perc_renderer.close()
    robust, clean = float(np.mean(mgr._acc)), float(np.mean(mgr._acc_clean))
    assert robust >= clean              # the DR-trained model holds at least as well...
    assert clean < 0.95                 # ...while the clean baseline measurably degrades at difficulty 1.0


@rendering
def test_manager_runs_live_and_publishes():
    import time

    mgr = FleetSimManager(render=True, n_robots=3)
    mgr.start()
    try:
        deadline = time.time() + 8.0
        while mgr.frame() is None and time.time() < deadline:
            time.sleep(0.1)
        assert mgr.frame() is not None and mgr.frame()[:2] == b"\xff\xd8"          # schematic JPEG
        assert mgr.frame3d() is not None and mgr.frame3d()[:2] == b"\xff\xd8"      # 3D overview JPEG
        cams = mgr.frame_cams()
        assert cams is not None and cams[:2] == b"\xff\xd8"                          # onboard-camera strip JPEG
        from PIL import Image
        import io as _io
        assert Image.open(_io.BytesIO(cams)).size == (3 * CAM_SIZE, CAM_SIZE)        # 3 robot views side by side
        hud = mgr.frame_hud()                                                         # perception "sees & thinks" HUD
        assert hud is not None and hud[:2] == b"\xff\xd8"
        assert Image.open(_io.BytesIO(hud)).size == (                                 # one cell per machine
            mgr.world.n_machines * (HUD_CROP + 2 * HUD_PAD), HUD_HEADER + HUD_CROP + HUD_FOOTER)
        t = mgr.telemetry()
        assert 0.0 <= (t["perception_accuracy"] or 0.0) <= 1.0
        assert len(t["robots"]) == 3 and t["delivered"] >= 0
    finally:
        mgr.stop()


@rendering
def test_command_center_routes():
    mgr = FleetSimManager(render=True, n_robots=3)
    with TestClient(create_app(mgr)) as client:
        assert client.get("/").status_code == 200
        body = client.get("/api/telemetry").json()
        assert "machines" in body and "controller" in body
        assert client.post("/api/controller", json={"name": "learned"}).json()["controller"] == "learned"
        assert client.post("/api/controller", json={"name": "nope"}).status_code == 422
        assert client.post("/api/difficulty", json={"value": 0.8}).json()["difficulty"] == 0.8
