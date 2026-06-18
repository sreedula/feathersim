"""Phase 6: the teleop + telemetry dashboard.

The control logic (tick-based autonomy SM, teleop preemption, telemetry) is tested headless by driving
``SimManager.tick()`` directly with a scripted perception and ``render=False`` — no GL, no model, fully
deterministic. The HTTP routes are tested with FastAPI's ``TestClient``. The live MJPEG camera stream
needs a GL context and is skipped on a headless host (set ``MUJOCO_GL=egl``/``osmesa``).
"""

import mujoco
import pytest
from fastapi.testclient import TestClient

from feathersim.dashboard.server import create_app
from feathersim.dashboard.sim_manager import AUTO, MANUAL, SimManager
from feathersim.perception.infer import PerceivedState
from feathersim.sim.machine import MachineState
from feathersim.sim.world import World


class ScriptedPerception:
    """Returns fixed per-machine readings, ignoring the renderer — lets the SM run without GL."""

    def __init__(self, script: dict[str, PerceivedState]) -> None:
        self.script = script

    def perceive(self, world, renderer):
        return dict(self.script)


def _all_idle(n: int = 3) -> ScriptedPerception:
    return ScriptedPerception(
        {f"machine_{i}": PerceivedState(MachineState.IDLE, False, 0.99) for i in range(n)}
    )


def _manager(script: ScriptedPerception | None = None, **kw) -> SimManager:
    return SimManager(perception=script or _all_idle(), render=False, **kw)


def _machine(world: World, name: str):
    return next(m for m in world.machines if m.name == name)


# --- tick-based autonomy (headless) --------------------------------------------------------------


def test_autonomy_tends_machine_perceived_done():
    """Driving ticks directly, the SM should navigate to, unload, and deliver a perceived-done machine."""
    script = ScriptedPerception(
        {
            "machine_0": PerceivedState(MachineState.IDLE, False, 0.99),
            "machine_1": PerceivedState(MachineState.DONE, True, 0.99),
            "machine_2": PerceivedState(MachineState.RUNNING, True, 0.99),
        }
    )
    mgr = _manager(script)
    _machine(mgr.world, "machine_1").state = MachineState.DONE  # genuinely done so pick succeeds

    for _ in range(3000):
        mgr.tick()
        if mgr.robot.delivered:
            break

    assert mgr.robot.delivered == ["part_machine_1_0"]
    assert _machine(mgr.world, "machine_1").parts_done == 1


def test_teleop_preempts_autonomy():
    """Even with a machine done and perceived done, manual mode tends nothing and moves under teleop."""
    script = ScriptedPerception(
        {f"machine_{i}": PerceivedState(MachineState.DONE, True, 0.99) for i in range(3)}
    )
    mgr = _manager(script)
    for i in range(3):
        _machine(mgr.world, f"machine_{i}").state = MachineState.DONE

    x0 = mgr.world.robot_pose()[0]
    mgr.set_teleop(0.8, 0.0, 0.0)  # forward — seizes manual control
    for _ in range(50):
        mgr.tick()

    assert mgr.mode == MANUAL
    assert mgr.world.robot_pose()[0] > x0 + 0.1     # moved forward under teleop
    assert mgr.robot.delivered == []                 # autonomy preempted — nothing tended
    assert _machine(mgr.world, "machine_0").state is MachineState.DONE  # never unloaded


def test_set_mode_resumes_autonomy_and_zeroes_teleop():
    mgr = _manager()
    mgr.set_teleop(0.5, 0.0, 0.0)
    assert mgr.mode == MANUAL

    mgr.set_mode(AUTO)

    assert mgr.mode == AUTO
    assert (mgr._teleop.vx, mgr._teleop.vy, mgr._teleop.omega) == (0.0, 0.0, 0.0)


def test_set_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _manager().set_mode("sideways")


def test_teleop_twist_is_clamped_to_limits():
    mgr = _manager()
    mgr.set_teleop(99.0, 0.0, 99.0)  # absurd request
    assert mgr._teleop.vx == mgr.gains.max_linear
    assert mgr._teleop.omega == mgr.gains.max_angular


def test_teleop_clamps_resultant_speed_not_each_axis():
    """A diagonal command is clamped by vector magnitude, so the resultant speed obeys max_linear."""
    mgr = _manager()
    mgr.set_teleop(10.0, 10.0, 0.0)  # equal diagonal, far over the limit
    speed = (mgr._teleop.vx ** 2 + mgr._teleop.vy ** 2) ** 0.5
    assert speed == pytest.approx(mgr.gains.max_linear)
    assert mgr._teleop.vx == pytest.approx(mgr._teleop.vy)  # direction preserved


def test_telemetry_snapshot_shape():
    script = ScriptedPerception(
        {f"machine_{i}": PerceivedState(MachineState.DONE, True, 0.97) for i in range(3)}
    )
    mgr = _manager(script)

    snap = mgr._build_telemetry()
    assert set(snap) >= {
        "sim_time", "mode", "skill", "holding", "delivered", "throughput_per_min", "robot", "machines",
    }
    assert snap["mode"] == AUTO and snap["holding"] is None
    assert set(snap["robot"]) == {"x", "y", "yaw"}
    assert len(snap["machines"]) == 3
    m0 = snap["machines"][0]
    assert set(m0) == {"name", "state", "perceived", "confidence", "parts_done"}
    assert m0["perceived"] is None  # nothing perceived until the SM first runs select

    mgr.tick()  # runs select → perceive populates the perceived fields
    assert mgr._build_telemetry()["machines"][0]["perceived"] == "done"


# --- HTTP routes ---------------------------------------------------------------------------------


def test_index_served():
    with TestClient(create_app(_manager())) as client:
        r = client.get("/")
        assert r.status_code == 200 and "FeatherSim" in r.text


def test_telemetry_route():
    with TestClient(create_app(_manager())) as client:
        r = client.get("/api/telemetry")
        assert r.status_code == 200
        assert "machines" in r.json() and "mode" in r.json()


def test_teleop_route_seizes_manual():
    with TestClient(create_app(_manager())) as client:
        r = client.post("/api/teleop", json={"vx": 0.5, "vy": 0.0, "omega": 0.0})
        assert r.status_code == 200 and r.json()["mode"] == MANUAL


def test_mode_route_round_trip():
    with TestClient(create_app(_manager())) as client:
        assert client.post("/api/teleop", json={"vx": 0.3}).json()["mode"] == MANUAL
        assert client.post("/api/mode", json={"mode": "auto"}).json()["mode"] == AUTO


def test_mode_route_rejects_bad_mode():
    with TestClient(create_app(_manager())) as client:
        assert client.post("/api/mode", json={"mode": "diagonal"}).status_code == 422


def test_camera_returns_503_without_rendering():
    with TestClient(create_app(_manager())) as client:
        assert client.get("/api/camera").status_code == 503


# --- live camera stream (needs a GL context) -----------------------------------------------------


def _rendering_available() -> bool:
    try:
        w = World(n_machines=1, seed=0)
        r = mujoco.Renderer(w.model, 32, 32)
        r.update_scene(w.data, w.overview_camera())
        r.render()
        r.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _rendering_available(),
    reason="MuJoCo rendering unavailable (set MUJOCO_GL=egl/osmesa on a headless host)",
)
def test_camera_renders_jpeg_frames():
    """With rendering on, the sim thread publishes JPEG frames and the MJPEG generator emits them.

    The generator is exercised directly rather than through ``TestClient.stream`` — an endless
    multipart response doesn't tear down cleanly under the test transport (uvicorn stops pulling it on
    browser disconnect in production)."""
    import time

    from feathersim.dashboard.server import _mjpeg

    mgr = SimManager(perception=_all_idle(), render=True)
    mgr.start()
    try:
        deadline = time.time() + 5.0
        while mgr.frame() is None and time.time() < deadline:
            time.sleep(0.05)
        frame = mgr.frame()
        assert frame is not None and frame[:2] == b"\xff\xd8"  # JPEG SOI

        gen = _mjpeg(mgr, fps=1000.0)
        try:
            chunk = next(gen)
        finally:
            gen.close()
        assert b"Content-Type: image/jpeg" in chunk and b"\xff\xd8" in chunk
    finally:
        mgr.stop()
