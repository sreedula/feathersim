"""Phase 5: the unattended autonomy loop.

The core acceptance — *selection is driven by perceived state, not ground truth* — is proven with a
scripted (fake) perception so it runs everywhere without a GL context: we make perception disagree
with the sim's truth and assert the loop follows perception. A final end-to-end test exercises the
real model + rendering (skipped on a headless host without a GL backend; set ``MUJOCO_GL=egl``).
"""

import mujoco
import pytest

from feathersim.autonomy import AutonomyReport, run_autonomy
from feathersim.autonomy.loop import TendEvent
from feathersim.perception.infer import PerceivedState
from feathersim.sdk.robot import PreconditionError, Robot, SkillError
from feathersim.sim.machine import MachineState
from feathersim.sim.world import World


def _machine(world: World, name: str):
    return next(m for m in world.machines if m.name == name)


def _force_done(world: World, name: str) -> None:
    """Put a machine in ground-truth ``done`` for test setup.

    Sets ``state`` directly and leaves ``phase_start`` stale — safe only because ``done`` is terminal
    under ``Machine.update`` (the FSM never leaves it without a ``reset``), so the forced state survives
    the ``world.step()`` calls that ``tend``'s navigation makes. A future non-terminal-done FSM would
    break this helper.
    """
    _machine(world, name).state = MachineState.DONE


class ScriptedPerception:
    """A stand-in for :class:`Perception` that returns fixed readings, ignoring world/renderer.

    Lets a test pin exactly what the loop *perceives* — including readings that disagree with the
    sim's ground truth — to prove selection is perception-driven.
    """

    def __init__(self, script: dict[str, PerceivedState]) -> None:
        self.script = script
        self.calls = 0

    def perceive(self, world, renderer):
        self.calls += 1
        return dict(self.script)


# --- selection follows PERCEIVED state, not ground truth -----------------------------------------


def test_loop_tends_machine_perceived_done():
    world = World(n_machines=3, seed=0)
    _force_done(world, "machine_0")
    perception = ScriptedPerception(
        {
            "machine_0": PerceivedState(MachineState.DONE, True, 0.99),
            "machine_1": PerceivedState(MachineState.IDLE, False, 0.99),
            "machine_2": PerceivedState(MachineState.RUNNING, True, 0.99),
        }
    )
    robot = Robot(world)

    report = run_autonomy(world, perception, renderer=None, target_parts=1, robot=robot)

    assert report.parts_delivered == 1
    assert report.per_machine["machine_0"] == 1
    assert robot.delivered == ["part_machine_0_0"]
    assert _machine(world, "machine_0").parts_done == 1


def test_loop_follows_perception_over_ground_truth():
    """machine_1 is *actually* done but *perceived* running → the loop must leave it alone."""
    world = World(n_machines=3, seed=0)
    _force_done(world, "machine_0")
    _force_done(world, "machine_1")  # genuinely done...
    perception = ScriptedPerception(
        {
            "machine_0": PerceivedState(MachineState.DONE, True, 0.99),
            "machine_1": PerceivedState(MachineState.RUNNING, True, 0.99),  # ...but perceived running
            "machine_2": PerceivedState(MachineState.IDLE, False, 0.99),
        }
    )

    report = run_autonomy(world, perception, renderer=None, target_parts=1)

    assert report.per_machine["machine_0"] == 1
    assert report.per_machine["machine_1"] == 0
    m1 = _machine(world, "machine_1")
    assert m1.parts_done == 0 and m1.state is MachineState.DONE  # untouched despite being done


def test_loop_does_not_starve_machines():
    """All three machines are perceived done every tick. Oldest-waiting-first scheduling must service
    each one — a naive "always pick the same priority" loop would tend one machine three times and
    starve the others (the bug the demo surfaced)."""
    world = World(n_machines=3, seed=0)
    for name in ("machine_0", "machine_1", "machine_2"):
        _force_done(world, name)
    perception = ScriptedPerception(
        {f"machine_{i}": PerceivedState(MachineState.DONE, True, 0.98) for i in range(3)}
    )

    report = run_autonomy(world, perception, renderer=None, target_parts=3)

    assert report.parts_delivered == 3
    assert report.per_machine == {"machine_0": 1, "machine_1": 1, "machine_2": 1}  # each serviced once


def test_loop_recovers_from_false_positive():
    """Highest-confidence reading is a false positive (machine idle); the loop falls through to the
    genuinely-done machine instead of crashing or livelocking."""
    world = World(n_machines=2, seed=0)
    _force_done(world, "machine_1")  # only machine_1 is really done
    perception = ScriptedPerception(
        {
            "machine_0": PerceivedState(MachineState.DONE, True, 0.90),  # false positive (idle)
            "machine_1": PerceivedState(MachineState.DONE, True, 0.60),  # real, lower confidence
        }
    )
    robot = Robot(world)

    report = run_autonomy(world, perception, renderer=None, target_parts=1, robot=robot)

    assert report.per_machine["machine_1"] == 1
    assert report.per_machine["machine_0"] == 0
    assert _machine(world, "machine_0").parts_done == 0  # never unloaded a not-done machine
    assert robot.delivered == ["part_machine_1_0"]


class _StubRobot:
    """Minimal robot stand-in — the loop only ever calls ``tend``. ``error`` is raised on every call."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def tend(self, machine: str) -> str:
        raise self.error


def test_loop_terminates_when_every_reading_is_a_false_positive():
    """Perception persistently (and wrongly) reports machines done. No tend ever succeeds, so the run
    must still terminate via the sim-time budget with nothing delivered — not hang or crash."""
    world = World(n_machines=2, seed=0)
    perception = ScriptedPerception(
        {f"machine_{i}": PerceivedState(MachineState.DONE, True, 0.99) for i in range(2)}
    )
    robot = _StubRobot(PreconditionError("not actually done"))

    report = run_autonomy(
        world, perception, renderer=None, target_parts=5, max_sim_seconds=2.0,
        wait_seconds=0.5, robot=robot,
    )

    assert report.parts_delivered == 0
    assert report.sim_seconds >= 2.0  # advanced to the budget rather than spinning in place


def test_loop_does_not_swallow_navigation_failures():
    """A genuine navigation failure (plain SkillError, not a precondition miss) must propagate — an
    unattended run should fail loudly on a real nav regression, not silently lose throughput."""
    world = World(n_machines=1, seed=0)
    perception = ScriptedPerception({"machine_0": PerceivedState(MachineState.DONE, True, 0.99)})
    robot = _StubRobot(SkillError("could not reach 'machine_0': 0.500m off"))

    with pytest.raises(SkillError) as excinfo:
        run_autonomy(world, perception, renderer=None, target_parts=1, robot=robot)
    assert not isinstance(excinfo.value, PreconditionError)  # nav failure, not a swallowed precondition


def test_min_confidence_gates_low_confidence_readings():
    """A low-confidence ``done`` reading is dropped before it costs a tending trip, even though the
    machine is genuinely done; a confident reading on another machine is serviced instead."""
    world = World(n_machines=2, seed=0)
    _force_done(world, "machine_0")
    _force_done(world, "machine_1")
    perception = ScriptedPerception(
        {
            "machine_0": PerceivedState(MachineState.DONE, True, 0.40),  # below the gate
            "machine_1": PerceivedState(MachineState.DONE, True, 0.95),
        }
    )

    report = run_autonomy(
        world, perception, renderer=None, target_parts=1, min_confidence=0.5
    )

    assert report.per_machine == {"machine_0": 0, "machine_1": 1}


def test_loop_waits_when_nothing_perceived_done():
    """Nothing is ready at first; the loop advances the sim (no tend) until the budget elapses."""
    world = World(n_machines=2, seed=0)
    perception = ScriptedPerception(
        {
            "machine_0": PerceivedState(MachineState.RUNNING, True, 0.99),
            "machine_1": PerceivedState(MachineState.IDLE, False, 0.99),
        }
    )

    report = run_autonomy(world, perception, renderer=None, target_parts=1, max_sim_seconds=0.5)

    assert report.parts_delivered == 0
    assert report.sim_seconds > 0  # it advanced sim time while waiting
    assert perception.calls >= 1


# --- report bookkeeping --------------------------------------------------------------------------


def test_report_throughput():
    report = AutonomyReport(
        parts_delivered=6,
        sim_seconds=120.0,
        per_machine={"machine_0": 6},
        events=[],
    )
    assert report.throughput_per_min == pytest.approx(3.0)
    assert AutonomyReport(0, 0.0, {}).throughput_per_min == 0.0  # no divide-by-zero at t=0


# --- end-to-end with the real perception model (needs a GL context) ------------------------------


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


@pytest.mark.skipif(
    not _rendering_available(),
    reason="MuJoCo rendering unavailable (set MUJOCO_GL=egl/osmesa on a headless host)",
)
def test_autonomy_end_to_end_with_real_perception():
    from feathersim.perception.dataset import IMAGE_SIZE
    from feathersim.perception.infer import Perception
    from feathersim.perception.train import load_or_train_model

    world = World(n_machines=3, seed=0)
    renderer = mujoco.Renderer(world.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
    perception = Perception(load_or_train_model())
    try:
        report = run_autonomy(
            world, perception, renderer, target_parts=3, max_sim_seconds=120.0
        )
    finally:
        renderer.close()

    assert report.parts_delivered == 3  # tended unattended to target
    assert sum(report.per_machine.values()) == 3
    assert report.sim_seconds > 0 and report.throughput_per_min > 0
    assert all(isinstance(e, TendEvent) for e in report.events)
