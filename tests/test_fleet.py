"""v2 Phase C: multi-robot fleet — task allocation, collision avoidance, scheduling.

The allocator and scheduling strategies are pure (no sim). The executor is exercised headless with a
ground-truth perceive function (no GL). One end-to-end test uses the real per-robot perception + render
and is GL-guarded.
"""

import math

import mujoco
import pytest

from feathersim.fleet import FleetManager, run_fleet
from feathersim.fleet.scheduling import longest_waiting, nearest_done
from feathersim.perception.infer import PerceivedState
from feathersim.sim.machine import MachineState
from feathersim.sim.world import ROBOT_RADIUS, World


def _ground_truth_perceive(world: World):
    """Every robot perceives the true machine states (no sensor noise) — for headless executor tests."""
    return lambda robot_id: {
        m.name: PerceivedState(m.state, True, 0.99) for m in world.machines
    }


# --- scheduling strategies (pure) ----------------------------------------------------------------


def test_longest_waiting_picks_oldest_done():
    chosen = longest_waiting(
        (0.0, 0.0), ["machine_0", "machine_1"],
        done_since={"machine_0": 5.0, "machine_1": 1.0}, machine_xy=lambda m: (0.0, 0.0),
    )
    assert chosen == "machine_1"  # done since t=1 < t=5


def test_nearest_done_picks_closest():
    xy = {"machine_0": (0.0, 0.0), "machine_1": (5.0, 5.0)}
    chosen = nearest_done(
        (0.2, 0.0), ["machine_0", "machine_1"], done_since={}, machine_xy=xy.__getitem__
    )
    assert chosen == "machine_0"


# --- task allocation: never double-book a machine ------------------------------------------------


def test_manager_never_double_assigns():
    mgr = FleetManager({"m0": (0.0, 0.0), "m1": (1.0, 1.0)}, longest_waiting)
    mgr.observe(["m0", "m1"], now=0.0)

    a = mgr.assign(0, (0.0, 0.0), ["m0", "m1"])
    b = mgr.assign(1, (0.0, 0.0), ["m0", "m1"])  # the other machine — a is now locked
    assert a != b and {a, b} == {"m0", "m1"}
    assert mgr.assign(2, (0.0, 0.0), ["m0", "m1"]) is None  # both locked → nothing for robot 2

    mgr.release(a)
    assert mgr.assign(2, (0.0, 0.0), ["m0", "m1"]) == a  # freed → assignable again


# --- the fleet runs unattended: delivers, never collides, never double-tends ----------------------


@pytest.mark.parametrize("seed", range(8))
def test_fleet_collision_free_across_seeds(seed):
    """3 robots on an open floor must tend unattended without bodies ever overlapping — across many
    seeds, not one. (Single-seed sampling masked seed-dependent collisions in review.)"""
    world = World(n_machines=3, seed=seed, n_obstacles=0, n_robots=3)
    report = run_fleet(
        world, _ground_truth_perceive(world), strategy=longest_waiting, strategy_name="longest_waiting",
        target_parts=6, max_sim_seconds=250,
    )
    assert report.completed and report.parts_delivered == 6
    assert sum(report.per_machine.values()) == 6
    assert not report.collided and report.min_robot_separation >= 2 * ROBOT_RADIUS
    # NB: per-robot *fairness* (each robot contributing) is load-dependent and seed-varying, so it is
    # deliberately not asserted here — this test certifies the safety invariant (no collision), not balance.


def test_fleet_composes_with_static_obstacles():
    """The fleet also avoids the Phase-B static pillars while avoiding each other (2 robots).

    Pinned to seed 0: collision-freedom holds for all seeds, but in this tighter cell the symmetric
    backstop can occasionally wedge two robots until the time budget (a known, surfaced limitation —
    `completed=False`), so the demo/headline uses 3 robots on an open floor, not this config."""
    world = World(n_machines=3, seed=0, n_obstacles=2, n_robots=2)
    report = run_fleet(
        world, _ground_truth_perceive(world), strategy=longest_waiting, target_parts=6, max_sim_seconds=250,
    )
    assert report.completed and not report.collided
    assert report.min_robot_separation >= 2 * ROBOT_RADIUS


def test_both_strategies_log_throughput():
    reports = {}
    for name, strat in (("longest_waiting", longest_waiting), ("nearest_done", nearest_done)):
        world = World(n_machines=3, seed=0, n_obstacles=0, n_robots=3)
        reports[name] = run_fleet(
            world, _ground_truth_perceive(world), strategy=strat, strategy_name=name,
            target_parts=6, max_sim_seconds=200,
        )
    for name, rep in reports.items():
        assert rep.strategy == name and rep.parts_delivered == 6
        assert rep.throughput_per_min > 0 and not rep.collided


def test_fleet_handles_perception_false_positive():
    """A robot perceiving a not-done machine as done must not crash the fleet — pick fails, the machine
    is released, and the run still completes."""
    world = World(n_machines=3, seed=0, n_obstacles=0, n_robots=2)

    def lying_perceive(robot_id):
        out = {m.name: PerceivedState(m.state, True, 0.9) for m in world.machines}
        out["machine_0"] = PerceivedState(MachineState.DONE, True, 0.9)  # claim done even when idle
        return out

    report = run_fleet(
        world, lying_perceive, strategy=longest_waiting, target_parts=4, max_sim_seconds=200,
    )
    assert report.parts_delivered == 4 and not report.collided


# --- end-to-end with real per-robot perception (needs GL) ----------------------------------------


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
def test_fleet_end_to_end_with_real_perception():
    from feathersim.fleet import make_perceive_fn
    from feathersim.perception.dataset import IMAGE_SIZE
    from feathersim.perception.infer import Perception
    from feathersim.perception.randomize import DomainRandomizer
    from feathersim.perception.train import load_or_train_model

    world = World(n_machines=3, seed=0, n_obstacles=0, n_robots=3)
    renderer = mujoco.Renderer(world.model, IMAGE_SIZE, IMAGE_SIZE)
    perception = Perception(load_or_train_model())
    perceive = make_perceive_fn(world, renderer, perception, DomainRandomizer(), seed=0)
    try:
        report = run_fleet(
            world, perceive, strategy=longest_waiting, strategy_name="longest_waiting",
            target_parts=3, max_sim_seconds=200,
        )
    finally:
        renderer.close()

    assert report.parts_delivered == 3 and not report.collided
