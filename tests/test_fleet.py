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


def test_plan_leg_ignores_robots_and_routes_static_only():
    """A* now plans around the *static* world only — other robots are handled reactively by ORCA, not by
    the planner. So a robot is never left path-less because a peer parked on its goal (the old planning
    deadly-embrace): a path through open floor always exists regardless of where the other robots sit."""
    from feathersim.fleet.executor import _plan_leg

    world = World(n_machines=3, seed=0, n_obstacles=0, n_robots=4)
    goal = (0.6, 0.4)                                  # open floor (no static obstacle here)
    world.set_base_pose(0.0, -0.5, 0.0, robot=0)
    world.set_base_pose(goal[0], goal[1], 0.0, robot=1)  # a peer sits right on robot 0's goal — irrelevant now
    path = _plan_leg(world, 0, goal)
    assert path is not None and path[-1] == pytest.approx(goal, abs=0.2)


@pytest.mark.parametrize("seed", range(8))
def test_fleet_scales_to_four_robots(seed):
    """The headline scaled floor: 4 robots tending 4 machines unattended. The hard case for coordination
    (clusters that wedged the old symmetric backstop). ORCA must keep it both collision-free *and* live
    (every part delivered) across seeds, with no slow near-wedge."""
    world = World(n_machines=4, seed=seed, n_obstacles=0, n_robots=4)
    report = run_fleet(
        world, _ground_truth_perceive(world), strategy=longest_waiting, strategy_name="longest_waiting",
        target_parts=8, max_sim_seconds=90,   # ORCA finishes ~30 s; tight enough to trip on a wedge regression
    )
    assert report.completed and report.parts_delivered == 8
    assert not report.collided and report.min_robot_separation >= 2 * ROBOT_RADIUS


def test_fleet_composes_with_static_obstacles():
    """The fleet also avoids the Phase-B static pillars while avoiding each other (2 robots). This tighter
    cell used to occasionally wedge the heuristic; ORCA (deadlock-free) completes it on every seed."""
    world = World(n_machines=3, seed=0, n_obstacles=2, n_robots=2)
    report = run_fleet(
        world, _ground_truth_perceive(world), strategy=longest_waiting, target_parts=6, max_sim_seconds=120,
    )
    assert report.completed and not report.collided and report.min_robot_separation >= 2 * ROBOT_RADIUS


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


def test_bench_fleet_smoke():
    """The benchmark script's per-config aggregator runs and reports a collision-free completed run."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    from bench_fleet import _bench_one

    row = _bench_one(2, 2, 0, 4, "longest_waiting", seeds=1)
    assert row["collision_free"] and row["completion_rate"] == 1.0
    assert row["throughput_per_min_mean"] > 0 and set(row) >= {"config", "strategy", "worst_sep"}


def test_fleet_orca_deflects_head_on_robots():
    """ORCA in the *fleet driving loop* (not just the pure module): two robots driven straight at each
    other's positions must deflect, never breach 2·radius, and pass through to swap sides."""
    from feathersim.fleet.executor import FleetController, _min_separation

    world = World(n_machines=2, seed=0, n_obstacles=0, n_robots=2)
    ctrl = FleetController(world, _ground_truth_perceive(world), strategy=longest_waiting)
    world.set_base_pose(-1.2, 0.0, 0.0, robot=0)
    world.set_base_pose(1.2, 0.0, 0.0, robot=1)
    ctrl.phase = ["to_table", "to_table"]                 # both driving → reciprocal ORCA peers
    ctrl.goal = [(1.2, 0.0, 0.0), (-1.2, 0.0, 0.0)]       # head-on: each aims at the other's start
    ctrl.path = [None, None]
    worst = math.inf
    for _ in range(900):
        ctrl._drive(0)
        ctrl._drive(1)
        world.step()
        worst = min(worst, _min_separation(world))
    assert worst >= 2 * ROBOT_RADIUS                       # never overlapped
    assert world.robot_pose(0)[0] > 0.5 and world.robot_pose(1)[0] < -0.5  # passed through, swapped sides
