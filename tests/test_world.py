"""Phase 1: MuJoCo world builds, steps deterministically, and drives machines to done."""

import pytest

from feathersim.sim.machine import MachineState
from feathersim.sim.world import World, build_mjcf


def _trace(world: World, steps: int) -> list[tuple]:
    """Collect a per-step trace of (time, machine-state-values) for comparison."""
    out = []
    for _ in range(steps):
        world.step()
        out.append((round(world.time, 6), tuple(s.value for s in world.states().values())))
    return out


@pytest.mark.parametrize("n", [1, 2, 3])
def test_world_builds(n):
    w = World(n_machines=n, seed=0)
    assert len(w.machines) == n
    assert w.time == 0.0
    assert all(m.state is MachineState.IDLE for m in w.machines)


def test_invalid_machine_count_rejected():
    with pytest.raises(ValueError):
        World(n_machines=0)
    with pytest.raises(ValueError):
        World(n_machines=5)            # 1–4 supported (one color per machine)


def test_step_advances_time():
    w = World(n_machines=2, seed=0)
    w.step()
    assert w.time == pytest.approx(w.model.opt.timestep)
    w.step()
    assert w.time == pytest.approx(2 * w.model.opt.timestep)


def test_robot_stays_at_origin_unactuated():
    # Phase 1: the base has planar joints but no actuation, so it must not drift.
    w = World(n_machines=3, seed=0)
    for _ in range(500):
        w.step()
    x, y, yaw = w.robot_pose()
    assert (abs(x), abs(y), abs(yaw)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_machines_reach_done():
    # idle_time<2.5 and cycle_time<6.0, so 10s is enough for every machine to finish once.
    w = World(n_machines=3, seed=0)
    seen_done = {m.name: False for m in w.machines}
    while w.time < 10.0:
        w.step()
        for m in w.machines:
            if m.state is MachineState.DONE:
                seen_done[m.name] = True
    assert all(seen_done.values()), seen_done


def test_same_seed_is_deterministic():
    a, b = World(n_machines=3, seed=7), World(n_machines=3, seed=7)
    # Identical machine timings...
    assert [(m.idle_time, m.cycle_time) for m in a.machines] == \
           [(m.idle_time, m.cycle_time) for m in b.machines]
    # ...and identical state traces.
    assert _trace(a, 600) == _trace(b, 600)


def test_seed_changes_timings():
    a = [(m.idle_time, m.cycle_time) for m in World(n_machines=3, seed=1).machines]
    b = [(m.idle_time, m.cycle_time) for m in World(n_machines=3, seed=2).machines]
    assert a != b


def test_mjcf_has_expected_bodies():
    xml = build_mjcf(2)
    assert 'name="robot_0"' in xml
    assert 'name="table"' in xml
    assert xml.count("<body name=\"machine_") == 2


def test_mjcf_builds_a_fleet():
    xml = build_mjcf(3, 0, 3)
    assert all(f'name="robot_{k}"' in xml for k in range(3))
    assert all(f'base_x_{k}' in xml for k in range(3))


def test_arm_slews_to_target_and_holds_the_base():
    from feathersim.sim.world import ARM_REACH, ARM_REST
    w = World(n_machines=1, seed=0, n_robots=1)
    assert w.arm_at(0, ARM_REST)                 # starts tucked in carry pose
    w.set_arm_target(0, ARM_REACH)
    for _ in range(80):
        w.step()
    assert w.arm_at(0, ARM_REACH)                # reached the grasp pose
    assert w.robot_pose(0) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)  # gravcomp → base undisturbed
    w.set_arm_target(0, ARM_REST)
    for _ in range(80):
        w.step()
    assert w.arm_at(0, ARM_REST)                 # retracted
