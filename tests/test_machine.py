"""Phase 1: pure machine-FSM transitions (no MuJoCo)."""

from feathersim.sim.machine import Machine, MachineState, next_state


# --- next_state: pure transition function ------------------------------------------------

def test_idle_starts_running_at_threshold():
    assert next_state(MachineState.IDLE, 1.99, idle_time=2.0, cycle_time=5.0) is MachineState.IDLE
    assert next_state(MachineState.IDLE, 2.0, idle_time=2.0, cycle_time=5.0) is MachineState.RUNNING


def test_running_finishes_at_threshold():
    assert next_state(MachineState.RUNNING, 4.9, idle_time=2.0, cycle_time=5.0) is MachineState.RUNNING
    assert next_state(MachineState.RUNNING, 5.0, idle_time=2.0, cycle_time=5.0) is MachineState.DONE


def test_done_is_terminal():
    assert next_state(MachineState.DONE, 1000.0, idle_time=2.0, cycle_time=5.0) is MachineState.DONE


def test_at_most_one_transition_per_call():
    # Even with huge elapsed, idle advances only to running (not straight to done).
    assert next_state(MachineState.IDLE, 1e6, idle_time=2.0, cycle_time=5.0) is MachineState.RUNNING


# --- Machine: FSM driven off a monotonic clock -------------------------------------------

def test_full_cycle_on_clock():
    m = Machine("m0", idle_time=2.0, cycle_time=5.0)
    assert m.update(0.0) is MachineState.IDLE
    assert m.update(1.9) is MachineState.IDLE
    assert m.update(2.0) is MachineState.RUNNING   # idle -> running, clock resets to now
    assert m.update(6.9) is MachineState.RUNNING   # 4.9s into the cycle
    assert m.update(7.0) is MachineState.DONE       # 5.0s into the cycle -> done
    assert m.update(100.0) is MachineState.DONE     # holds at done


def test_reset_unloads_and_reloads():
    m = Machine("m0", idle_time=2.0, cycle_time=5.0)
    m.update(2.0)   # running
    m.update(7.0)   # done
    assert m.parts_done == 0
    m.reset(7.0)
    assert m.state is MachineState.IDLE
    assert m.parts_done == 1
    assert m.phase_start == 7.0
    # And it cycles again from the reset moment.
    assert m.update(8.9) is MachineState.IDLE
    assert m.update(9.0) is MachineState.RUNNING


def test_reset_is_noop_unless_done():
    m = Machine("m0", idle_time=2.0, cycle_time=5.0)
    m.reset(1.0)  # still idle
    assert m.state is MachineState.IDLE
    assert m.parts_done == 0
    m.update(2.0)  # running
    m.reset(3.0)   # not done -> no-op
    assert m.state is MachineState.RUNNING
    assert m.parts_done == 0


def test_state_label_is_clean_string():
    assert str(MachineState.IDLE) == "idle"
    assert MachineState.DONE.value == "done"
    assert MachineState.RUNNING == "running"  # str mixin
