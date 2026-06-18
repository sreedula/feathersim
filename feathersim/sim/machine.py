"""Pure, timer-driven machine state machine — no MuJoCo import.

A CNC-style machine cycles ``idle -> running -> done`` on elapsed sim time, then holds at
``done`` until something unloads it (``reset`` -> ``idle``). Deliberately free of any sim
dependency so the transition logic is unit-testable without spinning up MuJoCo (see CLAUDE.md:
"kinematics & perception logic live in pure, testable functions").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MachineState(str, Enum):
    """The three states the autonomy loop reasons about (``str`` mixin → clean labels)."""

    IDLE = "idle"        # loaded with stock, waiting to start a cycle
    RUNNING = "running"  # machining in progress
    DONE = "done"        # finished part ready to be unloaded

    def __str__(self) -> str:  # so f-strings / print show "idle", not "MachineState.IDLE"
        return self.value


def next_state(
    state: MachineState, elapsed: float, idle_time: float, cycle_time: float
) -> MachineState:
    """Pure transition for one machine.

    ``elapsed`` is time spent *in the current state*. Thresholds are inclusive. ``done`` is
    terminal here — only :meth:`Machine.reset` (an unload) leaves it. Returns the (possibly
    unchanged) next state; performs at most one transition so the caller controls time-keeping.
    """
    if state is MachineState.IDLE and elapsed >= idle_time:
        return MachineState.RUNNING
    if state is MachineState.RUNNING and elapsed >= cycle_time:
        return MachineState.DONE
    return state


@dataclass
class Machine:
    """A machine plus the bookkeeping to drive its FSM off a monotonic sim clock.

    ``idle_time`` / ``cycle_time`` are durations in sim seconds. ``phase_start`` is the sim time
    at which the current state began; ``parts_done`` counts finished parts that have been unloaded.
    """

    name: str
    idle_time: float = 2.0
    cycle_time: float = 5.0
    state: MachineState = MachineState.IDLE
    phase_start: float = 0.0
    parts_done: int = 0

    def update(self, now: float) -> MachineState:
        """Advance the FSM to sim time ``now`` and return the resulting state.

        Resets the phase clock to ``now`` on each transition, so one ``update`` performs at most
        one transition — fine because the world steps on a small fixed timestep. ``now`` must be
        monotonic non-decreasing.
        """
        nxt = next_state(self.state, now - self.phase_start, self.idle_time, self.cycle_time)
        if nxt is not self.state:
            self.state = nxt
            self.phase_start = now
        return self.state

    def reset(self, now: float) -> None:
        """Unload a finished part and reload: ``done -> idle``. No-op unless ``done``."""
        if self.state is MachineState.DONE:
            self.parts_done += 1
            self.state = MachineState.IDLE
            self.phase_start = now
