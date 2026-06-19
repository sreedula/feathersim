"""Fleet task allocation: assign `done` machines to robots, never double-booking one. [v2 Phase C]

The manager is the single source of truth for *which robot owns which machine*. A machine is locked to a
robot from assignment until the part is placed, so two robots can never target the same machine. It also
tracks when each machine was first seen `done` (for the longest-waiting strategy).
"""

from __future__ import annotations

from feathersim.fleet.scheduling import Strategy


class FleetManager:
    """Allocates perceived-`done` machines to robots under a pluggable :data:`Strategy`."""

    def __init__(self, machine_xy: dict[str, tuple[float, float]], strategy: Strategy) -> None:
        self._machine_xy = machine_xy
        self.strategy = strategy
        self.assignments: dict[str, int] = {}     # machine name -> owning robot id
        self.done_since: dict[str, float] = {}     # machine name -> sim time first observed done

    def observe(self, perceived_done: list[str], now: float) -> None:
        """Record fresh `done` observations (any robot's) so longest-waiting has a timestamp to sort on."""
        for m in perceived_done:
            self.done_since.setdefault(m, now)

    def assign(self, robot_id: int, robot_xy: tuple[float, float], perceived_done: list[str]) -> str | None:
        """Claim the best *unassigned* machine this robot perceives `done`, or ``None`` if there's none."""
        candidates = [m for m in perceived_done if m not in self.assignments]
        if not candidates:
            return None
        machine = self.strategy(
            robot_xy, candidates, done_since=self.done_since, machine_xy=self._machine_xy.__getitem__
        )
        self.assignments[machine] = robot_id
        return machine

    def release(self, machine: str) -> None:
        """Free a machine once its part is placed (it's no longer `done`): drop the lock and its timer."""
        self.assignments.pop(machine, None)
        self.done_since.pop(machine, None)
