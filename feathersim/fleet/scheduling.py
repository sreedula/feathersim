"""Task-allocation scheduling strategies for the fleet — pure, no sim import. [v2 Phase C]

A strategy picks which unassigned, perceived-`done` machine a free robot should tend next. Two are
provided so throughput can be compared; both share a signature so they're interchangeable.
"""

from __future__ import annotations

import math
from collections.abc import Callable

XY = tuple[float, float]
# (robot position, candidate machine names) + context kwargs → chosen machine name.
Strategy = Callable[..., str]


def longest_waiting(robot_xy: XY, candidates: list[str], *, done_since: dict[str, float],
                    machine_xy: Callable[[str], XY]) -> str:
    """Tend the machine that's been `done` longest (smallest first-seen-done time). Starvation-free."""
    return min(candidates, key=lambda m: (done_since.get(m, math.inf), m))


def nearest_done(robot_xy: XY, candidates: list[str], *, done_since: dict[str, float],
                 machine_xy: Callable[[str], XY]) -> str:
    """Tend the `done` machine closest to the robot — minimizes travel, maximizes a single robot's rate."""
    return min(candidates, key=lambda m: (math.dist(robot_xy, machine_xy(m)), m))


STRATEGIES: dict[str, Strategy] = {
    "longest_waiting": longest_waiting,
    "nearest_done": nearest_done,
}
