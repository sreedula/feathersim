"""Multi-robot fleet: task allocation, prioritized collision avoidance, scheduling. [v2 Phase C]"""

from feathersim.fleet.executor import FleetReport, make_perceive_fn, run_fleet
from feathersim.fleet.manager import FleetManager
from feathersim.fleet.scheduling import STRATEGIES, longest_waiting, nearest_done

__all__ = [
    "run_fleet",
    "FleetReport",
    "make_perceive_fn",
    "FleetManager",
    "STRATEGIES",
    "longest_waiting",
    "nearest_done",
]
