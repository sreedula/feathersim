"""Tend one machine end-to-end using ONLY the FeatherSim skill SDK. [Phase 3 deliverable]

Run from the repo root::

    python3 examples/tend_one_machine.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from feathersim.sdk import Robot  # noqa: E402
from feathersim.sim.world import World  # noqa: E402

world = World(n_machines=3, seed=0)
robot = Robot(world)

robot.wait_until_done("machine_0")        # let the machine finish a part
part = robot.tend("machine_0")            # move → unload → carry to table → place

print(f"Delivered {part}; table now holds {robot.delivered}.")
print(f"machine_0 parts_done={robot.parts_done('machine_0')}, now {robot.machine_state('machine_0').value}.")
