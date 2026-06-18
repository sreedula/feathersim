"""Print per-tick ground-truth sim state. [Phase 1 demo]

Steps the headless MuJoCo world and prints, at a readable cadence, the sim time, each machine's
state, and the robot pose. When a machine reaches ``done`` it is auto-unloaded (``reset`` -> idle)
to simulate the tending the autonomy loop will do in Phase 5 — so you see the full cycle repeat.

Run from the repo root::

    python scripts/print_state.py --machines 3 --seconds 20 --seed 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Make the repo root importable when run as a plain script (python adds scripts/, not the root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from feathersim.sim.machine import MachineState  # noqa: E402
from feathersim.sim.world import World  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Print FeatherSim ground-truth state each tick.")
    ap.add_argument("--machines", type=int, default=3, help="number of machines (1–3)")
    ap.add_argument("--seconds", type=float, default=20.0, help="sim seconds to run")
    ap.add_argument("--seed", type=int, default=0, help="seed for machine timings")
    ap.add_argument("--every", type=float, default=0.5, help="print interval in sim seconds")
    args = ap.parse_args()

    world = World(n_machines=args.machines, seed=args.seed)
    dt = float(world.model.opt.timestep)
    print(f"FeatherSim world — {args.machines} machine(s), seed={args.seed}, dt={dt:.3f}s")
    for m in world.machines:
        print(f"  {m.name}: idle_time={m.idle_time}s  cycle_time={m.cycle_time}s")
    print()

    next_print = 0.0
    unloaded = 0
    while world.time < args.seconds:
        world.step()
        if world.time + 1e-9 >= next_print:
            states = "  ".join(f"{m.name}={m.state.value}" for m in world.machines)
            x, y, yaw = world.robot_pose()
            print(f"t={world.time:6.2f}s  {states}  robot=({x:+.2f},{y:+.2f},{yaw:+.2f})")
            next_print += args.every
            # Unload anything finished *after* showing it, so the done state is always visible.
            for m in world.machines:
                if m.state is MachineState.DONE:
                    m.reset(world.time)
                    unloaded += 1

    print(f"\nDone — simulated {args.seconds:.0f}s; auto-unloaded {unloaded} finished part(s).")


if __name__ == "__main__":
    main()
