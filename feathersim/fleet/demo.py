"""Multi-robot fleet demo: run the fleet under each scheduling strategy and compare. [v2 Phase C]

Run with: ``python -m feathersim.fleet.demo`` (or ``make fleet``).

Spins up a 3-robot fleet tending 3 machines on an open floor, with each robot reading the machine
cameras through its own randomized sensor. Runs the headline scheduling strategies back to back and
reports throughput, per-robot delivery counts, and the closest the robots ever came (collision check).
Needs a GL context to render perception (set ``MUJOCO_GL=egl``/``osmesa`` on a headless host).
"""

from __future__ import annotations

import mujoco

from feathersim.fleet import make_perceive_fn, run_fleet
from feathersim.fleet.scheduling import STRATEGIES
from feathersim.perception.dataset import IMAGE_SIZE
from feathersim.perception.infer import Perception
from feathersim.perception.randomize import DomainRandomizer
from feathersim.perception.train import load_or_train_model
from feathersim.sim.world import ROBOT_RADIUS, World

N_ROBOTS = 4
N_MACHINES = 4
TARGET_PARTS = 9


def main() -> None:
    print(f"FeatherSim — multi-robot fleet ({N_ROBOTS} robots, {N_MACHINES} machines)\n")
    model = load_or_train_model()

    results = {}
    for name, strategy in STRATEGIES.items():
        world = World(n_machines=N_MACHINES, seed=0, n_obstacles=0, n_robots=N_ROBOTS)
        try:
            renderer = mujoco.Renderer(world.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
        except Exception as exc:  # no GL context on this host
            raise SystemExit(
                f"Could not create a MuJoCo renderer ({exc}). The fleet renders per-robot perception "
                "and needs a GL context — on a headless host set MUJOCO_GL=egl (or osmesa) and retry."
            ) from exc
        perceive = make_perceive_fn(world, renderer, Perception(model), DomainRandomizer(), seed=0)

        print(f"── strategy: {name} ──")
        try:
            report = run_fleet(
                world, perceive, strategy=strategy, strategy_name=name,
                target_parts=TARGET_PARTS,
                on_event=lambda k, m, part, t: print(f"  [t={t:6.2f}s] robot {k} tended {m} → {part}"),
            )
        finally:
            renderer.close()
        results[name] = report
        print(
            f"  → {report.parts_delivered} parts in {report.sim_seconds:.1f}s = "
            f"{report.throughput_per_min:.1f}/min; per-robot {report.per_robot}; "
            f"closest approach {report.min_robot_separation:.2f}m "
            f"(collision-free: {not report.collided})\n"
        )

    best = max(results.values(), key=lambda r: r.throughput_per_min)
    collision_free = all(not r.collided for r in results.values())
    sep = "no collisions" if collision_free else "COLLISION DETECTED"
    print(f"Bodies stay ≥ {2 * ROBOT_RADIUS:.1f}m apart: {sep}. "
          f"Best throughput: {best.strategy} at {best.throughput_per_min:.1f} parts/min.")


if __name__ == "__main__":
    main()
