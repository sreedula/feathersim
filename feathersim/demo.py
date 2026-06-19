"""FeatherSim demo entry point — the unattended autonomy loop. [Phase 5]

Run with: ``python -m feathersim.demo`` (or ``make demo``).

Spins up the sim world, loads (or trains) the perception model, and runs the headless machine-tending
loop: perceive which machine is *done* → navigate → unload → carry → place, repeat, with no manual
input. Prints each tend as it happens and a throughput/uptime summary at the end. Needs a GL context
to render the perception camera (set ``MUJOCO_GL=egl``/``osmesa`` on a headless host).
"""

from __future__ import annotations

import mujoco

from feathersim.autonomy import TendEvent, run_autonomy
from feathersim.perception.dataset import IMAGE_SIZE
from feathersim.perception.infer import Perception
from feathersim.perception.train import load_or_train_model
from feathersim.sdk.robot import Robot
from feathersim.sim.world import World

N_MACHINES = 3
N_OBSTACLES = 2
TARGET_PARTS = 6


def _print_tend(event: TendEvent) -> None:
    print(
        f"  [t={event.sim_time:6.2f}s] tended {event.machine} "
        f"→ delivered {event.part} (perceived done @ {event.confidence:.0%} conf)"
    )


def main() -> None:
    print(
        f"FeatherSim — unattended autonomy loop "
        f"({N_MACHINES} machines, {N_OBSTACLES} obstacles, target {TARGET_PARTS} parts)\n"
    )

    world = World(n_machines=N_MACHINES, seed=0, n_obstacles=N_OBSTACLES)
    try:
        renderer = mujoco.Renderer(world.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
    except Exception as exc:  # no GL context on this host — perception can't render its camera
        raise SystemExit(
            f"Could not create a MuJoCo renderer ({exc}). The demo renders the perception camera and "
            "needs a GL context — on a headless host set MUJOCO_GL=egl (or osmesa) and retry."
        ) from exc
    perception = Perception(load_or_train_model())
    robot = Robot(world, plan=True)  # route around the obstacles via A*
    try:
        report = run_autonomy(
            world, perception, renderer, target_parts=TARGET_PARTS, robot=robot, on_event=_print_tend
        )
    finally:
        renderer.close()

    print(
        f"\nDelivered {report.parts_delivered} parts in {report.sim_seconds:.1f}s sim uptime "
        f"→ {report.throughput_per_min:.1f} parts/min"
    )
    per_machine = ", ".join(f"{name}: {n}" for name, n in sorted(report.per_machine.items()))
    print(f"Per machine — {per_machine}")


if __name__ == "__main__":
    main()
