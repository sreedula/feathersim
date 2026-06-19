"""Run the autonomy loop on the hand-coded controller vs. the learned policy, and compare. [v2 Phase D]

Run with: ``python -m feathersim.policy.demo``.

Drives the *same* unattended single-robot loop twice — once with the hand-coded P-controller, once with
the behavior-cloned policy as a drop-in controller — and reports throughput for each, showing the learned
brain runs the whole loop end-to-end and matches the expert. Needs a GL context (set ``MUJOCO_GL=egl``).
"""

from __future__ import annotations

import mujoco

from feathersim.autonomy import run_autonomy
from feathersim.perception.dataset import IMAGE_SIZE
from feathersim.perception.infer import Perception
from feathersim.perception.train import load_or_train_model
from feathersim.policy.policy import PolicyController
from feathersim.policy.train import load_or_train_policy
from feathersim.sdk.robot import Robot
from feathersim.sim.world import World

TARGET_PARTS = 6


def _make_renderer(world):
    try:
        return mujoco.Renderer(world.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
    except Exception as exc:  # no GL context on this host
        raise SystemExit(
            "The policy demo renders the perception camera and needs a GL context — "
            "on a headless host set MUJOCO_GL=egl (or osmesa) and retry."
        ) from exc


def _run(controller, label: str, perception) -> float:
    world = World(n_machines=3, seed=0, n_obstacles=0, n_robots=1)
    renderer = _make_renderer(world)
    robot = Robot(world, controller=controller)
    try:
        report = run_autonomy(world, perception, renderer, target_parts=TARGET_PARTS, robot=robot)
    finally:
        renderer.close()
    print(
        f"  {label:18s} {report.parts_delivered} parts in {report.sim_seconds:5.1f}s "
        f"→ {report.throughput_per_min:.1f} parts/min"
    )
    return report.throughput_per_min


def main() -> None:
    print("FeatherSim — hand-coded controller vs. behavior-cloned policy (same autonomy loop)\n")
    from feathersim.control.go_to_pose import velocity_command

    perception = Perception(load_or_train_model())
    policy = PolicyController(load_or_train_policy())

    expert = _run(velocity_command, "hand-coded:", perception)
    learned = _run(policy, "learned policy:", perception)
    print(f"\nLearned policy throughput is {learned / expert:.0%} of the hand-coded controller's.")


if __name__ == "__main__":
    main()
