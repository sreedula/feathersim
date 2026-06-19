"""Record the multi-robot fleet command-center schematic to an animated GIF for the README. [v2]

Steps the real :class:`FleetController` (3 robots, open floor) and captures the top-down schematic — robots,
planned paths, machines colored by true state — frame by frame. Needs a GL context for the per-robot
perception (set ``MUJOCO_GL=egl``/``osmesa`` on a headless host). Run from repo root::

    python3 scripts/record_fleet_gif.py --out docs/fleet.gif
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from feathersim.dashboard.fleet_manager import FleetSimManager  # noqa: E402
from feathersim.perception.dataset import IMAGE_SIZE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the FeatherSim fleet command center to a GIF.")
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2600, help="sim ticks to record")
    ap.add_argument("--stride", type=int, default=14, help="capture one frame every N ticks")
    ap.add_argument("--fps", type=float, default=14.0)
    ap.add_argument("--difficulty", type=float, default=0.4)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/fleet.gif"))
    args = ap.parse_args()

    mgr = FleetSimManager(render=False, n_robots=args.robots, difficulty=args.difficulty)
    mgr._perc_renderer = mujoco.Renderer(mgr.world.model, IMAGE_SIZE, IMAGE_SIZE)
    frames = []
    try:
        for step in range(args.steps):
            mgr.ctrl.step()
            if step % args.stride == 0:
                frames.append(mgr._schematic_image().convert("P", palette=1))  # web palette → small GIF
    finally:
        mgr._perc_renderer.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    head, *tail = frames
    head.save(args.out, save_all=True, append_images=tail, duration=int(1000 / args.fps), loop=0, optimize=True)
    kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} — {len(frames)} frames, {kb:.0f} KB; fleet delivered {mgr.ctrl.delivered} parts")


if __name__ == "__main__":
    main()
