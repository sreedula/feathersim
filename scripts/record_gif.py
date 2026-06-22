"""Record the unattended autonomy loop to an animated GIF for the README. [Definition of done]

Runs the real :func:`feathersim.autonomy.run_autonomy` and captures the overhead camera every few sim
steps, then encodes the frames as a GIF. Frame capture is done by intercepting ``world.step`` so the
autonomy policy itself isn't duplicated — we record exactly what the loop does.

Needs a GL context to render (set ``MUJOCO_GL=egl``/``osmesa`` on a headless host). Run from repo root::

    python3 scripts/record_gif.py --parts 3 --out docs/autonomy.gif
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from feathersim.autonomy import run_autonomy  # noqa: E402
from feathersim.perception.dataset import IMAGE_SIZE  # noqa: E402
from feathersim.perception.infer import Perception  # noqa: E402
from feathersim.perception.train import load_or_train_model  # noqa: E402
from feathersim.sdk.robot import Robot  # noqa: E402
from feathersim.sim.world import World  # noqa: E402


class _FrameRecorder:
    """Intercepts ``world.step`` to grab an overhead frame every ``stride`` steps."""

    def __init__(self, world: World, renderer: mujoco.Renderer, stride: int) -> None:
        self.world = world
        self.renderer = renderer
        self.camera = world.overview_camera()
        self.stride = stride
        self.frames: list[Image.Image] = []
        self._n = 0
        self._inner_step = world.step
        world.step = self._step  # type: ignore[method-assign]

    def _step(self) -> None:
        self._inner_step()
        self._n += 1
        if self._n % self.stride == 0:
            self.frames.append(Image.fromarray(self.world.render(self.renderer, self.camera)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the FeatherSim autonomy loop to a GIF.")
    ap.add_argument("--machines", type=int, default=3, help="number of machines (1–3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", type=int, default=3, help="parts to deliver before stopping")
    ap.add_argument("--size", type=int, default=360, help="GIF frame size (px)")
    ap.add_argument("--stride", type=int, default=8, help="capture one frame every N sim steps")
    ap.add_argument("--fps", type=float, default=15.0, help="GIF playback frames per second")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/autonomy.gif"))
    args = ap.parse_args()

    world = World(n_machines=args.machines, seed=args.seed)
    perception = Perception(load_or_train_model())
    feed = mujoco.Renderer(world.model, height=args.size, width=args.size)
    perc = mujoco.Renderer(world.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
    recorder = _FrameRecorder(world, feed, args.stride)
    robot = Robot(world, plan=True, animate_arm=True)  # show the arm reach into the machine / over the table
    try:
        report = run_autonomy(world, perception, perc, target_parts=args.parts, robot=robot)
    finally:
        feed.close()
        perc.close()

    if not recorder.frames:
        raise SystemExit("no frames captured")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    head, *tail = recorder.frames
    head.save(
        args.out,
        save_all=True,
        append_images=tail,
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    kb = args.out.stat().st_size / 1024
    print(
        f"wrote {args.out} — {len(recorder.frames)} frames, {kb:.0f} KB; "
        f"delivered {report.parts_delivered} parts in {report.sim_seconds:.1f}s sim "
        f"({report.throughput_per_min:.1f}/min)"
    )


if __name__ == "__main__":
    main()
