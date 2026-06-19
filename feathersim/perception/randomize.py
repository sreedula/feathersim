"""Domain randomization for robust perception. [v2 Phase A]

Two stages, kept separable so each is independently testable:

- **Scene** (mutates the MuJoCo model before a render): randomized scene-light position / intensity /
  color tint, and a per-machine occluder box partially blocking the status light. Sampled as pure data
  (:class:`SceneRandomization`); :func:`apply_scene` pushes it onto a :class:`~feathersim.sim.world.World`.
- **Sensor** (pure numpy on the rendered uint8 crop): additive Gaussian noise + directional motion blur.

The status-light *label color* (gray/amber/green) is never randomized — that is the signal. Occluder
size/offset are bounded so the light is never *fully* hidden, keeping every sample solvable. All
sampling is driven by a passed ``np.random.Generator`` so datasets are reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- pure image-space corruption -----------------------------------------------------------------


def gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise (std ``sigma`` on the 0–255 scale); returns a new uint8 image."""
    if sigma <= 0.0:
        return image.copy()  # always hand back a fresh array (no shared-reference surprise)
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def motion_blur(image: np.ndarray, length: int, angle: float) -> np.ndarray:
    """Directional blur: average ``length`` copies shifted along ``angle`` (radians). ``length<=1`` is
    a no-op. Pure and deterministic given its args (shifts use wrap-around — fine for a centered cue)."""
    if length <= 1:
        return image
    dx, dy = math.cos(angle), math.sin(angle)
    acc = np.zeros(image.shape, dtype=np.float32)
    for t in np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length):
        shifted = np.roll(image, (int(round(dy * t)), int(round(dx * t))), axis=(0, 1))
        acc += shifted
    return (acc / length).astype(np.uint8)


# --- scene randomization (sampled as data, applied to a World) ------------------------------------


@dataclass(frozen=True)
class Occluder:
    """A status-light occluder placement (offsets from the geom's base pose; ``size`` is a half-extent)."""

    dx: float
    dz: float
    size: float


@dataclass(frozen=True)
class SceneRandomization:
    """A fully-specified randomized scene for one sample: light + per-machine occluders (``None`` = none)."""

    light_pos_xy: tuple[float, float]
    light_diffuse: tuple[float, float, float]
    occluders: tuple[Occluder | None, ...]


@dataclass(frozen=True)
class DomainRandomizer:
    """Config + samplers for domain randomization. Defaults are tuned to be hard but solvable."""

    # scene-light
    light_xy_jitter: float = 1.6                      # ± metres on the light's x and y
    light_intensity: tuple[float, float] = (0.45, 1.05)   # diffuse magnitude (default clean = 0.9)
    light_tint: float = 0.22                          # max per-channel deviation around the magnitude
    # occluder
    occluder_prob: float = 0.5
    # Half-extent + offset bounded so the occluder only ever *partially* covers the light. NB the box
    # sits ~0.12 m in front of the light, so its projected shadow is larger than its physical size;
    # these bounds are tuned empirically (worst case leaves ~13 of 93 light px visible — see
    # tests/test_randomize.py / LEARNINGS.md), not derived from the half-extent alone.
    occluder_size: tuple[float, float] = (0.03, 0.09)
    occluder_offset: float = 0.10                     # max |dx|, |dz| lateral/vertical offset
    # sensor
    noise_sigma: tuple[float, float] = (4.0, 26.0)
    blur_prob: float = 0.55
    blur_length: tuple[int, int] = (3, 9)             # kernel length in px (inclusive)

    def sample_scene(self, rng: np.random.Generator, n_machines: int) -> SceneRandomization:
        """Sample a randomized scene (light + one occluder slot per machine)."""
        x, y = rng.uniform(-self.light_xy_jitter, self.light_xy_jitter, size=2)
        base = float(rng.uniform(*self.light_intensity))
        d = rng.uniform(-self.light_tint, self.light_tint, size=3)
        diffuse = (
            float(np.clip(base + d[0], 0.1, 1.4)),
            float(np.clip(base + d[1], 0.1, 1.4)),
            float(np.clip(base + d[2], 0.1, 1.4)),
        )
        occluders = tuple(
            (
                Occluder(
                    dx=float(rng.uniform(-self.occluder_offset, self.occluder_offset)),
                    dz=float(rng.uniform(-self.occluder_offset, self.occluder_offset)),
                    size=float(rng.uniform(*self.occluder_size)),
                )
                if rng.random() < self.occluder_prob
                else None
            )
            for _ in range(n_machines)
        )
        return SceneRandomization((float(x), float(y)), diffuse, occluders)

    def corrupt_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply sensor noise then (probabilistically) motion blur to a rendered uint8 crop."""
        image = gaussian_noise(image, float(rng.uniform(*self.noise_sigma)), rng)
        if rng.random() < self.blur_prob:
            length = int(rng.integers(self.blur_length[0], self.blur_length[1] + 1))
            image = motion_blur(image, length, float(rng.uniform(0.0, math.pi)))
        return image


def apply_scene(world, scene: SceneRandomization) -> None:
    """Push a sampled :class:`SceneRandomization` onto ``world`` (lighting + every machine's occluder)."""
    world.randomize_lighting(scene.light_pos_xy, scene.light_diffuse)
    for i, occ in enumerate(scene.occluders):
        if occ is None:
            world.set_occluder(i, present=False)
        else:
            world.set_occluder(i, present=True, dx=occ.dx, dz=occ.dz, size=occ.size)
