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
    """A fully-specified randomized scene for one sample: a relative key-light jitter + a per-machine
    occluder slot (``None`` = none)."""

    light_offset_xy: tuple[float, float]
    light_diffuse_scale: float
    occluders: tuple[Occluder | None, ...]


@dataclass(frozen=True)
class DomainRandomizer:
    """Config + samplers for domain randomization. Defaults are tuned to be hard but solvable.

    DR corrupts three things: the **key light** (shifted/dimmed *relative* to its authored cinematic pose,
    so the clean render baseline is preserved), the **status-light occluder** (scene stage), and **sensor
    noise/blur** (pixel stage). Lighting variation is what the robust model learns to handle and the clean
    one can't — the dominant source of the robust-vs-clean accuracy gap.
    """

    # key light (relative to authored)
    light_xy_jitter: float = 1.2                       # ± metres shift on the light's x and y
    light_scale: tuple[float, float] = (0.6, 1.2)      # diffuse multiplier (1.0 = authored brightness)
    # occluder — must only *partially* cover the light (a full cover is unlabelable and hurts even the
    # robust model). NB the box sits ~0.12 m in front of the light, so its projected shadow is bigger than
    # its half-extent; these bounds are tuned empirically (worst case still leaves the light partly visible
    # — see tests/test_randomize.py / LEARNINGS.md), not derived from the half-extent alone.
    occluder_prob: float = 0.55
    occluder_size: tuple[float, float] = (0.03, 0.09)
    occluder_offset: float = 0.10                     # max |dx|, |dz| lateral/vertical offset
    # sensor
    noise_sigma: tuple[float, float] = (4.0, 27.0)
    blur_prob: float = 0.55
    blur_length: tuple[int, int] = (3, 9)             # kernel length in px (inclusive)

    @classmethod
    def at_difficulty(cls, difficulty: float) -> "DomainRandomizer":
        """A randomizer scaled by ``difficulty`` ∈ [0, 1] — 0 is clean (no corruption), 1 is full DR.

        Lets the dashboard dial perception difficulty live: occluder presence, noise σ, and blur probability
        all scale with ``difficulty`` (occluder/blur *extents* keep their defaults — presence is the
        dominant lever, so a low-difficulty scene rarely shows one at all).
        """
        d = max(0.0, min(1.0, difficulty))
        return cls(
            light_xy_jitter=1.2 * d, light_scale=(1.0 - 0.4 * d, 1.0 + 0.2 * d),
            occluder_prob=0.5 * d, noise_sigma=(0.0, 27.0 * d), blur_prob=0.55 * d,
        )

    def sample_scene(self, rng: np.random.Generator, n_machines: int) -> SceneRandomization:
        """Sample a randomized scene — a relative key-light jitter + one occluder slot per machine."""
        offset = tuple(rng.uniform(-self.light_xy_jitter, self.light_xy_jitter, size=2))
        diffuse_scale = float(rng.uniform(*self.light_scale))
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
        return SceneRandomization((float(offset[0]), float(offset[1])), diffuse_scale, occluders)

    def corrupt_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply sensor noise then (probabilistically) motion blur to a rendered uint8 crop."""
        image = gaussian_noise(image, float(rng.uniform(*self.noise_sigma)), rng)
        if rng.random() < self.blur_prob:
            length = int(rng.integers(self.blur_length[0], self.blur_length[1] + 1))
            image = motion_blur(image, length, float(rng.uniform(0.0, math.pi)))
        return image


def apply_scene(world, scene: SceneRandomization) -> None:
    """Push a sampled :class:`SceneRandomization` onto ``world`` — relative key-light jitter + occluders."""
    world.randomize_lighting(scene.light_offset_xy, scene.light_diffuse_scale)
    for i, occ in enumerate(scene.occluders):
        if occ is None:
            world.set_occluder(i, present=False)
        else:
            world.set_occluder(i, present=True, dx=occ.dx, dz=occ.dz, size=occ.size)
