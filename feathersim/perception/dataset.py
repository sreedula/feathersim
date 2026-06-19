"""Auto-labeled perception dataset: render machine close-ups, label from ground-truth configs. [Phase 4]

This is the headline auto-labeling pipeline. The sim is placed in randomized but *known*
configurations — each sample picks a machine, a state, and whether a part is on the bed, sampled
**independently** for class balance and so the two heads aren't trivially correlated — then the
machine's camera is rendered and the labels are read straight from the chosen config. No human
labeling. Camera azimuth/elevation/distance are jittered slightly so the task isn't a one-pixel lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from feathersim.perception.randomize import DomainRandomizer, apply_scene
from feathersim.sim.machine import MachineState
from feathersim.sim.world import World

STATE_CLASSES = (MachineState.IDLE, MachineState.RUNNING, MachineState.DONE)
STATE_TO_INDEX = {s: i for i, s in enumerate(STATE_CLASSES)}
IMAGE_SIZE = 64


@dataclass
class Dataset:
    """Rendered frames with their ground-truth labels."""

    images: np.ndarray        # (N, H, W, 3) uint8
    state_labels: np.ndarray  # (N,) int64, index into STATE_CLASSES
    part_labels: np.ndarray   # (N,) float32 in {0.0, 1.0}

    def __len__(self) -> int:
        return len(self.images)

    def subset(self, idx: np.ndarray) -> "Dataset":
        return Dataset(self.images[idx], self.state_labels[idx], self.part_labels[idx])


def generate_dataset(
    n_samples: int = 480,
    *,
    seed: int = 0,
    image_size: int = IMAGE_SIZE,
    n_machines: int = 3,
    jitter: bool = True,
    randomizer: DomainRandomizer | None = None,
) -> Dataset:
    """Render ``n_samples`` auto-labeled machine close-ups from randomized ground-truth configs.

    With ``randomizer`` set, domain randomization is applied — randomized scene lighting + status-light
    occluders (3D, before render) and Gaussian noise + motion blur (on the crop, after render) — so the
    model trains under hard conditions. ``randomizer=None`` reproduces the clean v1 pipeline.
    """
    rng = np.random.default_rng(seed)
    world = World(n_machines=n_machines, seed=seed)
    renderer = mujoco.Renderer(world.model, height=image_size, width=image_size)
    try:
        images = np.empty((n_samples, image_size, image_size, 3), dtype=np.uint8)
        states = np.empty(n_samples, dtype=np.int64)
        parts = np.empty(n_samples, dtype=np.float32)
        for k in range(n_samples):
            i = int(rng.integers(n_machines))
            # Light EVERY machine with an independent random config — matching what a live camera
            # sees (sync_visuals lights all machines) — and label the centered target machine i.
            configs = [
                (STATE_CLASSES[int(rng.integers(len(STATE_CLASSES)))], bool(rng.integers(2)))
                for _ in range(n_machines)
            ]
            for j, (sj, pj) in enumerate(configs):
                world.set_machine_visual(j, sj, pj)
            if randomizer is not None:
                apply_scene(world, randomizer.sample_scene(rng, n_machines))
            else:
                world.reset_scene()
            state, part = configs[i]
            cam = world.machine_camera(i)
            if jitter:
                cam.azimuth += float(rng.uniform(-8.0, 8.0))
                cam.elevation += float(rng.uniform(-4.0, 4.0))
                cam.distance *= float(rng.uniform(0.95, 1.08))
            image = world.render_machine(renderer, i, cam)
            if randomizer is not None:
                image = randomizer.corrupt_image(image, rng)
            images[k] = image
            states[k] = STATE_TO_INDEX[state]
            parts[k] = 1.0 if part else 0.0
    finally:
        renderer.close()
    return Dataset(images, states, parts)


def train_val_split(ds: Dataset, *, val_fraction: float = 0.25, seed: int = 0) -> tuple[Dataset, Dataset]:
    """Shuffle and split into (train, val)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    n_val = max(1, int(len(ds) * val_fraction))
    return ds.subset(idx[n_val:]), ds.subset(idx[:n_val])


def majority_baseline(state_labels: np.ndarray, n_states: int = len(STATE_CLASSES)) -> float:
    """Accuracy of always predicting the most common state — the bar perception must clear."""
    counts = np.bincount(state_labels, minlength=n_states)
    return float(counts.max() / counts.sum())
