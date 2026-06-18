"""Phase 4: perception — auto-labeled dataset, 2-head CNN, training beats baseline, inference.

Rendering needs a GL context; on a headless box without one these tests skip (set
``MUJOCO_GL=egl`` or ``osmesa``). Training is seeded so the "beats baseline" assertion is stable.
"""

import mujoco
import numpy as np
import pytest
import torch

from feathersim.perception import (
    STATE_CLASSES,
    Perception,
    PerceptionCNN,
    generate_dataset,
    images_to_tensor,
    majority_baseline,
    train_val_split,
)
from feathersim.perception.train import train
from feathersim.sim.machine import MachineState
from feathersim.sim.world import World


def _rendering_available() -> bool:
    try:
        w = World(n_machines=1, seed=0)
        r = mujoco.Renderer(w.model, 32, 32)
        r.update_scene(w.data, w.machine_camera(0))
        r.render()
        r.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _rendering_available(),
    reason="MuJoCo rendering unavailable (set MUJOCO_GL=egl/osmesa on a headless host)",
)


@pytest.fixture(scope="module")
def trained():
    """Generate a small dataset and train once; reused across the module."""
    full = generate_dataset(n_samples=360, seed=0)
    train_ds, val_ds = train_val_split(full, seed=0)
    model, metrics = train(train_ds, val_ds, epochs=10, seed=0)
    return full, val_ds, model, metrics


# --- dataset -----------------------------------------------------------------------------

def test_dataset_shapes_and_labels(trained):
    full, _, _, _ = trained
    assert full.images.shape == (360, 64, 64, 3)
    assert full.images.dtype == np.uint8
    assert 0 <= full.images.min() and full.images.max() <= 255
    assert set(np.unique(full.state_labels)) == {0, 1, 2}        # all three states present
    assert set(np.unique(full.part_labels)) == {0.0, 1.0}        # both part labels present


def test_train_val_split_partitions():
    full = generate_dataset(n_samples=40, seed=1)
    tr, va = train_val_split(full, val_fraction=0.25, seed=1)
    assert len(tr) == 30 and len(va) == 10
    assert len(tr) + len(va) == len(full)


# --- model -------------------------------------------------------------------------------

def test_model_output_shapes():
    model = PerceptionCNN()
    x = torch.rand(5, 3, 64, 64)
    state_logits, part_logits = model(x)
    assert state_logits.shape == (5, 3)
    assert part_logits.shape == (5,)


def test_images_to_tensor_normalizes():
    imgs = (np.random.default_rng(0).integers(0, 256, (4, 64, 64, 3))).astype(np.uint8)
    t = images_to_tensor(imgs)
    assert t.shape == (4, 3, 64, 64)
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


# --- training beats baseline (the acceptance criterion) ----------------------------------

def test_state_accuracy_beats_majority_baseline(trained):
    _, val_ds, _, metrics = trained
    assert metrics["state_majority_baseline"] == pytest.approx(
        majority_baseline(val_ds.state_labels)
    )
    assert metrics["state_accuracy"] >= metrics["state_majority_baseline"] + 0.2
    assert metrics["state_accuracy"] >= 0.8
    assert metrics["part_accuracy"] >= 0.8


# --- inference ---------------------------------------------------------------------------

def test_perception_read_clear_frames(trained):
    _, _, model, _ = trained
    perception = Perception(model)
    world = World(n_machines=3, seed=3)
    renderer = mujoco.Renderer(world.model, 64, 64)
    try:
        correct = 0
        total = 0
        for state in STATE_CLASSES:
            for part in (False, True):
                for j in range(3):
                    world.set_machine_visual(j, MachineState.IDLE, False)
                world.set_machine_visual(1, state, part)
                reading = perception.read(world.render_machine(renderer, 1))
                assert 0.0 <= reading.confidence <= 1.0
                correct += reading.machine_state is state and reading.part_present == part
                total += 1
    finally:
        renderer.close()
    assert correct >= total - 1  # at most one miss across the 6 clear configs


def test_perceive_reads_live_machine_states(trained):
    _, _, model, _ = trained
    perception = Perception(model)
    world = World(n_machines=3, seed=4)
    world.machines[0].state = MachineState.IDLE
    world.machines[1].state = MachineState.RUNNING
    world.machines[2].state = MachineState.DONE
    renderer = mujoco.Renderer(world.model, 64, 64)
    try:
        readings = perception.perceive(world, renderer)
    finally:
        renderer.close()
    assert readings["machine_0"].machine_state is MachineState.IDLE
    assert readings["machine_1"].machine_state is MachineState.RUNNING
    assert readings["machine_2"].machine_state is MachineState.DONE
