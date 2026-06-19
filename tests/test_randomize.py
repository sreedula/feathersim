"""v2 Phase A: domain randomization for robust perception.

The image-space augmentations and scene sampling are pure and tested without GL. The scene-applies-to-
render and the headline "robust beats clean under randomization" checks need a GL context and skip on a
headless host (set ``MUJOCO_GL=egl``/``osmesa``).
"""

import mujoco
import numpy as np
import pytest

from feathersim.perception.randomize import (
    DomainRandomizer,
    Occluder,
    apply_scene,
    gaussian_noise,
    motion_blur,
)
from feathersim.sim.world import World


def _spot_image() -> np.ndarray:
    """A black 64×64 crop with a bright off-center square (so blur/noise have something to act on)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[26:34, 30:40] = 240
    return img


# --- pure image-space augmentations --------------------------------------------------------------


def test_gaussian_noise_zero_sigma_is_noop():
    img = _spot_image()
    assert np.array_equal(gaussian_noise(img, 0.0, np.random.default_rng(0)), img)


def test_gaussian_noise_changes_pixels_and_preserves_shape_dtype():
    img = _spot_image()
    out = gaussian_noise(img, 20.0, np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert (out != img).any()


def test_gaussian_noise_deterministic_given_seed():
    img = _spot_image()
    a = gaussian_noise(img, 15.0, np.random.default_rng(7))
    b = gaussian_noise(img, 15.0, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_motion_blur_length_one_is_noop():
    img = _spot_image()
    assert np.array_equal(motion_blur(img, 1, 0.5), img)


def test_motion_blur_changes_and_preserves_shape_dtype():
    img = _spot_image()
    out = motion_blur(img, 7, 0.3)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert (out != img).any()


def test_corrupt_image_deterministic_and_preserves_shape():
    img = _spot_image()
    dr = DomainRandomizer()
    a = dr.corrupt_image(img, np.random.default_rng(3))
    b = dr.corrupt_image(img, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert a.shape == img.shape and a.dtype == np.uint8


# --- scene sampling ------------------------------------------------------------------------------


def test_sample_scene_reproducible():
    dr = DomainRandomizer()
    assert dr.sample_scene(np.random.default_rng(1), 3) == dr.sample_scene(np.random.default_rng(1), 3)


def test_sample_scene_within_bounds():
    dr = DomainRandomizer()
    scene = dr.sample_scene(np.random.default_rng(0), 3)
    assert len(scene.occluders) == 3
    assert all(abs(v) <= dr.light_xy_jitter for v in scene.light_offset_xy)
    assert dr.light_scale[0] <= scene.light_diffuse_scale <= dr.light_scale[1]
    for occ in scene.occluders:
        if occ is not None:
            assert isinstance(occ, Occluder)
            assert dr.occluder_size[0] <= occ.size <= dr.occluder_size[1]
            assert abs(occ.dx) <= dr.occluder_offset and abs(occ.dz) <= dr.occluder_offset


# --- GL-dependent: scene affects renders, and robustness holds -----------------------------------


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


rendering = pytest.mark.skipif(
    not _rendering_available(),
    reason="MuJoCo rendering unavailable (set MUJOCO_GL=egl/osmesa on a headless host)",
)


@rendering
def test_occluder_changes_the_render():
    world = World(n_machines=3, seed=0)
    r = mujoco.Renderer(world.model, 64, 64)
    try:
        world.sync_visuals()
        world.reset_scene()
        clean = world.render_machine(r, 1).copy()
        # Force an occluder on the target machine so the scene-stage corruption definitely fires.
        from feathersim.perception.randomize import Occluder, SceneRandomization
        apply_scene(world, SceneRandomization((0.0, 0.0), 1.0, (None, Occluder(0.0, 0.0, 0.09), None)))
        randomized = world.render_machine(r, 1)
    finally:
        r.close()
    assert (clean != randomized).any()


def test_reset_scene_restores_authored_lighting():
    """DR jitters the key light *relative* to its authored cinematic pose; reset_scene must restore that
    exact pose (so the dashboard's 3D feed and the perception clean-baseline are both correctly lit)."""
    world = World(n_machines=3, seed=0)
    authored = world.model.light_pos[0].copy(), world.model.light_diffuse[0].copy()
    apply_scene(world, DomainRandomizer().sample_scene(np.random.default_rng(0), 3))
    assert not (world.model.light_pos[0] == authored[0]).all()   # DR actually moved the light...
    world.reset_scene()
    assert (world.model.light_pos[0] == authored[0]).all()        # ...and reset restored it exactly
    assert (world.model.light_diffuse[0] == authored[1]).all()


@rendering
def test_randomized_dataset_differs_from_clean():
    from feathersim.perception.dataset import generate_dataset

    clean = generate_dataset(n_samples=16, seed=0)
    randomized = generate_dataset(n_samples=16, seed=0, randomizer=DomainRandomizer())
    assert clean.images.shape == randomized.images.shape
    assert sorted(set(randomized.state_labels.tolist())) and (clean.images != randomized.images).any()


@rendering
def test_domain_randomization_degrades_a_clean_model():
    """The mechanism behind the phase: a clean-trained model that aces clean renders measurably drops
    under randomization. (The *robust* model recovering this is scale-dependent — see the committed
    metrics.json assertion below and LEARNINGS.md — so it isn't retrained at unit scale here.)"""
    from feathersim.perception.dataset import generate_dataset, train_val_split
    from feathersim.perception.train import evaluate, train

    dr = DomainRandomizer()
    # 500 samples so the clean model trains tightly on the richer cinematic renders (it undertrains at 300).
    clean_tr, clean_val = train_val_split(generate_dataset(n_samples=500, seed=0), seed=0)
    _, rand_val = train_val_split(generate_dataset(n_samples=400, seed=0, randomizer=dr), seed=0)

    clean_model, _ = train(clean_tr, clean_val, seed=0)
    clean_on_clean = evaluate(clean_model, clean_val)["state_accuracy"]
    clean_on_rand = evaluate(clean_model, rand_val)["state_accuracy"]

    assert clean_on_clean > 0.9                     # aces the clean conditions it trained on
    assert clean_on_rand < clean_on_clean - 0.1     # but degrades under randomization


def test_committed_metrics_show_robust_beats_clean():
    """Lock in the full-scale Phase-A headline from the committed artifact (no retraining): under
    randomization the robust model beats the clean model and clears the baseline; the robust model trades
    at most a hair of clean accuracy for that robustness. Regenerated by `make train`."""
    import json

    from feathersim.perception.train import METRICS_PATH

    m = json.loads(METRICS_PATH.read_text())
    sa = m["state_accuracy"]
    assert sa["robust_model"]["randomized"] > sa["clean_model"]["randomized"]      # robust wins under DR
    assert sa["clean_model"]["randomized"] < sa["clean_model"]["clean"]            # clean degrades under DR
    # The robust model trades at most a hair of clean accuracy for its big robustness gain (the DR
    # tradeoff) — it stays close to the clean model on clean conditions, not necessarily ≥.
    assert sa["robust_model"]["clean"] >= sa["clean_model"]["clean"] - 0.05
    assert sa["robust_model"]["randomized"] > m["state_majority_baseline"] + 0.2   # clears baseline
