"""Train the perception CNN on auto-labeled sim data and log metrics. [Phase 4]

``python3 -m feathersim.perception.train`` (or ``make train``) generates a dataset, trains, evaluates
on a held-out split against the majority-class baseline, writes ``metrics.json`` + ``model.pt`` next
to this module, and prints the metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from feathersim.perception.dataset import (
    Dataset,
    generate_dataset,
    majority_baseline,
    train_val_split,
)
from feathersim.perception.model import PerceptionCNN, images_to_tensor
from feathersim.perception.randomize import DomainRandomizer

ARTIFACT_DIR = Path(__file__).resolve().parent
# metrics.json schema (written by main, asserted by tests/test_randomize.py — keep in sync):
#   {"state_accuracy": {"clean_model": {"clean", "randomized"},
#                       "robust_model": {"clean", "randomized"}},
#    "state_majority_baseline", "robust_part_accuracy_randomized", "n_val"}
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_PATH = ARTIFACT_DIR / "model.pt"              # deployed model — the DR-robust one (v2 Phase A)
MODEL_CLEAN_PATH = ARTIFACT_DIR / "model_clean.pt"  # v1-style clean-trained model, kept for comparison


def evaluate(model: PerceptionCNN, ds: Dataset) -> dict[str, float]:
    """State accuracy, part accuracy, and the majority-class baseline on ``ds``."""
    model.eval()
    with torch.no_grad():
        state_logits, part_logits = model(images_to_tensor(ds.images))
        state_pred = state_logits.argmax(1).numpy()
        part_pred = (torch.sigmoid(part_logits) > 0.5).numpy()
    return {
        "state_accuracy": float((state_pred == ds.state_labels).mean()),
        "part_accuracy": float((part_pred == ds.part_labels.astype(bool)).mean()),
        "state_majority_baseline": majority_baseline(ds.state_labels),
        "n_val": int(len(ds)),
    }


def train(
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[PerceptionCNN, dict[str, float]]:
    """Train the CNN (CE on state + BCE on part-present) and return ``(model, val_metrics)``."""
    torch.manual_seed(seed)
    model = PerceptionCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

    x = images_to_tensor(train_ds.images)
    y_state = torch.from_numpy(train_ds.state_labels)
    y_part = torch.from_numpy(train_ds.part_labels)
    n = len(train_ds)

    model.train()
    for _ in range(epochs):
        for batch in torch.randperm(n).split(batch_size):
            opt.zero_grad()
            state_logits, part_logits = model(x[batch])
            loss = ce(state_logits, y_state[batch]) + bce(part_logits, y_part[batch])
            loss.backward()
            opt.step()

    return model, evaluate(model, val_ds)


def load_or_train_model() -> PerceptionCNN:
    """Return a ready-to-use perception model: load ``model.pt`` if present, else train a fresh one.

    Lets the demo run as a single command on a clean checkout (``model.pt`` is gitignored) without a
    separate ``make train`` step. The fallback trains a smaller **domain-randomized** dataset so the
    deployed model matches what ``make train`` produces (the robust model).
    """
    model = PerceptionCNN()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        return model.eval()
    train_ds, val_ds = train_val_split(
        generate_dataset(n_samples=480, seed=0, randomizer=DomainRandomizer()), seed=0
    )
    model, _ = train(train_ds, val_ds, seed=0)
    return model.eval()


def load_or_train_clean_model() -> PerceptionCNN:
    """Return the **clean-trained** comparison model: load ``model_clean.pt`` if present, else train one
    on clean (DR-off) data. Used by the dashboard to show the clean baseline degrading next to the robust
    model as the difficulty slider rises."""
    model = PerceptionCNN()
    if MODEL_CLEAN_PATH.exists():
        model.load_state_dict(torch.load(MODEL_CLEAN_PATH, map_location="cpu"))
        return model.eval()
    train_ds, val_ds = train_val_split(generate_dataset(n_samples=480, seed=0), seed=0)
    model, _ = train(train_ds, val_ds, seed=0)
    return model.eval()


def _state_acc(model: PerceptionCNN, ds: Dataset) -> float:
    return evaluate(model, ds)["state_accuracy"]


def main() -> None:
    # Train two equally-sized models: clean (v1 recipe) and robust (domain-randomized), then evaluate
    # both on a clean held-out set AND a randomized one — the 2×2 before/after matrix (v2 Phase A).
    clean_tr, clean_val = train_val_split(generate_dataset(n_samples=720, seed=0), seed=0)
    rand_tr, rand_val = train_val_split(
        generate_dataset(n_samples=720, seed=0, randomizer=DomainRandomizer()), seed=0
    )

    clean_model, _ = train(clean_tr, clean_val, seed=0)
    robust_model, _ = train(rand_tr, rand_val, seed=0)

    metrics = {
        "state_accuracy": {
            "clean_model": {"clean": _state_acc(clean_model, clean_val),
                            "randomized": _state_acc(clean_model, rand_val)},
            "robust_model": {"clean": _state_acc(robust_model, clean_val),
                             "randomized": _state_acc(robust_model, rand_val)},
        },
        "state_majority_baseline": majority_baseline(rand_val.state_labels),
        "robust_part_accuracy_randomized": evaluate(robust_model, rand_val)["part_accuracy"],
        "n_val": int(len(rand_val)),
    }

    torch.save(robust_model.state_dict(), MODEL_PATH)
    torch.save(clean_model.state_dict(), MODEL_CLEAN_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    sa = metrics["state_accuracy"]
    print(json.dumps(metrics, indent=2))
    print(
        f"under randomization: clean model {sa['clean_model']['randomized']:.1%} → "
        f"robust model {sa['robust_model']['randomized']:.1%} "
        f"(+{sa['robust_model']['randomized'] - sa['clean_model']['randomized']:.1%}); "
        f"baseline {metrics['state_majority_baseline']:.1%}. Saved {MODEL_PATH.name}, {MODEL_CLEAN_PATH.name}."
    )


if __name__ == "__main__":
    main()
