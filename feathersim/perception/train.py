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

ARTIFACT_DIR = Path(__file__).resolve().parent
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_PATH = ARTIFACT_DIR / "model.pt"


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


def main() -> None:
    full = generate_dataset(n_samples=720, seed=0)
    train_ds, val_ds = train_val_split(full, seed=0)
    model, metrics = train(train_ds, val_ds, seed=0)

    torch.save(model.state_dict(), MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    beat = metrics["state_accuracy"] - metrics["state_majority_baseline"]
    print(json.dumps(metrics, indent=2))
    print(f"state accuracy beats majority baseline by {beat:+.1%}; saved {MODEL_PATH.name}, {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
