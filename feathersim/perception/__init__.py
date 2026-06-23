"""Render + auto-label pipeline, a small CNN, training, and `perception.read`. [Phase 4]"""

from feathersim.perception.dataset import (
    STATE_CLASSES,
    Dataset,
    generate_dataset,
    majority_baseline,
    train_val_split,
)
from feathersim.perception.infer import PerceivedState, Perception
from feathersim.perception.model import PerceptionCNN, images_to_tensor
from feathersim.perception.randomize import DomainRandomizer, gaussian_noise, motion_blur

# NB: training entry points live in feathersim.perception.train and are intentionally NOT imported
# here, so `python -m feathersim.perception.train` doesn't trip the "module already imported" warning.

__all__ = [
    "STATE_CLASSES",
    "Dataset",
    "generate_dataset",
    "train_val_split",
    "majority_baseline",
    "PerceptionCNN",
    "images_to_tensor",
    "Perception",
    "PerceivedState",
    "DomainRandomizer",
    "gaussian_noise",
    "motion_blur",
]
