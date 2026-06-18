"""Inference: read machine state from a camera frame. [Phase 4]

``perception.read(image) -> PerceivedState`` is the seam the autonomy loop (Phase 5) consumes —
it returns the model's *prediction* from pixels, never the sim's ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from feathersim.perception.dataset import STATE_CLASSES
from feathersim.perception.model import PerceptionCNN, images_to_tensor
from feathersim.sim.machine import MachineState


@dataclass
class PerceivedState:
    """A perception reading for one machine (prediction, not ground truth)."""

    machine_state: MachineState
    part_present: bool
    confidence: float  # softmax probability of the predicted state


class Perception:
    """Wraps a trained :class:`PerceptionCNN` and reads frames into :class:`PerceivedState`."""

    def __init__(self, model: PerceptionCNN) -> None:
        self.model = model.eval()

    def read(self, image: np.ndarray) -> PerceivedState:
        """Read a single ``(H, W, 3)`` uint8 frame."""
        with torch.no_grad():
            state_logits, part_logits = self.model(images_to_tensor(image[None]))
            probs = torch.softmax(state_logits, dim=1)[0]
            idx = int(probs.argmax())
            part = bool(torch.sigmoid(part_logits)[0] > 0.5)
        return PerceivedState(STATE_CLASSES[idx], part, float(probs[idx]))

    def perceive(self, world, renderer: mujoco.Renderer) -> dict[str, PerceivedState]:
        """Sync the scene to live state, then render + read every machine's camera.

        ``sync_visuals`` only paints what a real camera would see; the readings still come from the
        model's view of the pixels, so this is genuine perception (it can be wrong).
        """
        world.sync_visuals()
        return {
            f"machine_{i}": self.read(world.render_machine(renderer, i))
            for i in range(world.n_machines)
        }
