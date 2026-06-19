"""The behavior-cloning policy network and its I/O scaling. [v2 Phase D]

A tiny MLP that maps the go-to-pose **observation** (the goal in the robot's body frame, ``(forward,
left, dyaw)``) to a body **twist** ``(vx, vy, omega)`` — exactly the signature of the hand-coded
``velocity_command`` it imitates. Observations and actions are scaled to ≈[-1, 1] for stable training;
the ``tanh`` output keeps actions within the controller's velocity limits.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from feathersim.control.go_to_pose import PoseGains

# Observation scale: forward/left errors span the cell (~±3 m); heading error is ±pi.
OBS_SCALE = (3.0, 3.0, math.pi)
# Action scale = the expert controller's velocity caps, taken straight from PoseGains so the coupling is
# explicit. The policy is therefore tied to these training-time caps (the tanh head outputs ±ACTION_SCALE);
# the dataset is generated with the same default gains, so the two can't silently drift apart.
_CAPS = PoseGains()
ACTION_SCALE = (_CAPS.max_linear, _CAPS.max_linear, _CAPS.max_angular)


class PolicyMLP(nn.Module):
    """3 → hidden → hidden → 3 MLP with a ``tanh`` head (normalized obs → normalized twist)."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Scale raw observations ``(forward, left, dyaw)`` to ≈[-1, 1]."""
    return obs / np.array(OBS_SCALE, dtype=np.float32)


def normalize_action(action: np.ndarray) -> np.ndarray:
    """Scale raw twists ``(vx, vy, omega)`` to ≈[-1, 1] (the network's target range)."""
    return action / np.array(ACTION_SCALE, dtype=np.float32)


def denormalize_action(action: np.ndarray) -> np.ndarray:
    """Invert :func:`normalize_action` — network output → real twist."""
    return action * np.array(ACTION_SCALE, dtype=np.float32)
