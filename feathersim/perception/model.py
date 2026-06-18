"""Small 2-head perception CNN. [Phase 4]

Reads a machine close-up and predicts its state (3-class: idle/running/done) and whether a part is
on the bed (binary). Deliberately tiny — the headline is the auto-labeling pipeline and closing the
loop, not SOTA vision, and a small net trains in seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_STATES = 3  # idle / running / done — indices match feathersim.perception.dataset.STATE_CLASSES


class PerceptionCNN(nn.Module):
    """3 strided conv blocks → concatenated global max + average pool → state and part heads.

    The **max** branch preserves the small, localized status-light signal (which average pooling
    washes out); the **avg** branch captures the broader bed-part region. Both heads see both.
    """

    def __init__(self, n_states: int = N_STATES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),   # 64 → 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 32 → 16
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # 16 → 8
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.state_head = nn.Linear(64, n_states)
        self.part_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x``: ``(N, 3, H, W)`` in [0, 1]. Returns ``(state_logits (N, n_states), part_logits (N,))``."""
        f = self.features(x)
        z = torch.cat([self.max_pool(f).flatten(1), self.avg_pool(f).flatten(1)], dim=1)
        return self.state_head(z), self.part_head(z).squeeze(1)


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    """``(N, H, W, 3)`` uint8 → ``(N, 3, H, W)`` float in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(images)).float().div_(255.0).permute(0, 3, 1, 2)
