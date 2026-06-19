"""The learned policy as a drop-in controller. [v2 Phase D]

:class:`PolicyController` has the same call signature as :func:`velocity_command` — ``(pose, target,
gains) -> twist`` — so it slots straight into ``drive_to_pose(..., velocity_fn=policy)`` and hence
``Robot(..., controller=policy)``, letting the whole autonomy loop run on the learned brain.
"""

from __future__ import annotations

import numpy as np
import torch

from feathersim.control.go_to_pose import PoseGains, Pose, goal_in_body_frame
from feathersim.policy.model import PolicyMLP, denormalize_action, normalize_obs


class PolicyController:
    """Wraps a trained :class:`PolicyMLP` and exposes it as a ``velocity_fn``."""

    def __init__(self, model: PolicyMLP) -> None:
        self.model = model.eval()

    def __call__(self, pose: Pose, target: Pose, gains: PoseGains = PoseGains()) -> Pose:
        # ``gains`` is accepted only to match the velocity_fn signature; the policy's output is fixed to
        # the caps it was trained against (tanh head × ACTION_SCALE = the default PoseGains caps), so it
        # is *not* re-scaled to a caller's gains — the policy is tied to its training-time velocity limits.
        obs = np.array(goal_in_body_frame(pose, target), dtype=np.float32)
        with torch.no_grad():
            out = self.model(torch.from_numpy(normalize_obs(obs))[None]).numpy()[0]
        vx, vy, omega = denormalize_action(out)
        return (float(vx), float(vy), float(omega))
