"""Train the behavior-cloning policy and compare it to the hand-coded expert. [v2 Phase D]

``python3 -m feathersim.policy.train`` (or ``make policy``) generates expert (obs → action) data, trains
the MLP, then runs a **closed-loop** comparison — driving a base from random starts to random goals under
the expert P-controller vs. the learned policy — and writes ``metrics_policy.json``. The net is tiny, so
this runs in seconds on CPU (no GPU needed).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from feathersim.control.go_to_pose import drive_to_pose, velocity_command
from feathersim.policy.dataset import generate_dataset
from feathersim.policy.model import PolicyMLP, normalize_action, normalize_obs
from feathersim.policy.policy import PolicyController
from feathersim.sim.world import World

ARTIFACT_DIR = Path(__file__).resolve().parent
POLICY_PATH = ARTIFACT_DIR / "policy.pt"            # trained weights (gitignored; regenerate via make policy)
# metrics_policy.json schema: {"val_mse", "expert_reach", "policy_reach", "expert_avg_steps",
#                              "policy_avg_steps", "n_episodes"}
METRICS_PATH = ARTIFACT_DIR / "metrics_policy.json"


def train(
    obs: np.ndarray, actions: np.ndarray, *, epochs: int = 40, lr: float = 1e-3,
    batch_size: int = 256, seed: int = 0, val_fraction: float = 0.2,
) -> tuple[PolicyMLP, float]:
    """Behavior-clone the expert (MSE on normalized twists); return ``(model, val_mse)``."""
    torch.manual_seed(seed)
    x = torch.from_numpy(normalize_obs(obs))
    y = torch.from_numpy(normalize_action(actions))
    n = len(x)
    perm = torch.randperm(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = PolicyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for batch in train_idx[torch.randperm(len(train_idx))].split(batch_size):
            opt.zero_grad()
            loss = mse(model(x[batch]), y[batch])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_mse = float(mse(model(x[val_idx]), y[val_idx]))
    return model, val_mse


def closed_loop_compare(policy_fn, *, n_episodes: int = 40, seed: int = 123) -> dict[str, float]:
    """Drive random start→goal episodes under the expert and the policy; compare reach rate + steps."""
    rng = np.random.default_rng(seed)
    # Sample start/goal in the open band between the machine (y≈1.5) and table (y≈-1.5) so reach rate
    # reflects controller quality, not incidental kinematic clipping through a fixture geom.
    def pt() -> tuple[float, float, float]:
        return (rng.uniform(-1.5, 1.5), rng.uniform(-0.8, 0.8), rng.uniform(-math.pi, math.pi))

    episodes = [(*pt(), *pt()) for _ in range(n_episodes)]

    def run(velocity_fn) -> tuple[float, float]:
        reached, steps = 0, []
        for sx, sy, syaw, gx, gy, gyaw in episodes:
            world = World(n_machines=1, seed=0, n_robots=1)
            world.set_base_pose(sx, sy, syaw)
            result = drive_to_pose(world, (gx, gy, gyaw), velocity_fn=velocity_fn, max_steps=2000)
            reached += int(result.reached)
            steps.append(result.steps)
        return reached / n_episodes, float(np.mean(steps))

    expert_reach, expert_steps = run(velocity_command)
    policy_reach, policy_steps = run(policy_fn)
    return {
        "expert_reach": expert_reach, "policy_reach": policy_reach,
        "expert_avg_steps": expert_steps, "policy_avg_steps": policy_steps,
        "n_episodes": n_episodes,
    }


def load_or_train_policy() -> PolicyMLP:
    """Load ``policy.pt`` if present, else behavior-clone a fresh policy (fast, CPU)."""
    model = PolicyMLP()
    if POLICY_PATH.exists():
        model.load_state_dict(torch.load(POLICY_PATH, map_location="cpu"))
        return model.eval()
    obs, actions = generate_dataset(n_samples=12000, seed=0)
    model, _ = train(obs, actions)
    return model.eval()


def main() -> None:
    obs, actions = generate_dataset(n_samples=20000, seed=0)
    model, val_mse = train(obs, actions)
    metrics = {"val_mse": val_mse, **closed_loop_compare(PolicyController(model))}

    torch.save(model.state_dict(), POLICY_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print(json.dumps(metrics, indent=2))
    print(
        f"behavior cloning: val MSE {val_mse:.2e}; learned policy reaches "
        f"{metrics['policy_reach']:.0%} of goals (expert {metrics['expert_reach']:.0%}) in "
        f"{metrics['policy_avg_steps']:.0f} steps vs expert {metrics['expert_avg_steps']:.0f}."
    )


if __name__ == "__main__":
    main()
