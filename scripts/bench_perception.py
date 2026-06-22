"""Benchmark perception robustness: deployed (robust) vs clean-baseline model accuracy as the
domain-randomization difficulty slider rises — the headline DR story, as numbers.

For each difficulty level, render a fresh domain-randomized eval set and score both models on machine-state
classification (vs the majority-class baseline). The deployed model should hold where the clean one
crumbles. Needs a GL context (it renders crops); run::

    python scripts/bench_perception.py
    python scripts/bench_perception.py --samples 300 --json docs/perception_bench.json

This mirrors ``scripts/bench_fleet.py`` for the perception layer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from feathersim.perception.dataset import generate_dataset  # noqa: E402
from feathersim.perception.randomize import DomainRandomizer  # noqa: E402
from feathersim.perception.train import (  # noqa: E402
    evaluate,
    load_or_train_clean_model,
    load_or_train_model,
)

_DIFFICULTIES = (0.0, 0.25, 0.5, 0.75, 1.0)


def bench(samples: int, seed: int) -> list[dict]:
    """Score the robust and clean models on a randomized eval set at each difficulty level."""
    robust, clean = load_or_train_model(), load_or_train_clean_model()
    rows = []
    for d in _DIFFICULTIES:
        randomizer = None if d == 0.0 else DomainRandomizer.at_difficulty(d)
        ds = generate_dataset(n_samples=samples, seed=seed, randomizer=randomizer)
        rb = evaluate(robust, ds)
        cl = evaluate(clean, ds)
        rows.append({
            "difficulty": d,
            "robust_state_acc": round(rb["state_accuracy"], 3),
            "clean_state_acc": round(cl["state_accuracy"], 3),
            "gap": round(rb["state_accuracy"] - cl["state_accuracy"], 3),
            "baseline": round(rb["state_majority_baseline"], 3),
            "n": rb["n_val"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark perception robustness across DR difficulty.")
    ap.add_argument("--samples", type=int, default=200, help="eval renders per difficulty level")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    rows = bench(args.samples, args.seed)

    hdr = f"{'difficulty':>10} {'robust':>7} {'clean':>7} {'gap':>7} {'baseline':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['difficulty']:>10.2f} {r['robust_state_acc']:>7.3f} {r['clean_state_acc']:>7.3f} "
              f"{r['gap']:>+7.3f} {r['baseline']:>9.3f}")

    full = next(r for r in rows if r["difficulty"] == 1.0)
    print(f"\nat full difficulty: robust {full['robust_state_acc']:.0%} vs clean "
          f"{full['clean_state_acc']:.0%} (gap {full['gap']:+.0%}); baseline {full['baseline']:.0%}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
