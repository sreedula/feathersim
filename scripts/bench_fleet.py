"""Benchmark the multi-robot fleet: throughput, collision-freedom, and completion across configurations
and scheduling strategies, with ground-truth perception (fast + deterministic).

This is the quantitative record of the ORCA coordination win — every config (including the tight pillar
cell) completes collision-free, with no slow near-wedge seeds. Run::

    python scripts/bench_fleet.py                 # default sweep, prints a table
    python scripts/bench_fleet.py --seeds 20      # more seeds
    python scripts/bench_fleet.py --json docs/fleet_bench.json

Each row aggregates ``--seeds`` runs of one (machines × robots × obstacles, strategy) config.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from feathersim.fleet import run_fleet  # noqa: E402
from feathersim.fleet.scheduling import longest_waiting, nearest_done  # noqa: E402
from feathersim.perception.infer import PerceivedState  # noqa: E402
from feathersim.sim.machine import MachineState  # noqa: E402
from feathersim.sim.world import ROBOT_RADIUS, World  # noqa: E402

_STRATEGIES = {"longest_waiting": longest_waiting, "nearest_done": nearest_done}
# (machines, robots, obstacles, parts) — the headline 4×4, the smaller fleets, and the tight pillar cell.
_CONFIGS = [(4, 4, 0, 8), (3, 3, 0, 6), (3, 2, 0, 6), (2, 2, 0, 6), (3, 2, 2, 6)]
_CONTACT = 2.0 * ROBOT_RADIUS  # bodies overlap below this centre distance


def _ground_truth(world: World):
    """A perfect perceiver — isolates *coordination* from perception noise for the benchmark."""
    return lambda k: {m.name: PerceivedState(m.state, True, 0.99) for m in world.machines}


def _bench_one(nm: int, nr: int, obs: int, parts: int, strat_name: str, seeds: int) -> dict:
    strategy = _STRATEGIES[strat_name]
    secs, seps, done = [], [], 0
    for seed in range(seeds):
        world = World(n_machines=nm, seed=seed, n_obstacles=obs, n_robots=nr)
        r = run_fleet(world, _ground_truth(world), strategy=strategy, strategy_name=strat_name,
                      target_parts=parts, max_sim_seconds=200.0)
        secs.append(r.sim_seconds)
        seps.append(r.min_robot_separation)
        done += int(r.completed)
    worst_sep = min(seps)
    return {
        "config": f"{nm}m×{nr}r" + (f"+{obs}obs" if obs else ""),
        "strategy": strat_name, "seeds": seeds, "completed": done,
        "completion_rate": done / seeds,
        "worst_sep": round(worst_sep, 3), "collision_free": worst_sep >= _CONTACT,
        "sim_seconds_mean": round(statistics.mean(secs), 1), "sim_seconds_max": round(max(secs), 1),
        "throughput_per_min_mean": round(statistics.mean(parts * 60.0 / s for s in secs), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the FeatherSim fleet coordination.")
    ap.add_argument("--seeds", type=int, default=12, help="runs aggregated per config")
    ap.add_argument("--strategies", nargs="+", default=list(_STRATEGIES), choices=list(_STRATEGIES))
    ap.add_argument("--json", type=pathlib.Path, default=None, help="also write the rows as JSON here")
    args = ap.parse_args()

    rows = [
        _bench_one(nm, nr, obs, parts, strat, args.seeds)
        for strat in args.strategies for (nm, nr, obs, parts) in _CONFIGS
    ]

    hdr = f"{'config':<11} {'strategy':<16} {'done':>6} {'cfree':>6} {'sep':>6} {'mean_s':>7} {'max_s':>6} {'p/min':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['config']:<11} {r['strategy']:<16} {r['completed']:>3}/{r['seeds']:<2} "
              f"{('yes' if r['collision_free'] else 'NO'):>6} {r['worst_sep']:>6.3f} "
              f"{r['sim_seconds_mean']:>7.1f} {r['sim_seconds_max']:>6.1f} {r['throughput_per_min_mean']:>6.1f}")

    all_cfree = all(r["collision_free"] for r in rows)
    all_done = all(r["completion_rate"] == 1.0 for r in rows)
    print(f"\nall collision-free: {all_cfree}   all configs complete every seed: {all_done}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
