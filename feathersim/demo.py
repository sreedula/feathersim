"""FeatherSim demo entry point.

Run with: ``python -m feathersim.demo``

At later phases this launches the unattended machine-tending autonomy loop. For now (Phase 0
scaffold) it prints system status so the single-command launch path exists and stays green.
"""

from __future__ import annotations

import feathersim


def main() -> None:
    print(f"FeatherSim v{feathersim.__version__} — scaffold ready (Phase 0).")
    print("Autonomy loop arrives in Phase 5. See PLAN.md for the roadmap.")


if __name__ == "__main__":
    main()
