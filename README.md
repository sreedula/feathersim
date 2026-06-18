# FeatherSim

A machine-tending autonomy stack **in simulation**, inspired by Feather Robotics: a holonomic
wheeled mobile robot autonomously tends several CNC-style machines, exposes a developer skill SDK,
perceives machine state from an onboard camera (model trained on auto-labeled sim data), runs an
unattended autonomy loop, and is teleoperable from a browser dashboard.

> Status: **Phase 0 — scaffold**. See [`PLAN.md`](PLAN.md) for the roadmap.

## Quickstart

```
make install   # install dependencies
make test      # run the test suite
make demo      # run the demo entry point
```

## Docs

- [`PLAN.md`](PLAN.md) — phased roadmap + acceptance criteria
- [`DECISIONS.md`](DECISIONS.md) — architecture decision log
- [`LEARNINGS.md`](LEARNINGS.md) — gotchas & sim/training quirks
- [`CLAUDE.md`](CLAUDE.md) — conventions + architecture summary
