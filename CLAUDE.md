# FeatherSim — CLAUDE.md

FeatherSim is a machine-tending autonomy stack **in simulation**, inspired by Feather Robotics:
a holonomic wheeled mobile robot that autonomously tends several CNC-style machines, with a
developer-facing skill SDK, a perception model trained on auto-labeled sim data, an unattended
autonomy loop, and a teleop/fleet dashboard. No real hardware.

**This file is the source of truth for conventions and how to run things. Re-read it at the
start of every session.**

## Architecture (one screen)

- `feathersim/sim/` — PyBullet world: robot base, 2–3 machines (state machine: idle→running→done), parts table. Pure sim.
- `feathersim/kinematics/` — holonomic drive math (body velocity → wheel commands). **Pure functions, no sim import.**
- `feathersim/control/` — go-to-pose controller for the base.
- `feathersim/perception/` — render + auto-label pipeline (labels from sim ground truth), a small PyTorch CNN, training, and `perception.read(camera) -> state`.
- `feathersim/sdk/` — developer API hiding sim details: `robot.move_to / pick / place / tend`. First-class deliverable.
- `feathersim/autonomy/` — the headline loop: perceive which machine is `done` → navigate → unload → load → repeat, unattended.
- `feathersim/dashboard/` — FastAPI + single-file vanilla-JS UI: live camera feed, WASD/joystick teleop override, telemetry.
- `feathersim/demo.py` — `python -m feathersim.demo` entry point.

## Conventions

- Python 3.11+ (developed on 3.13). Keep deps in `requirements.txt`.
- Kinematics & perception *logic* live in pure, testable functions — testable without spinning up PyBullet.
- Tests in `tests/`, run with `pytest`. **Nothing advances on red.**
- Small, frequent commits. Messages: `phaseN: <what>`.
- Keep the system runnable & demoable at the end of every phase (walking skeleton).

## Run / test / demo

```
make install     # pip install -r requirements.txt
make test        # pytest
make demo        # python -m feathersim.demo  (headless autonomy demo)
make dashboard   # launch FastAPI teleop/telemetry UI
make train       # train the perception model on auto-labeled sim data
```

## The engineering loop (mandatory, every phase)

1. Refine acceptance criteria in `PLAN.md`.
2. Implement the smallest vertical slice that satisfies them.
3. Write tests, run them. Red blocks progress.
4. Self-review against acceptance criteria **and** `LEARNINGS.md`.
5. Update `DECISIONS.md` / `LEARNINGS.md`.
6. Commit. Tick `PLAN.md` checkboxes.
7. Next phase.

See `PLAN.md` for the roadmap, `DECISIONS.md` for why, `LEARNINGS.md` for gotchas.
