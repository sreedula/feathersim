# FeatherSim — CLAUDE.md

FeatherSim is a machine-tending autonomy stack **in simulation**, inspired by Feather Robotics:
a holonomic wheeled mobile robot that autonomously tends several CNC-style machines, with a
developer-facing skill SDK, a perception model trained on auto-labeled sim data, an unattended
autonomy loop, and a teleop/fleet dashboard. No real hardware.

**This file is the source of truth for conventions and how to run things. Re-read it at the
start of every session.**

## Architecture (one screen)

- `feathersim/sim/` — MuJoCo world: 1–4 robot bases (each with an arm + gripper + onboard camera), 1–4 machines (state machine: idle→running→done), output table, obstacles; cinematic materials for the live feed. Pure sim.
- `feathersim/kinematics/` — holonomic drive math (body velocity → wheel commands). **Pure functions, no sim import.**
- `feathersim/control/` — go-to-pose controller for the base.
- `feathersim/perception/` — render + auto-label pipeline (labels from sim ground truth), a small PyTorch CNN, training, and `perception.read(camera) -> state`.
- `feathersim/sdk/` — developer API hiding sim details: `robot.move_to / pick / place / tend`. First-class deliverable.
- `feathersim/autonomy/` — the headline loop: perceive which machine is `done` → navigate → unload → load → repeat, unattended.
- `feathersim/dashboard/` — FastAPI + single-file vanilla-JS UI: live camera feed, WASD/joystick teleop override, telemetry.
- `feathersim/demo.py` — `python -m feathersim.demo` entry point.

## Conventions

- Python 3.11+ (developed on 3.13). Keep deps in `requirements.txt`.
- Kinematics & perception *logic* live in pure, testable functions — testable without spinning up MuJoCo.
- Tests in `tests/`, run with `pytest`. **Nothing advances on red.**
- Small, frequent commits. Messages: `phaseN: <what>`.
- Keep the system runnable & demoable at the end of every phase (walking skeleton).

## Run / test / demo

```
make install     # pip install -r requirements.txt
make test        # pytest
make demo        # python -m feathersim.demo  (headless single-robot autonomy demo)
make fleet       # headless multi-robot fleet demo (scheduling comparison)
make bench       # benchmark fleet coordination across configs × strategies → docs/fleet_bench.json
make dashboard   # launch the multi-robot command center (FastAPI)
make teleop      # launch the single-robot WASD-teleop dashboard
make train       # train the perception model on auto-labeled sim data
make policy      # behavior-clone the controller into a learned policy
```

## Subagents (project, in `.claude/agents/`)

Use for context hygiene and fresh-perspective review. The main thread still owns architecture
and implementation — do NOT over-delegate. Include file paths, the relevant acceptance criteria,
and any error text in the delegation prompt (subagents start with a fresh context).

- **`test-runner`** (haiku) — run pytest after any change; returns only pass/fail + concise
  tracebacks. Keeps verbose output out of the main thread.
- **`reviewer`** (sonnet) — read-only senior review at the end of every phase, after green and
  before commit. Returns prioritized findings + SHIP/NEEDS WORK. Address CRITICAL/HIGH before
  proceeding; log deferred MEDIUM/LOW in `PLAN.md`.
- **`docs-researcher`** (sonnet) — exact API signatures + minimal snippet for PyBullet / MuJoCo /
  PyTorch / FastAPI. Use instead of guessing unfamiliar APIs.
- **`Explore`** (built-in) — read-only codebase search; use instead of dumping files into the thread.

> Agent files load at session start. After editing `.claude/agents/`, restart the session
> (or create them via `/agents`) before delegating.

## The engineering loop (mandatory, every phase)

1. Refine acceptance criteria in `PLAN.md`.
2. Implement the smallest vertical slice that satisfies them (use `docs-researcher` for unfamiliar APIs).
3. Delegate to **`test-runner`**. Nothing advances on red — fix until green.
4. Delegate to **`reviewer`**. Address every CRITICAL/HIGH finding; note deferred MEDIUM/LOW in `PLAN.md`.
5. Update `DECISIONS.md` / `LEARNINGS.md`.
6. Commit. Tick `PLAN.md` checkboxes.
7. Next phase.

See `PLAN.md` for the roadmap, `DECISIONS.md` for why, `LEARNINGS.md` for gotchas.
