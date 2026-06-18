# FeatherSim — LEARNINGS.md

Append a dated entry whenever something surprises you (sim quirks, training failures, flaky
tests). This is the compounding mechanism: never get bitten by the same issue twice.

## 2026-06-18 — Dev environment baseline
Python 3.13.5 (anaconda), pip 25.1, pytest 8.3.4, git 2.50.1 on macOS (darwin 24.6).

## 2026-06-18 — PyBullet won't build here → switched sim to MuJoCo [RESOLVED]
The Phase-0 worry was real. `pip install pybullet` fails on this box (macOS, Apple clang 17,
framework Python 3.13): PyBullet ships no macOS wheels, and the source build dies compiling its
vendored zlib — the old zlib headers collide with the new macOS SDK `_stdio.h`, throwing
`error: expected identifier or '('` / `expected ')'` expanded from the `NULL` macro (with `ZEXPORT`
fragments in the trace). Not a Python-version issue — there's no wheel on Mac at any version, so a
venv wouldn't help; only conda-forge (env fragmentation) or a C patch (non-reproducible) would.
**Resolution:** switched the backend to MuJoCo (`pip install mujoco`, official prebuilt wheel,
instant). Verified: imports, steps headless deterministically, and `mujoco.Renderer` returns
offscreen RGB frames. See DECISIONS.md.

## 2026-06-18 — MuJoCo headless render needs a GL context, but it worked out of the box here
`mujoco.Renderer(model, h, w).render()` returned a real `(h, w, 3)` uint8 frame with no extra setup
on this macOS box. On headless Linux/CI you may need `MUJOCO_GL=egl` (or `osmesa`) in the env, since
the default GLFW backend wants a display. Note this if camera rendering (Phase 4) ever fails in CI.

## 2026-06-18 — pytest import path
Tests import the top-level `feathersim` package. Set `pythonpath = ["."]` in
`[tool.pytest.ini_options]` so `pytest` from repo root puts the root on `sys.path` (avoids
`ModuleNotFoundError` without needing an editable install or a root `conftest.py`).

## 2026-06-18 — Perception gotchas (Phase 4): four things that bit me
Building the auto-labeling + CNN pipeline surfaced four non-obvious failures, each fixed:
1. **Global avg-pool washed out the small status light** → the state head collapsed to a constant
   (acc 0.31, *below* the 0.38 baseline) while the larger bed-part head learned fine. Fix: a global
   **max-pool** branch (concat with avg) preserves the localized light signal; also enlarged the light.
   Lesson: for a small, localized cue, average pooling is the wrong inductive bias — use max.
2. **No contrast = no signal.** The first bed part was whitish (0.80) against a whitish floor (0.82);
   part acc stalled ~0.75. A vivid blue part (high contrast vs floor, dark door, and machine bodies)
   jumped it to 1.0. Design the visual cue for contrast against *everything* it can sit in front of.
3. **Train/serve mismatch.** Training rendered neighbors neutral but live `perceive`/`sync_visuals`
   lights *all* machines; edge machines whose camera grazes a lit neighbor then read at ~chance
   (conf 0.34). Fix: light every machine with an independent random config in the dataset and label
   the centered target — train distribution now matches serving.
4. **Visual-cue geoms are still collision geoms by default.** The new part geom protruded in front of
   the machine and physically blocked the robot's tending approach (SDK drive stalled ~0.16 m short).
   Fix: `contype="0" conaffinity="0"` on the light/part geoms — they're perception cues, not obstacles.

## 2026-06-18 — `python` ≠ `python3` here: deps live under `python3` only
On this machine `python` → conda base (`/opt/anaconda3`, no project deps) while `python3` → the
python.org framework build (`/Library/Frameworks/Python.framework/Versions/3.13`) where
`pip install` actually put mujoco/torch/numpy. Running `python scripts/print_state.py` died with
`ModuleNotFoundError: No module named 'mujoco'`; `python3 ...` works. The Makefile already uses
`python3` for every target — keep it that way and always invoke `python3`/`make`, never bare
`python`, for this project.

## 2026-06-18 — The demo caught a scheduler-fairness bug the unit tests didn't (Phase 5)
Running `make demo` on the real model delivered 6 parts but `machine_0: 0, machine_1: 3, machine_2: 3` —
one machine starved. Diagnosis: perception was *not* at fault (a quick probe with the robot parked at the
table read all three machines' `done` state with 100% agreement, 0 misreads). The cause was the loop's
tie-break: sorting perceived-done candidates by `(confidence, name)` descending, with confidence ~constant
at 0.98, made the order purely name-descending, so `machine_0` lost every tie and never got serviced within
the 6-part target. Fix: schedule **oldest-waiting-first** (track when each machine was first seen done) →
even `2/2/2`. Lessons: (1) a green unit suite doesn't prove the *emergent* behavior is right — always run
the actual demo and eyeball the distribution, not just the total; (2) when one component looks guilty
(perception), cheaply falsify that hypothesis before "fixing" it — the bug was in the scheduler.

## 2026-06-18 — Subagents load at session start
Files added to `.claude/agents/` are NOT available mid-session — they're read when the session
starts. After creating/editing them, restart Claude Code (or add via `/agents`) before trying to
delegate, or the Agent call won't find the custom agent. This is why Phase 0 ends with a
mandatory restart before Phase 1.
