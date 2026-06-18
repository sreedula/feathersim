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

## 2026-06-18 — `python` ≠ `python3` here: deps live under `python3` only
On this machine `python` → conda base (`/opt/anaconda3`, no project deps) while `python3` → the
python.org framework build (`/Library/Frameworks/Python.framework/Versions/3.13`) where
`pip install` actually put mujoco/torch/numpy. Running `python scripts/print_state.py` died with
`ModuleNotFoundError: No module named 'mujoco'`; `python3 ...` works. The Makefile already uses
`python3` for every target — keep it that way and always invoke `python3`/`make`, never bare
`python`, for this project.

## 2026-06-18 — Subagents load at session start
Files added to `.claude/agents/` are NOT available mid-session — they're read when the session
starts. After creating/editing them, restart Claude Code (or add via `/agents`) before trying to
delegate, or the Agent call won't find the custom agent. This is why Phase 0 ends with a
mandatory restart before Phase 1.
