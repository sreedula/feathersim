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

## 2026-06-18 — MuJoCo GL contexts are thread-affine: create the renderer on the thread that renders (Phase 6)
The dashboard runs the sim on a background thread. I first built the `mujoco.Renderer`s in
`SimManager.__init__` (the main/caller thread) but called `.render()` from the sim thread — and
`manager.frame()` stayed `None` forever (the daemon thread silently died). On macOS the renderer's GL
context is bound to the thread that *created* it; rendering from another thread fails. Fix: create the
renderers at the top of `_run()` (on the sim thread) and close them there in a `finally` — never in
`__init__`/`stop()`. Corollary: a wedged sim thread can't have its renderers closed from `stop()` either
(same affinity), so `stop()` just warns if the join times out. On headless Linux/CI the same code needs
`MUJOCO_GL=egl`. This is the Phase-4 "headless render needs a GL context" note, now with a threading twist.

## 2026-06-18 — An endless MJPEG generator hangs FastAPI's TestClient (Phase 6)
Consuming the `multipart/x-mixed-replace` camera stream through `TestClient.stream(...)` and `break`-ing
out hung the whole test run (had to SIGKILL; exit 144). The server-side generator is `while True: … sleep`,
and the test transport doesn't simulate a browser disconnect, so it never stops pulling / tearing down.
Two fixes: (1) test the generator function directly (`next(gen)`; `gen.close()`) and the manager's
`frame()` rather than driving the infinite stream through TestClient; (2) make the generator loop on
`manager.is_running()` so real shutdown ends it cleanly (in production Starlette stops pulling it on
disconnect, but don't rely on that for the stop path). Lesson: never drive an unbounded streaming
response through TestClient — exercise the generator in isolation.

## 2026-06-18 — Domain randomization needs *data scale* to pay off — it backfires when small (v2 Phase A)
Building the robust-perception comparison, the unit-scale check (`n=300`) showed the **opposite** of the
full-scale result: the DR-trained model got **0.61** on the randomized set vs the clean model's **0.72** —
robust *lost*. At full scale (`n=720`) it flips to 0.844 vs 0.744 (+10 pts). With little data the DR model
underfits the much harder (occlusion + noise + blur) distribution while the clean model coasts on the easy
fraction of randomized samples. Lesson: don't assert a scale-sensitive ML *outcome* in a fast unit test —
it'll be flaky-by-construction. Instead the suite proves the **mechanism** that always holds (a clean model
*degrades* under randomization) with one small training run, and locks the **headline** (robust > clean) by
asserting the committed full-scale `metrics.json`. The real proof is the regenerated artifact + `make train`.

## 2026-06-18 — A 3D occluder's *projected* shadow is bigger than its physical size (v2 Phase A)
The status-light occluder is a small box placed ~0.12 m *in front* of the light (toward the camera), so on
the image it covers more than its half-extent would suggest — a box smaller than the light sphere can still
shadow most of it. My first justification ("half-extent < light radius ⇒ never fully hidden") was wrong
reasoning even though the bound happened to hold. The "never fully hidden / always labelable" invariant is
**empirical** (tuned so worst case leaves ~13 of 93 light px visible), not derivable from the half-extent.
Lesson: for anything between the camera and the cue, reason in *image/projected* space, not object space.

## 2026-06-18 — Subagents load at session start
Files added to `.claude/agents/` are NOT available mid-session — they're read when the session
starts. After creating/editing them, restart Claude Code (or add via `/agents`) before trying to
delegate, or the Agent call won't find the custom agent. This is why Phase 0 ends with a
mandatory restart before Phase 1.
