# FeatherSim — PLAN.md

Phased roadmap. Each phase has acceptance criteria and a task checklist. The system stays
runnable & demoable at the end of every phase (walking skeleton). Nothing advances on red.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress

---

## Phase 0 — Scaffold, subagents & loop setup  `[x]`

**Acceptance:** `pytest` runs green (≥1 trivial test); the four engineering docs exist; the three
subagent files in `.claude/agents/` are valid; a single-command launch (`make demo`) and test
runner (`make test`) work; git initialized.

- [x] Repo structure (package + subpackages + tests)
- [x] `CLAUDE.md`, `PLAN.md`, `DECISIONS.md`, `LEARNINGS.md`
- [x] `requirements.txt`, `pyproject.toml`, `Makefile`, `.gitignore`, `README.md`
- [x] Trivial passing test (`tests/test_smoke.py`)
- [x] `python -m feathersim.demo` runs
- [x] `git init` + first commit
- [x] Subagents: `.claude/agents/{reviewer,test-runner,docs-researcher}.md`
- [x] **Checkpoint: show structure, plan & subagents, then STOP — user restarts so agents load before Phase 1**

## Phase 1 — Sim world  `[x]`

**Acceptance:** a headless MuJoCo world (floor, holonomic robot base, parts table, 2–3 machines)
steps deterministically — same seed → identical state trace; each machine runs a pure, timer-driven
FSM `idle→running→done` (+ `reset` on unload); `scripts/print_state.py` prints per-tick ground truth
(sim time, each machine's state, robot pose). Tests cover the pure FSM and sim determinism — green.

- [x] MJCF world: floor, holonomic robot base (planar x/y/yaw joints), parts table
- [x] 2–3 machine bodies, each with a door geom + `MachineState` (idle/running/done)
- [x] Pure, unit-testable timer FSM (`feathersim/sim/machine.py`), no MuJoCo import
- [x] `World.step()` advances physics + ticks FSMs; deterministic `seed`
- [x] `scripts/print_state.py` prints per-tick ground truth
- [x] Tests: FSM transitions (pure) + sim determinism (same seed → same trace)

> Deferred (reviewer LOW, none blocking): in Phase 2 assert base motion *under* a control input
> (the Phase-1 "stays at origin" test can't fail by construction); keep `states()` ground-truth vs.
> Phase-4 perceived state explicitly separated.

## Phase 2 — Holonomic motion  `[x]`  ← checkpoint (confirm kinematics)

**Approach (confirmed):** 4-wheel **mecanum** base. Pure inverse/forward kinematics; the control
loop routes the commanded body twist *through* wheel space (IK→FK) each tick so the wheel math is
load-bearing, then drives the sim base's planar joints. Idealized base (kinematic velocity control,
no contact/obstacle avoidance) — fine for free-space navigation.

**Acceptance:** pure unit tests on the mecanum math pass (incl. body→wheels→body round-trip is
identity); a P-controller drives the sim base from the origin to a commanded `(x, y, θ)` — including
strafe + rotation — within position/heading tolerance, and halts on arrival. Green.

- [x] `kinematics/holonomic.py`: `MecanumGeometry`, `body_to_wheels` (vx,vy,ω → 4 wheel speeds),
      `wheels_to_body` (FK) — pure, no sim import
- [x] Inverse + forward kinematics; body→wheels→body round-trip tests
- [x] `control/go_to_pose.py`: pure `velocity_command` (world error → body twist, clamped) +
      `drive_to_pose(world, target)` sim driver routing through the mecanum IK/FK
- [x] `World.command_base_velocity` (body twist → world-frame joint velocities) + `stop_base`
- [x] Sim test: reaches pose within position/heading tolerance; base halts after arrival
- [x] **Checkpoint: confirm kinematics approach** — mecanum, confirmed with user

> Reviewer: SHIP, all findings LOW and addressed in-phase (dead line removed, `wrap_to_pi` docstring
> softened, `BaseDriver` Protocol added for the sim seam). `DriveResult.steps` early-return vs.
> timeout asymmetry is an intentional, tested choice.

## Phase 3 — Skill SDK  `[x]`

**Approach (confirmed):** a `Robot` facade wrapping a `World`, hiding joints/MJCF/kinematics. Parts
are logical handoffs (no arm): `pick` unloads a `done` machine (FSM `reset`, which auto-reloads it),
`place` deposits onto the output table. Skills enforce pre/postconditions and raise `SkillError`.

**Acceptance:** a ~10-line script tends one machine end-to-end using ONLY SDK calls; skills enforce
pre/postconditions (raise on violation); tests cover composition + each pre/postcondition. Green.

- [x] `sdk/robot.py`: `Robot` facade — `move_to(pose|fixture)` on go-to-pose; `wait_until_done`
- [x] `pick(machine)` (precond: at machine + `done` + not holding) / `place(target)` (precond: holding)
- [x] `tend(machine)` composing move→pick→move→place; `SkillError` on precondition violations
- [x] `World`: `fixtures` (ground-truth positions) + `set_base_pose` (reset/teleport)
- [x] `examples/tend_one_machine.py`: tend one machine end-to-end via the SDK only
- [x] Tests: skill composition + every pre/postcondition

> Reviewer: SHIP. MEDIUM (example leaked `world.machines[...]`, undercutting "SDK-only") fixed by
> adding `Robot.machine_state` / `parts_done` accessors; LOWs addressed (off-contract target → clean
> `SkillError`; typed `world` param; `tending_pose` y=0 fallback note). `place` explicit-pose path now
> tested. 63 tests green.

## Phase 4 — Perception + auto-labeling  `[x]`  ← checkpoint (metrics)

**Approach (confirmed):** make machine state visually observable — a per-machine **status light**
(idle=gray, running=amber, done=green) + a **bed part** geom toggled for `part-present`. A per-machine
**cropped camera** renders the close-up. Dataset is auto-labeled: place the sim in randomized
ground-truth configs (state × part-present, independently sampled for class balance + decorrelated
heads), render, read labels straight from the config — zero hand labeling. Small 2-head CNN.

**Acceptance:** held-out machine-state accuracy **beats the majority-class baseline** by a clear
margin; metrics logged to a committed file; `perception.read` returns predicted (state, part-present)
from a camera frame. Tests deterministic (seeded) — green.  **Result: state acc 1.0 vs 0.39 baseline.**

- [x] `sim/world.py`: per-machine `light_i` + `part_i` geoms (non-colliding); `set_machine_visual` /
      `sync_visuals`; per-machine `MjvCamera`; render helper
- [x] `perception/dataset.py`: render crops + auto-label from ground-truth configs (balanced split)
- [x] `perception/model.py`: small 2-head CNN (state: 3-class, part-present: binary), max+avg pool
- [x] `perception/train.py`: train/val, metrics → `metrics.json` (committed); `make train` CLI
- [x] `perception/infer.py`: `Perception.read(image)` / `perceive(world)` → predicted state
- [x] Tests: dataset shapes/labels, model output shapes, train beats baseline, infer correctness

> Reviewer: SHIP, all LOW. Addressed in-phase: `_PART_RGBA` wired into MJCF (no dup), serve-time
> part-coupling documented in `sync_visuals`. Deferred LOW: per-machine camera is loosely cropped
> (neighbors graze the frame edge) — revisit only if Phase-5 perception gets noisier. Render tests
> skip on a headless host without a GL backend (set `MUJOCO_GL=egl`).

## Phase 5 — Autonomy loop  `[x]`

**Approach (confirmed):** an unattended `run_autonomy(world, perception, renderer, ...)` loop that
composes Phases 2–4. Each iteration it calls `perception.perceive(world, renderer)` (predictions from
pixels, **never** `world.states()`), picks the machine **perceived** `done` that has been **waiting
longest** (oldest-first — fair and throughput-maximizing, since a done machine stops cycling; an optional
`min_confidence` gate drops low-confidence readings), and tends it via the SDK (`Robot.tend` → navigate →
unload → carry → place). If nothing is perceived done it advances the sim a beat and re-perceives; a false
positive raises `PreconditionError` and the loop drops that candidate and continues, while a genuine
navigation failure (plain `SkillError`) is *not* swallowed. Returns an `AutonomyReport` (parts delivered,
sim uptime, throughput/min, per-machine counts, event log). `python -m feathersim.demo` runs it headless.

**Acceptance:** robot tends N machines continuously for M cycles with no manual input; skill selection
is driven by **perceived** state (proven by a test where perception disagrees with ground truth and
the loop follows perception); the loop is robust to perception false positives; throughput/uptime
logged. Green.

- [x] `autonomy/loop.py`: `run_autonomy` (perceive → pick perceived-`done` → `tend` → repeat) + `AutonomyReport`
- [x] Uses perception (`perceive`), not ground truth, for skill selection; recovers from false positives
- [x] Runs N machines × M cycles unattended; report logs throughput + uptime
- [x] `train.load_or_train_model` (load `model.pt` or train a fresh one) so the demo is one command
- [x] `demo.py`: wire world + perception + loop; print live tends + final throughput/uptime
- [x] Tests: loop follows **perceived** state (disagrees-with-truth case); skips false positives; e2e unattended run

> Reviewer: SHIP (after one NEEDS-WORK round, all findings addressed in-phase). Fixed: HIGH
> all-false-positive termination now has a bounded-run test + "every iteration progresses" invariant;
> MEDIUM nav-failure masking → `PreconditionError(SkillError)` taxonomy so the loop swallows only
> precondition misses and a real nav failure propagates loudly; MEDIUM typed the `Perceiver` seam.
> LOWs: `min_confidence` gate added, PLAN reconciled to oldest-first, demo GL guard, docstrings.
> The demo (not the unit tests) caught a scheduler-fairness bug — confidence tie-break starved
> `machine_0`; switched to oldest-waiting-first (even 2/2/2). See LEARNINGS.md.

## Phase 6 — Teleop + dashboard  `[x]`

**Approach (confirmed):** a `SimManager` owns the `World` + `Robot` + `Perception` and runs the sim on a
single background thread (MuJoCo isn't thread-safe), so all stepping/rendering happens there and HTTP
handlers only read a published snapshot (telemetry dict + latest JPEG) under a lock. Autonomy is
re-expressed as a **tick-based, preemptible** state machine (`select → to_machine → pick → to_table →
place`) reusing the pure `velocity_command` + SDK skills — so a teleop command can seize control
mid-skill and resume on release. FastAPI server (`create_app`, exposes `app` for `make dashboard`):
MJPEG camera feed, `/api/telemetry`, `/api/teleop` (seizes manual), `/api/mode` (resume auto). Single
`static/index.html` with inline vanilla JS. `SimManager` takes injected perception + `render=False` so
the control logic tests run headless; route tests use `TestClient`; the camera route is render-guarded.

**Acceptance:** open `make dashboard` in a browser, watch the autonomy loop run live (feed + per-machine
telemetry + throughput/uptime), press WASD to seize manual control (autonomy pauses), release/Resume to
hand back. Green.

- [x] `World.overview_camera()` + `render()` (DRY with `render_machine`)
- [x] `dashboard/sim_manager.py`: `SimManager` — threaded sim, AUTO/MANUAL modes, tick-based autonomy SM, telemetry snapshot, JPEG feed
- [x] `dashboard/server.py`: `create_app` + routes (`/`, `/api/telemetry`, `/api/teleop`, `/api/mode`, MJPEG `/api/camera`)
- [x] `dashboard/static/index.html`: single-file UI — live feed, machine cards, WASD teleop, mode toggle
- [x] `make dashboard` launches it (uvicorn `...:app`)
- [x] Tests: SimManager autonomy tends via perceived state; teleop preempts autonomy; mode resume; telemetry shape; routes via TestClient

> Reviewer: SHIP (no CRITICAL/HIGH). Thread-safety verified — only `_mode`/`_teleop`/`_snapshot`/`_frame`
> cross threads, all behind the lock; no HTTP-thread sim/renderer access; preempt-and-resume genuinely
> holds (pick/place atomic per tick, held part survives via `robot.holding`, stale pick fails cleanly to
> `select`). Addressed in-phase: `stop()` warns on join timeout; MJPEG generator loops on `is_running()`
> so shutdown ends it; dropped `--reload` (stateful in-process sim); teleop clamps resultant speed (vector
> magnitude), not per-axis; throughput-over-total-sim-time documented. Deferred LOW: backgrounded-tab still
> drives the 20fps render loop (no `visibilitychange` teardown); `_skill_text`/`_target_machine` coupling
> is documented-only (no live bug). The GL-thread-affinity render bug is logged in LEARNINGS.md.

## Definition of done

- [ ] `README.md` with an autonomy-loop GIF + one-command launch
- [ ] All tests green
- [ ] Four engineering docs filled in
