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

## Phase 4 — Perception + auto-labeling  `[ ]`  ← checkpoint (metrics)

**Acceptance:** model beats a trivial baseline on a held-out sim set; metrics logged.

- [ ] `dataset.py`: render camera frames + auto-label from sim ground truth
- [ ] `model.py`: small CNN (machine-state + part-present heads)
- [ ] `train.py`: train/val split, metrics logging
- [ ] `infer.py`: `perception.read(camera) -> state`
- [ ] Held-out accuracy beats majority-class baseline; metrics committed

## Phase 5 — Autonomy loop  `[ ]`

**Acceptance:** robot tends N machines continuously for M cycles with no manual input.

- [ ] Loop: perceive → pick a `done` machine → navigate → unload → load → repeat
- [ ] Uses perception (not ground truth) for skill selection
- [ ] Runs N machines × M cycles unattended; logs throughput/uptime
- [ ] Tests: loop selects correct machine from perceived state

## Phase 6 — Teleop + dashboard  `[ ]`

**Acceptance:** watch the loop in a browser and seize manual control.

- [ ] FastAPI server: camera-feed endpoint + telemetry (per-machine state, parts done, uptime, current skill)
- [ ] Single-file vanilla-JS UI
- [ ] WASD/joystick manual override that preempts autonomy
- [ ] `make dashboard` launches it

## Definition of done

- [ ] `README.md` with an autonomy-loop GIF + one-command launch
- [ ] All tests green
- [ ] Four engineering docs filled in
