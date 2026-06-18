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

## Phase 2 — Holonomic motion  `[ ]`  ← checkpoint (confirm kinematics)

**Acceptance:** unit tests on the kinematics math pass; robot reaches a commanded floor pose
within tolerance in sim.

- [ ] `holonomic.py`: body velocity (vx, vy, ω) → wheel commands (pure functions)
- [ ] Inverse + forward kinematics; round-trip tests
- [ ] `go_to_pose.py`: controller drives base to target (x, y, θ)
- [ ] Sim test: reaches pose within position/heading tolerance
- [ ] **Checkpoint: confirm kinematics approach**

## Phase 3 — Skill SDK  `[ ]`

**Acceptance:** a ~10-line script tends one machine end-to-end using only SDK calls.

- [ ] `robot.move_to(pose)` on top of go-to-pose
- [ ] `robot.pick(part)` / `robot.place(target)`
- [ ] `robot.tend(machine)` composing the above
- [ ] Example script: tend one machine end-to-end
- [ ] Tests: skill composition / pre + postconditions

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
