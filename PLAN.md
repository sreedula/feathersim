# FeatherSim — PLAN.md

Phased roadmap. Each phase has acceptance criteria and a task checklist. The system stays
runnable & demoable at the end of every phase (walking skeleton). Nothing advances on red.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress

---

## Phase 0 — Scaffold, subagents & loop setup  `[~]`

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
- [ ] **Checkpoint: show structure, plan & subagents, then STOP — user restarts so agents load before Phase 1**

## Phase 1 — Sim world  `[ ]`

**Acceptance:** headless PyBullet sim steps deterministically; 2–3 machines cycle
`idle→running→done` on a timer; `scripts/print_state.py` prints ground-truth state each tick.

- [ ] PyBullet world: floor, holonomic robot base, parts table
- [ ] 2–3 machine bodies, each with a door + `MachineState` (idle/running/done)
- [ ] Machine state machine on a timer (pure, unit-testable)
- [ ] Headless `step()` loop; deterministic seeding
- [ ] `scripts/print_state.py` prints per-tick ground truth
- [ ] Tests: machine state transitions

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
