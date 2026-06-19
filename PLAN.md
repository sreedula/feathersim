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

- [x] `README.md` with an autonomy-loop GIF + one-command launch (`scripts/record_gif.py` → `docs/autonomy.gif`)
- [x] All tests green (94 passing)
- [x] Four engineering docs filled in

**Project complete.** Phases 0–6 shipped; the walking skeleton is a full machine-tending autonomy stack
— sim → mecanum kinematics → go-to-pose control → skill SDK → auto-labeled perception (state acc 1.0 vs
0.39 baseline) → unattended autonomy loop → browser teleop/telemetry dashboard.

---

# FeatherSim v2 — the 10x expansion

Five hard systems stacked on the working v1. Same engineering loop, same subagents, same checkpoints.
Extend v1 modules; don't rewrite them. The dashboard stays launchable at the end of every phase.

## Phase A — Brutal perception (domain randomization)  `[x]`  ← checkpoint (approach + before/after acc)

**Approach (confirmed):** a seeded `DomainRandomizer` threaded through the
render/auto-label pipeline, split into 3D-scene randomization (before render) and image-space sensor
corruption (after render):
- *Scene (mutate the MuJoCo model per sample):* randomize the worldbody `<light>` position, diffuse
  intensity, and color tint; add a per-machine non-colliding `occluder_i` geom placed at a random offset
  partially in front of the status light (random size, present with probability `p`) — a real 3D occluder
  that shades and parallaxes correctly, not a painted patch.
- *Sensor (pure numpy/PIL on the uint8 crop):* additive Gaussian noise (random σ) + directional motion
  blur (random angle/length kernel). Pure, unit-testable functions — no sim needed.
- The status-light **label color** (gray/amber/green) is never randomized — that's the signal; only its
  *appearance* under randomized illumination / occlusion / noise / blur is.

**Experiment:** train two equally-sized models — `clean` (DR off, the v1 recipe) and `robust` (DR on) —
then evaluate **both** on a randomized held-out set *and* a clean held-out set (a 2×2 matrix). Deploy the
robust model as `model.pt`; keep `model_clean.pt` for comparison (both gitignored). All numbers →
committed `metrics.json`.

**Acceptance:** robust model beats clean model on the randomized held-out set by a clear margin; the
clean model degrades from its clean accuracy under randomization while the robust model holds; the 2×2
accuracy matrix + majority baseline are logged to `metrics.json`. Tests: each augmentation is a
deterministic pure function (same seed → identical output; shape/dtype preserved; pixels actually
change); DR is reproducible; robust ≥ clean on the randomized set. Green.

- [x] `perception/randomize.py`: `DomainRandomizer` config + pure `gaussian_noise` / `motion_blur` + scene helpers
- [x] `sim/world.py`: per-machine `occluder_i` geom + `set_occluder` / `randomize_lighting` / `reset_scene`
- [x] `perception/dataset.py`: `generate_dataset(..., randomizer=...)` threads DR through; class balance preserved
- [x] `perception/train.py`: train clean + robust; 2×2 eval matrix → `metrics.json`; save `model.pt` (robust) + `model_clean.pt`
- [x] Tests: augmentation determinism/shape; DR changes pixels; clean model degrades; committed metrics show robust>clean
- [x] **Checkpoint: confirm randomization approach + show before/after accuracy**

> **Result:** under randomization clean model 74.4% → robust model 84.4% (+10 pts); both 1.0 on clean;
> baseline 37.2%. (committed `metrics.json`).
> Reviewer: SHIP (no CRITICAL/HIGH). Label-color invariant, determinism/no-leak, and no train/serve gap
> all verified. Fixed in-phase: corrected the occluder-bound comment (projected shadow ≠ physical size),
> `gaussian_noise(σ≤0)` returns a copy, dropped a `type: ignore`, documented the metrics.json schema.
> Deferred MEDIUM: the "robust>clean" headline is asserted from the committed full-scale `metrics.json`,
> not retrained in CI (it reverses at small n — see LEARNINGS); the suite proves only the degradation
> mechanism live. The robust model is now `model.pt` (deployed); demo still tends 6 parts (2/2/2).

## Phase B — Path planning + obstacle avoidance  `[x]`

**Acceptance:** occupancy grid of the floor (machines, table, obstacles); A* global planner to a target
pose; a waypoint-follower drives the path via the existing mecanum kinematics; 1–2 static obstacles the
robot routes around. Unit tests on the planner (finds path / reports no-path); robot reaches targets
without intersecting obstacles in sim.

- [x] `planning/occupancy.py` (pure): `Rect`, `OccupancyGrid`, `build_grid` (inflate by robot radius)
- [x] `planning/astar.py` (pure): 8-connected A* (octile, no corner-cut) + `plan_path` (world coords + LOS smoothing)
- [x] `planning/follow.py`: `follow_path` waypoint follower reusing `drive_to_pose` per leg
- [x] `sim/world.py`: optional `n_obstacles` pillars; `obstacle_rects` / `occupancy_grid`
- [x] `sdk/robot.py`: opt-in `Robot(plan=True)` → `move_to` plans + follows (default off = v1 behavior)
- [x] `demo.py`: obstacles + planning; robot routes around, still delivers 6 parts
- [x] Tests: A* path/no-path/around-barrier; body clears obstacles on every driven leg; tending poses reachable

> Reviewer: SHIP after **two** NEEDS-WORK rounds. R1: test asserted clearance>0.1 but body radius is 0.2
> (didn't prove body non-intersection). R2 (CRITICAL): the fixed test sampled only table→machine_2 (easy
> dir); the machine→table return legs bowed to 0.169 m — **body 0.031 m inside a non-colliding pillar,
> silently**. Root cause: grid inflation protects the *center*, but the P-controlled follower bows outside
> the validated segments on turns. Fix: tightened follower `waypoint_tolerance` 0.08→0.04, obstacle-only
> `OBSTACLE_CLEARANCE=0.08`, repositioned pillars to (±0.62,0) clear of tending corridors, parametrized the
> safety test over **all** driven legs (table↔machine + origin-start + machine→machine). Worst-case body
> clearance now ~0.29 m vs 0.2 m radius. Both gotchas logged in LEARNINGS.md. Deferred LOW: static grid
> built once (Phase C rebuilds per-step for moving robots).

## Phase C — Multi-robot fleet + scheduling  `[x]`  ← checkpoint (coordination + collision avoidance)

**Acceptance:** 2–3 robots in the MJCF (each its own base, camera, perception read); a fleet manager
allocates `done` machines so two robots never target the same one; inter-robot collision avoidance
(cell reservation on the grid, or mutual yielding); ≥2 scheduling strategies (e.g. nearest-done vs
longest-waiting) with throughput measured for each. N robots tend M machines unattended without
colliding or double-assigning; throughput logged per strategy.

- [x] Multi-robot `World` (`n_robots`, indexed base methods, `driver(k)`); `Robot(robot_id=)` — v1 default preserved
- [x] `fleet/scheduling.py` (pure `longest_waiting`/`nearest_done`) + `fleet/manager.py` (lock — no double-assign)
- [x] `fleet/executor.py`: tick loop, per-robot SM, plan-around-others + symmetric contact backstop, per-robot perception, `FleetReport`
- [x] `fleet/demo.py` + `make fleet`: 3-robot open-floor fleet, both strategies, collision-free
- [x] Tests: scheduling/allocation (pure), no-double-assign, collision-free over **8 seeds**, compose-with-obstacles, false-positive recovery, GL e2e

> **Confirmed at checkpoint:** prioritized avoidance + per-robot randomized perception (user: "yea go").
> Reviewer: SHIP after **two** NEEDS-WORK rounds. R1 (CRITICAL): collision-freedom was seed-tuned — bodies
> overlapped on ~½ of seeds; the test pinned `seed=0`. Root cause: a never-yielding priority leader
> rear-ends a yielded follower. Fixed with a **symmetric predictive backstop** (verified collision-free
> over 160 runs, worst 0.432 m vs 0.40 m) + the test now sweeps 8 seeds + `FleetReport.completed` surfaces
> timeouts. R2 (MEDIUM/LOW): removed a seed-fragile fairness assertion; refreshed stale priority-scheme
> docstrings. Findings logged in LEARNINGS.md. Deferred: tight cells (3 robots + pillars, or 2+pillars on
> ~13% of seeds) can wedge until the time budget — bounded + surfaced; demo uses 3 robots on open floor.
> Strategies **tie on throughput** (robot-saturated → robot-limited); they'd differ on wait/travel.

## Phase D — Learned policy (behavior cloning)  `[x]`  ← checkpoint (BC setup + GPU/CPU)

**Acceptance:** log (observation → action) pairs from the v1 hand-coded controller (the "expert") over
many cycles; train a small policy net to imitate it (behavior cloning); run the loop driven by the
learned policy and compare throughput/behavior to the expert. Learned policy runs end-to-end and
approaches the expert's throughput; comparison logged. Flag if CPU-only training is too slow to run here.

- [x] `control/go_to_pose.py`: `goal_in_body_frame` (shared BC obs) + `velocity_fn` param on `drive_to_pose`
- [x] `policy/`: `PolicyMLP` (3→128→128→3 tanh), `dataset` (expert twists), `PolicyController` (drop-in `velocity_fn`), `train` (BC + closed-loop compare), `demo`, `make policy`
- [x] `sdk/robot.py` + `planning/follow.py`: `controller` threaded through **both** straight-line and planned branches
- [x] Tests: dataset==expert, BC matches expert, drives closed-loop, honors planning, committed metrics, GL loop-on-policy
- [x] **Checkpoint: confirmed BC setup + CPU decision** (user: "yea go on"); CPU sufficient (tiny MLP, seconds)

> **Result:** val MSE 1.5e-3; learned policy reaches 100% of goals (= expert) in 208 steps vs 224; full
> loop delivers 6 parts at **112% of the expert's throughput** (committed `metrics_policy.json`).
> Reviewer: SHIP after one NEEDS-WORK round. Fixed: HIGH — `velocity_fn` now threads through `follow_path`
> so `plan=True` honors the learned controller (was silently dropped); MEDIUM — `ACTION_SCALE` derived from
> `PoseGains` (coupling explicit), dropped the dead/misleading clamp in `PolicyController`; LOW — robust
> demo GL guard. Findings logged: offline BC loss doesn't predict closed-loop reach; tanh head can't hit a
> saturated target (LEARNINGS). Deferred: the BC is a *soft* task (memoryless control law) — the headline
> is the pipeline + closed-loop deploy, not difficulty; obstacle-aware policy offered + deferred.

## Phase E — Command-center dashboard  `[x]`  ← checkpoint (final demo review)

**Acceptance:** dashboard shows all robots at once, each robot's planned path overlaid, per-robot
perceived-vs-true machine states, current task assignments, and live scheduling decisions; a toggle
between hand-coded controller and learned policy; a perception-difficulty slider that dials domain
randomization up/down live and shows accuracy reacting. All live in the browser.

- [x] Extracted `FleetController` (shared by `run_fleet` + dashboard); behavior-preserving (Phase-C tests pass)
- [x] `dashboard/fleet_manager.py`: threaded fleet, per-robot DR perception, top-down schematic w/ planned paths, telemetry
- [x] `dashboard/fleet_server.py` + `static/fleet.html`: command center (telemetry, schematic MJPEG, controller toggle, difficulty slider)
- [x] `DomainRandomizer.at_difficulty` + `load_or_train_clean_model` (robust vs clean accuracy, live)
- [x] `make dashboard` → command center; `make teleop` → Phase-6 dashboard
- [x] Tests: controls, telemetry shape, schematic JPEG, slider robust>clean divergence, routes

> Reviewer: SHIP (no CRITICAL/HIGH). Threading verified (HTTP reads published snapshots under lock;
> control writes atomic/lock-free); `FleetController` refactor behavior-preserving; slider divergence real
> (clean 1.0→0.79, robust holds 0.91–1.0). Fixed in-build: the slider only bit once `_perceive` applied the
> *scene* stage (occluders), not just sensor noise (LEARNINGS). Deferred LOWs: scale occluder/blur extents
> with difficulty; `policy.py` twist annotated `-> Pose` (alias reuse). **The final-demo review checkpoint
> is the message to the user below.**

## v2 Definition of done

- [x] `README.md` updated with a multi-robot fleet GIF (`scripts/record_fleet_gif.py` → `docs/fleet.gif`), the rule-vs-learned toggle, and the difficulty slider
- [x] One-command launch (`make dashboard`); all tests passing (162)
- [x] `DECISIONS.md` + `LEARNINGS.md` reflect every v2 choice

**v2 complete.** Five systems stacked on the v1 walking skeleton: domain-randomized perception (robust
84.4% vs clean 74.4% under DR) → A* path planning + obstacle avoidance → multi-robot fleet (collision-free
over 160 seeded runs, scheduling measured) → behavior-cloned policy (112% of expert throughput) →
command-center dashboard (paths, per-robot perception, controller toggle, live difficulty slider).

---

# v3 — "make it look like a real sim" (ongoing polish loop)

## Iteration 1 — Cinematic world + live 3D feed  `[x]`

**Done:** glossy materials, gradient skybox, checker floor with reflections, shadows, two-tier robots
(`build_mjcf` `<visual>`/`<asset>`); a **live 3D overview feed** in the command center (`/api/camera3d`,
the hero view) alongside the tactical schematic. Perception retrained on the new look. Two perception
upgrades: **relative-lighting DR** (key light jittered around its authored pose; `reset_scene` restores it
→ no train/serve gap, feed stays lit) and a **mix-trained robust model** (clean+randomized → **1.0 clean,
0.91 under DR, +19 pts** over the clean model). Reviewer caught the `reset_scene` key-light clobber; fixed
+ pinned by a test. 163 tests green; demos + dashboard verified.

**Next ideas:** manipulator arms + physical part transport; a parts stack on the table; animated machine
doors; per-robot onboard camera feeds in the dashboard.

## Iteration 2 — Manipulator arms + physical part transport  `[x]`

**Done:** each robot gains an articulated **arm + gripper** (non-colliding, kinematic base); a part
**physically rides the gripper** between pick and place; delivered parts **pile up on the output table**
(a 4×3 stack). `Robot.pick/place` toggle the carried-part visual and grow the stack via `World.set_carried`
/ `deposit_part`. It now reads as a real machine-tending cell. Reviewer SHIP (additive visual; no race, no
perception leak — verified 0 px into machine crops; carried flag balanced). Added 4 transport tests +
documented the SM-arrival ≤ SDK-`_at` tolerance coupling. 167 tests green.

## Iteration 3 — Animated arms (reach / grasp / retract)  `[x]`

**Done:** robot arms now visibly **reach into a machine to grasp** on pick and **extend over the table to
place**. The arm is a hinged sub-body slewed kinematically in `World.step` (`set_arm_target`/`arm_at`); the
fleet SM runs reach→grasp→retract sub-phases. `gravcomp="1"` + qvel-zeroing make the arm dynamically inert
(base stays exactly static). Reviewer SHIP — no defects; corrected mis-attributed drift comments (the arm
adds zero drift; the real culprit was residual drive velocity). Fleet still collision-free + delivers
(8-seed sweep, worst min_sep 0.442). +1 arm test. 168 tests green.

**Next ideas:** animate the single-robot loop's arm too; per-robot onboard camera feeds; animated machine
doors; scale to more machines / a bigger cell.

## Iteration 4 — Per-robot onboard camera feeds  `[x]`

**Done:** each robot has a forward-facing onboard `<camera>`; the command center renders all of them into
a colour-labelled strip (`/api/robotcams`) shown under the 3D feed — each robot's eye-view of the machine
it's tending, its own gripper reaching in, and the part. Same thread-safe sim-thread-render → cached-bytes
pattern as the 3D feed; non-physical cameras, no perception/logic impact. +1 test (strip is 3×CAM_SIZE).
169 tests green.

## Iteration 5 — Scale to a 4-robot / 4-machine floor + deadlock-free coordination  `[x]`

**Done:** the headline floor is now **4 robots tending 4 machines**. Scaling exposed the symmetric backstop's
deadlock (clusters freeze); fixed with a **stuck-triggered priority-yield deadlock breaker** (lower-id
robots back away from higher-id ones only once genuinely stuck) + a wider `_MIN_SEP` for simultaneous
closure. Machine spacing is now fixed so perception generalizes to 4 machines **with no retrain (100%)**.
Reviewer SHIP — no collision path, no permanent deadlock, fair throughput, behavior-identical 3-robot
configs. Verified collision-free + every-part-delivered across 40 seeds (worst body sep 0.45 m). 176 green.

**Deferred (reviewer, non-blocking):** id-priority isn't throughput-optimal and a rare seed takes ~90 s to
unwind a near-wedge (the 0.02 m `_MIN_SEP`/`_SLOT_SPACING` margin); the proper long-term fix is ORCA /
prioritized planning. Logged in LEARNINGS + DECISIONS.

**Next ideas:** ORCA-style local avoidance (true deadlock-free coordination); animate the single-robot
loop's arm; per-robot route colors in the 3D feed.
