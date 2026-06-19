# FeatherSim — DECISIONS.md

Architecture decision log. Each entry: date, decision, why, tradeoff.

## 2026-06-18 — Separate sibling repo, not inside loci-mvp
**Decision:** FeatherSim lives at `/Users/sreekare/feathersim` as its own git repo.
**Why:** The invocation directory (`loci-mvp`) is an unrelated React/Three.js app with its own
`CLAUDE.md` and history. Mixing a Python robotics stack in would collide and confuse.
**Tradeoff:** Two repos to manage vs. one; chosen for clean isolation.

## 2026-06-18 — ~~PyBullet for simulation~~ (SUPERSEDED, same day — see below)
**Decision:** PyBullet over MuJoCo for the sim world.
**Why:** Fast iteration, trivial install, easy headless stepping and synthetic camera rendering
(`getCameraImage`) for the auto-labeling pipeline; ample examples.
**Tradeoff:** Less contact-accurate than MuJoCo, but machine-tending here is coarse pick/place +
base navigation, so fidelity is sufficient.
**Superseded by the MuJoCo decision below:** the "trivial install" premise failed on this machine.

## 2026-06-18 — MuJoCo for simulation (reverses the PyBullet decision above)
**Decision:** MuJoCo, not PyBullet, for the sim world.
**Why:** PyBullet was chosen *for* trivial install + headless camera rendering. On this machine
(macOS, Apple clang 17, framework Python 3.13) PyBullet ships no wheels and its source build fails:
its vendored zlib collides with the new macOS SDK's `_stdio.h` (`NULL`/`ZEXPORT` token errors).
So the premise no longer holds here. MuJoCo's official wheels install in seconds with no compiler,
support Python 3.13 on arm64, step deterministically headless, and render offscreen camera frames
(`mujoco.Renderer`) — exactly what the auto-labeling pipeline needs. Verified all three before switching.
**Tradeoff:** Worlds are authored in MJCF instead of URDF and the API differs from PyBullet; the
kinematics/perception/SDK/autonomy layers are sim-agnostic by design (sim state injected at the edges),
so the blast radius is confined to `feathersim/sim/`.

## 2026-06-18 — Perception: visual state cues + balanced auto-labeling from ground-truth configs
**Decision:** Make machine state visually observable (per-machine **status light** idle=gray /
running=amber / done=green, plus a **bed-part** geom for part-present), render a per-machine cropped
camera, and auto-label by placing the sim in **randomized ground-truth configurations** — every
machine lit with an independently sampled (state, part) — then reading labels straight from the config.
A small 2-head CNN (concat max+avg global pool) predicts state (3-class) + part-present (binary).
**Why:** A vision model needs a visual cue, so state had to be made renderable. Sampling configs
directly (rather than only logging a running sim) gives **class balance** and **decorrelated heads**
(part-present sampled independently of state), so the part head is a real task, not a shadow of state.
Lighting *all* machines in the dataset matches what the live camera sees (`sync_visuals` lights all),
eliminating a train/serve gap. Result: held-out state accuracy 1.0 vs a 0.39 majority baseline.
**Tradeoff:** The cues are synthetic and high-contrast, so the task is easy — fine, the headline is the
auto-labeling pipeline + closing the loop, not hard perception. The per-machine crop is only loosely
cropped (neighbors graze the frame edge); acceptable because all machines are lit consistently. Cue
geoms are non-colliding so they don't obstruct navigation. Metrics committed to `perception/metrics.json`;
the trained `model.pt` is gitignored (regenerate via `make train`).

## 2026-06-18 — Kinematics & perception logic as pure functions
**Decision:** Drive kinematics and perception decision logic live in pure functions with no sim
import; sim state is injected at the edges.
**Why:** Lets us unit-test the math/logic fast without spinning up PyBullet, per `CLAUDE.md`.
**Tradeoff:** A little extra plumbing to pass state in — worth it for test speed/reliability.

## 2026-06-18 — Perception starts as a small PyTorch CNN
**Decision:** Begin with a small CNN with two heads (machine state, part-present) trained on
auto-labeled sim renders; YOLO-style detection only if classification proves insufficient.
**Why:** The headline is the auto-labeling pipeline + closing the loop, not SOTA detection.
A small CNN trains fast on sim data and is easy to evaluate against a baseline.
**Tradeoff:** Won't localize objects precisely; acceptable since fixtures are at known poses.

## 2026-06-18 — FastAPI + single-file vanilla JS for the dashboard
**Decision:** FastAPI backend, one static `index.html` with vanilla JS for feed/teleop/telemetry.
**Why:** No build step, minimal deps, easy to demo; matches the "single-file frontend" guidance.
**Tradeoff:** Not a rich SPA; fine for a demo dashboard.

## 2026-06-18 — Mecanum drive, with control routed through wheel space; kinematic base velocity
**Decision:** The holonomic base is a 4-wheel **mecanum** drive. The go-to-pose loop computes a
body twist, passes it through the mecanum inverse kinematics (twist → wheel speeds) **and back out**
through the forward kinematics every tick, then commands the sim base. The sim base itself is driven
by **kinematic velocity control**: write the planar-joint velocities (`qvel`), then `mj_step`.
**Why:** Routing through IK→FK makes the wheel kinematics genuinely load-bearing (a bug in either
breaks pose-reaching), not decorative — while the FK is the exact inverse of the IK, so the
round-trip is identity and adds no error. Mecanum gives true holonomy (independent x/y/yaw) with a
clean, orthogonal-column kinematic matrix that's easy to verify. Kinematic `qvel` control is exact
and deterministic here because the base has no actuators, joint damping, or gravity component on its
planar DOFs, so there's no tracking error or servo tuning to babysit at this phase.
**Tradeoff:** Kinematic control ignores contact forces, so the base will drive *through* obstacles —
fine for Phase-2 free-space navigation, but real obstacle avoidance / contact-aware motion is out of
scope until (if) needed. Swapping to velocity actuators later is localized to `feathersim/sim/`.

## 2026-06-18 — Skill SDK: a `Robot` facade with logical part handoffs
**Decision:** The SDK is a `Robot` facade over `World` exposing `move_to / pick / place / tend`
(+ `wait_until_done`, read accessors). With no arm in the sim, parts are **logical**: `pick` requires
being parked at a `done` machine and not already holding, then calls the machine FSM's `reset`
(which both unloads the finished part and reloads fresh stock, so the machine resumes cycling);
`place` deposits the carried part onto the output table. Skills enforce preconditions and raise
`SkillError`. Approach poses are derived from `World.fixtures` (ground-truth positions), so the SDK
never touches joints/qpos/MJCF. `move_to`/`place` accept either a fixture name or an explicit pose.
**Why:** The headline deliverable is a clean developer API that hides sim details, not a manipulation
sim. Folding unload+reload into one `reset` matches a real machine-tender swapping parts in a single
visit and keeps `tend` a single trip (go → unload → carry → place). Precondition checks make misuse
loud instead of silently wrong, which the example and tests exercise.
**Tradeoff:** No physical grasp/attachment, so `pick`/`place` can't fail for physical reasons (missed
grasp, collision); they're state transitions gated on pose + machine state. Acceptable for the
autonomy-loop goal; a physical gripper would be a later, sim-local change. The read accessors expose
**ground-truth** machine state for scripting — the Phase-5 autonomy loop must instead consume
perception, keeping ground-truth and perceived state separate (per the Phase-1 deferred note).

## 2026-06-18 — Autonomy loop: oldest-waiting-first scheduling on *perceived* state
**Decision:** `run_autonomy(world, perception, renderer, ...)` selects the next machine to tend purely
from `Perception.perceive` (predictions from rendered pixels) — never `world.states()` / the SDK's
ground-truth accessors. Among machines *perceived* `done`, it services the one that has been waiting
longest (tracked via a per-machine `done_since` timestamp), with a deterministic name tie-break. An
optional `min_confidence` gate drops low-confidence `done` readings before they cost a trip. The loop
is bounded by both `target_parts` and a soft `max_sim_seconds` budget; every iteration makes progress
(a tend advances sim time, otherwise a `wait` advance does).
**Why:** Selecting on perception (not ground truth) is the whole point of the phase — it keeps the
perceived/ground-truth split honest (the Phase-1 deferred note, reinforced in the Phase-3 SDK decision).
*Oldest-first* rather than highest-confidence because a `done` machine has stopped cycling (it's blocked),
so servicing the longest-waiting one is both starvation-free and throughput-maximizing — confidence is
near-constant (~0.98) and a poor scheduling key. The demo surfaced the bug directly: a confidence+name
tie-break starved `machine_0` (0 parts) while machines 1–2 got 3 each; oldest-first gives an even 2/2/2.
**Tradeoff:** Oldest-first needs the loop to remember `done_since` across iterations (small state) and
assumes `done` is terminal until unload (true of the FSM). A persistent all-false-positive perception
would burn the whole `max_sim_seconds` budget delivering nothing — acceptable, and it fails *bounded*
and loud rather than hanging.

## 2026-06-18 — Skill error taxonomy: `PreconditionError` vs. plain `SkillError`
**Decision:** Split `SkillError` into a `PreconditionError(SkillError)` subclass for world/robot-state
precondition violations (not-done, not-parked, already/not-holding) vs. plain `SkillError` for genuine
navigation failures (`could not reach`) and lookups (unknown machine/fixture). The autonomy loop catches
**only** `PreconditionError` (a perception false positive — recover and move on); a nav failure propagates.
**Why:** Catching `SkillError` broadly made a real navigation regression indistinguishable from a benign
perception misread, so an unattended run would silently lose throughput instead of failing loudly. The
subclassing keeps every existing `pytest.raises(SkillError)` valid (Precondition *is* a SkillError).
**Tradeoff:** A little more error surface; worth it for the loud-failure guarantee on the headline loop.

## 2026-06-18 — Dashboard: one sim thread, snapshot-publishing, tick-based preemptible autonomy
**Decision:** The dashboard's `SimManager` owns the `World`/`Robot`/`Perception` and runs the sim on a
**single background thread**. That thread does *all* stepping, perception, and rendering; HTTP handlers
never touch MuJoCo — the thread publishes a telemetry dict + the latest camera JPEG under a lock, and
handlers read those. Autonomy is re-expressed as a **tick-based state machine** (`select → to_machine →
pick → to_table → place`, reusing the pure `velocity_command` + SDK skills) instead of the blocking
Phase-5 `run_autonomy`, so each tick can instead apply an operator twist when the mode is MANUAL —
teleop preempts mid-skill and the SM state survives the interlude (resume = hand back). FastAPI
(`create_app` + lifespan starts/stops the manager); MJPEG `/api/camera`, `/api/telemetry`,
`/api/teleop` (seizes manual), `/api/mode` (resume auto); one static `index.html` with vanilla JS.
**Why:** `MjModel`/`MjData` aren't thread-safe, so confining every sim touch to one thread (handlers read
snapshots) is the simplest correct concurrency model — no locks around the physics, just around the two
published artifacts. A tick SM is the minimal change that makes autonomy *interruptible* at fine grain
(the blocking `tend` could only be preempted at trip boundaries). Selection still consumes only
perception, preserving the perceived/ground-truth split end-to-end into the UI.
**Tradeoff:** The tick SM duplicates the *sequencing* of `Robot.tend` (though not the primitives), so the
tend recipe now lives in two places — acceptable, and the SDK skills/preconditions remain the single
source of truth for each step. GL contexts are thread-affine (macOS), so renderers must be created and
closed *on the sim thread* (in `_run`), not in `__init__` — a real bug that cost a debugging round.
Throughput is reported over total sim time (manual override counts against the rate).

## 2026-06-18 — v2 Phase A: domain randomization in two separable stages; deploy the robust model
**Decision:** Make perception hard via a seeded `DomainRandomizer`, split into (1) a **3D-scene** stage
that mutates the MuJoCo model before each render — randomized worldbody-light position/intensity/color
tint, plus a per-machine non-colliding `occluder_i` box partially blocking the status light — and (2) a
**sensor** stage of pure numpy ops on the rendered crop (additive Gaussian noise + directional motion
blur). The status-light *label color* is never randomized (only its appearance). `make train` now trains
two equally-sized models — `clean` (DR off, v1 recipe) and `robust` (DR on) — evaluates both on a clean
and a randomized held-out set (a 2×2 matrix in committed `metrics.json`), and **deploys the robust model
as `model.pt`** (keeping `model_clean.pt` for comparison). Result: under randomization the clean model
drops 1.0→0.744 while the robust model holds 0.844 (+10 pts), both 1.0 on clean, baseline 0.372.
**Why:** The split keeps each stage independently unit-testable (sensor ops are pure; scene ops need
sim) and mirrors a real pipeline (physical scene vs sensor noise). A real 3D occluder (not a painted
patch) shades and parallaxes correctly — more defensible. Deploying the robust model is free: it scores
1.0 on the clean renders the live loop/dashboard produce (occluders hidden, default light = the MJCF
defaults `reset_scene` restores), so there's no train/serve gap, and it's far better if conditions ever
degrade. The label-color invariant keeps the task honest; bounded occluder size/offset keep every sample
solvable (worst case leaves ~13 of 93 light px visible).
**Tradeoff:** The "robust beats clean" headline is **data-scale-dependent** (it reverses at small n —
see LEARNINGS), so the fast unit suite proves the *mechanism* (DR degrades a clean model) and asserts the
*outcome* from the committed full-scale `metrics.json` rather than retraining in CI. The committed-metrics
test therefore validates a regenerated artifact, not the live pipeline — an accepted trade for suite speed.

## 2026-06-18 — v2 Phase B: A* on an inflated occupancy grid, opt-in on the Robot
**Decision:** Global navigation is A* over a boolean **occupancy grid** of the floor (`planning/`,
pure: `OccupancyGrid`/`build_grid`, `astar`/`plan_path` with octile heuristic, no diagonal
corner-cutting, and greedy line-of-sight smoothing). Obstacles are axis-aligned `Rect`s — machines,
table, and static pillars — **inflated by the robot radius** so a center on a free cell ⇒ body clear of
the real obstacle. A `follow_path` waypoint follower drives the path by reusing the Phase-2
`drive_to_pose` per leg. Planning is **opt-in** on the SDK (`Robot(..., plan=True)` builds the grid once
and routes `move_to` around obstacles); default `plan=False` keeps v1's straight-line behavior, so all
v1 tests/consumers are untouched. The demo runs `World(n_obstacles=2)` + `Robot(plan=True)`.
**Why:** A* on a grid is the standard, transparent, easily-tested global planner; pure functions keep it
unit-testable without sim. Reusing `drive_to_pose` per leg avoids a second controller. Opt-in keeps the
walking skeleton intact and lets the dashboard adopt planning in Phase E (path overlay) rather than now.
**Tradeoff (and the load-bearing subtlety):** inflation guarantees the *planned path* (straight segments,
checked via `segment_free`) clears obstacles, but the P-controlled follower **bows outside those segments
on turns**, so the body can clip even though A* is correct. Closing this took: tightening the follower's
intermediate `waypoint_tolerance` (0.08→0.04), an **extra obstacle-only inflation margin**
(`OBSTACLE_CLEARANCE=0.08`, machines/table stay at radius so tending poses remain reachable), and placing
pillars on the table↔machine diagonals **clear of every tending-pose corridor** (a pillar next to a goal
binds clearance regardless of inflation). Worst-case body clearance over every driven leg is now ~0.29 m
vs the 0.2 m radius, enforced by a test parametrized over all legs. The grid is static (built once); Phase
C will rebuild it per-step for moving robots.

## 2026-06-18 — v2 Phase C: tick-based multi-robot fleet, planning + a symmetric contact backstop
**Decision:** The fleet runs N robots on a **single tick loop** (the blocking `run_autonomy` can't drive
robots sharing one `mj_step`): each tick every robot advances its own SM (`select→to_machine→pick→
to_table→place`) and the world steps once. `World` is now multi-robot (`n_robots`, indexed base methods,
bodies homed at the origin with start poses written into `qpos` so `qpos`==world pos, `driver(k)` →
`_RobotDriver`); `Robot(robot_id=...)` drives its base; `robot_id=0` default keeps v1/v2A/v2B intact.
**Task allocation** (`FleetManager`) locks each machine to one robot — never double-booked (the tick loop
is sequential, so a lock commits before the next robot runs); released at *pick* (the part is unloaded,
the machine free to re-tend). **Scheduling** is pluggable (`longest_waiting`, `nearest_done`).
**Collision avoidance** is two layers: every robot **plans around all others** (inflated obstacles,
periodic replan via the Phase-B grid with `extra_obstacles`), plus a **symmetric predictive backstop** —
a robot stops if its predicted next step would land within a body-clearance of *any* other robot.
**Per-robot perception**: each robot reads the machine cameras through its own `corrupt_image` RNG
(Phase-A synergy), so reads can genuinely disagree.
**Why:** A tick loop is the only thing that composes for a shared world. Planning-around-others keeps
robots apart proactively; the symmetric backstop is the *guarantee* — verified collision-free
(min separation ≥ 2·radius) over 160 independent runs (40 seeds × 2 configs × 2 strategies), worst
0.432 m vs the 0.40 m body clearance. It's symmetric (not priority-only) because a never-yielding leader
*rear-ends* a yielded follower — the bug that made an earlier priority scheme collide on ~half of seeds.
**Tradeoff:** Dropping strict priority loses the structural no-deadlock guarantee, so two robots in a
*tight* cell (3 robots + static pillars, or 2 robots + pillars on ~13% of seeds) can wedge until
`max_sim_seconds` — bounded and surfaced as `FleetReport.completed == False`; the demo therefore uses
3 robots on an **open floor** (robot↔robot coordination, collision-free on every seed tested). Throughput
is logged per strategy but the two **tie** under load: when robots are saturated, total throughput is
robot-limited regardless of which done machine is chosen — the strategies differ in wait/travel, not rate.

## 2026-06-18 — v2 Phase D: behavior-clone the go-to-pose controller, drop-in via a `velocity_fn` seam
**Decision:** Replace the hand-coded controller with a learned one by **behavior cloning**. The expert is
the pure `velocity_command` P-controller; its output depends *only* on the goal in the body frame
(`goal_in_body_frame`, extracted and shared), which makes the control law **memoryless** — so the BC
dataset is generated by *sampling* random body-frame goals and recording the expert twist (no rollout
needed; a sampled goal *is* a valid observation, so there's no covariate shift in the data). A tiny MLP
(3→128→128→3, tanh head) is trained with MSE on normalized actions. It deploys as a drop-in **`velocity_fn`**:
`drive_to_pose` and `follow_path` take a `velocity_fn` (default = `velocity_command`), and `Robot(controller=)`
threads it through *both* the straight-line and planned branches, so the whole autonomy loop — perception →
plan → drive → tend — runs on the learned brain. Result: 100% goal reach (= expert) in 208 steps vs 224;
full loop delivers 6 parts at 112% of the expert's throughput. `make policy`; CPU-only (seconds).
**Why:** A `velocity_fn` seam is the minimal, honest integration — the policy is genuinely the controller
for every drive, not a side demo. Sampling instead of rolling out is correct *because* the law is
memoryless; calling that out keeps the BC honest (the hard part here is closed-loop stability, not the
regression). Scaling `ACTION_SCALE` from `PoseGains` (not magic numbers) keeps dataset and policy caps
coupled.
**Tradeoff:** This is a *soft* BC problem (smooth, low-dim, single-step-invertible map), so "matches the
expert" is a modest bar — the value is the end-to-end pipeline (expert-data → BC → closed-loop deploy →
comparison), per the project's "headline is the pipeline, not SOTA" stance. The policy is tied to its
training-time velocity caps (the `gains` arg on `PolicyController` is signature-only, documented). The
tanh head can't exactly hit the controller's *saturated* ±max outputs (≈80% of sampled targets saturate),
a benign under-shoot that makes the policy marginally faster; noted in LEARNINGS. An obstacle-aware
navigation policy (richer obs, learns avoidance end-to-end) was offered as a heavier alternative and
deferred.

## 2026-06-19 — v2 Phase E: command center on a shared `FleetController`, with a 2D schematic + live controls
**Decision:** The capstone dashboard reuses the Phase-C fleet by **extracting a `FleetController`** (one
`step()` = advance all robots + world once, exposing live `phase`/`target`/`path`/`last_readings`/
`assignments`); `run_fleet` becomes a thin loop over it. A `FleetSimManager` runs the controller on one
background thread (all stepping/rendering there; HTTP reads a published snapshot + JPEG under a lock —
the Phase-6 model) and publishes a **top-down PIL schematic** (no GL) with each robot's planned path
overlaid, plus telemetry (per-robot phase/target, per-machine true *and* perceived state, assignments).
Two live controls: a **controller toggle** that swaps `FleetController.velocity_fn` (hand-coded ↔ learned
policy), and a **perception-difficulty slider** that scales DR via `DomainRandomizer.at_difficulty(d)` and
shows **both** the deployed robust model and a clean baseline degrading — the robust holds, the clean
crumbles (clean 1.0→0.79, robust 0.91–1.0 across the slider). `make dashboard` = command center;
`make teleop` = the Phase-6 single-robot dashboard.
**Why:** Extracting `FleetController` avoids duplicating the fleet SM (the alternative the Phase-6 dash
took, which the reviewer would flag), and makes the same tested engine drive both the headless runner and
the live dashboard. A 2D schematic (vs a 3D render) is the right command-center view — it shows paths,
assignments, and every robot at once clearly, and needs no GL. Showing robust *vs* clean accuracy live
turns Phase A's static result into the demo's most compelling interactive moment.
**Tradeoff:** The schematic is a top-down abstraction, not the photoreal 3D feed (fine — clarity > realism
for a fleet view). The cross-thread control writes (`velocity_fn`, `difficulty`) are single atomic ops,
intentionally lock-free under the GIL (documented). The slider's `at_difficulty` scales probabilities +
dominant magnitudes but keeps occluder/blur extents fixed (presence is the dominant lever).

## 2026-06-18 — Three project subagents for the engineering loop
**Decision:** Add `test-runner` (haiku), `reviewer` (sonnet), `docs-researcher` (sonnet) in
`.claude/agents/`. The per-phase loop delegates testing and end-of-phase review to them.
**Why:** Context hygiene (keep verbose pytest output and API lookups out of the main thread) and
a fresh-perspective review pass that drives elegance. The main thread keeps ownership of
architecture and implementation.
**Tradeoff:** Slight coordination overhead and a required session restart to load the agents;
worth it for cleaner context and an independent review gate.
