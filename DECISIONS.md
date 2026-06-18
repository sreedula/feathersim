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

## 2026-06-18 — Three project subagents for the engineering loop
**Decision:** Add `test-runner` (haiku), `reviewer` (sonnet), `docs-researcher` (sonnet) in
`.claude/agents/`. The per-phase loop delegates testing and end-of-phase review to them.
**Why:** Context hygiene (keep verbose pytest output and API lookups out of the main thread) and
a fresh-perspective review pass that drives elegance. The main thread keeps ownership of
architecture and implementation.
**Tradeoff:** Slight coordination overhead and a required session restart to load the agents;
worth it for cleaner context and an independent review gate.
