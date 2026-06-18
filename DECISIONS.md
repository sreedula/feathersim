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

## 2026-06-18 — Three project subagents for the engineering loop
**Decision:** Add `test-runner` (haiku), `reviewer` (sonnet), `docs-researcher` (sonnet) in
`.claude/agents/`. The per-phase loop delegates testing and end-of-phase review to them.
**Why:** Context hygiene (keep verbose pytest output and API lookups out of the main thread) and
a fresh-perspective review pass that drives elegance. The main thread keeps ownership of
architecture and implementation.
**Tradeoff:** Slight coordination overhead and a required session restart to load the agents;
worth it for cleaner context and an independent review gate.
