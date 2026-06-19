# FeatherSim

A machine-tending autonomy stack **in simulation**, inspired by Feather Robotics: holonomic wheeled
mobile robots autonomously tend several CNC-style machines. It ships a developer **skill SDK**, a
**perception** model trained on **auto-labeled** sim data (hardened with **domain randomization**),
**A\* path planning**, an unattended **single- and multi-robot autonomy** stack, a **behavior-cloned**
controller, and a browser **command center**. Pure simulation — no real hardware.

![The multi-robot fleet, rendered live](docs/fleet.gif)

*Four robots — each with an **arm + gripper** — tend four machines unattended and collision-free
(status-light dome = machine state). Each perceives from its own camera, a fleet manager assigns work
without double-booking, A\* plans each path, a **priority-yield deadlock breaker** keeps the busy floor
live, and parts **physically ride a gripper from machine to a growing stack on the table**. The command
center streams this **live 3D view**, each robot's **onboard camera**, and a tactical top-down with paths
overlaid — with a hand-coded↔learned controller toggle and a perception-difficulty slider — at
`make dashboard`.*

## Quickstart

```bash
make install     # pip install -r requirements.txt  (Python 3.11+)
make test        # pytest — 176 tests
make demo        # headless single-robot autonomy loop (routes around obstacles)
make fleet       # headless 4-robot fleet, compares scheduling strategies, collision-free
make dashboard   # the command center at http://localhost:8000  (3D feed + onboard cams + paths + toggle + slider)
make teleop      # single-robot dashboard with WASD override
make train       # retrain perception (clean + domain-randomized models)
make policy      # behavior-clone the controller into a learned policy
```

## The stack, layer by layer

A walking skeleton built in vertical slices — runnable and demoable at every phase:

| Layer | Package | What it does |
|---|---|---|
| **Sim** | `feathersim/sim/` | MuJoCo world (1–4 robots with **arm + gripper** and an onboard camera, 1–4 machines, table, obstacles) + a pure timer FSM `idle→running→done`. Cinematic materials/shadows for the live feed; deterministic per seed. |
| **Kinematics** | `feathersim/kinematics/` | Holonomic **mecanum** drive math — pure inverse/forward kinematics, no sim import. |
| **Control** | `feathersim/control/` | Go-to-pose P-controller; the body twist is routed through the wheel IK→FK each tick. Pluggable `velocity_fn`. |
| **Skill SDK** | `feathersim/sdk/` | A `Robot` facade hiding joints/MJCF/kinematics: `move_to / pick / place / tend`. Preconditions raise `SkillError`. |
| **Perception** | `feathersim/perception/` | Renders per-machine crops, **auto-labels** from ground-truth configs, trains a 2-head CNN. The deployed model trains on a **clean+randomized mix**: **1.0 clean *and* 0.91 under domain randomization** (+19 pts over a clean-only model's 0.71; 0.37 baseline). |
| **Planning** | `feathersim/planning/` | Occupancy grid + 8-connected **A\*** + line-of-sight smoothing + a waypoint follower. Routes around obstacles. |
| **Autonomy** | `feathersim/autonomy/` | The single-robot loop: perceive → tend the longest-waiting perceived-`done` machine → repeat. Selects on **perception, never ground truth**. |
| **Fleet** | `feathersim/fleet/` | Multi-robot tick engine (up to 4): task allocation (no double-booking), **symmetric collision avoidance** + a **priority-yield deadlock breaker**, pluggable scheduling. |
| **Policy** | `feathersim/policy/` | **Behavior-cloned** go-to-pose controller; a tiny MLP drop-in for the P-controller — runs the whole loop at 112% of the expert's throughput. |
| **Dashboard** | `feathersim/dashboard/` | FastAPI + single-file vanilla JS: the multi-robot **command center** (live cinematic 3D feed, per-robot onboard cameras, tactical top-down) and the single-robot **WASD-teleop** dashboard. |

## v2 — five hard systems on top of the walking skeleton

- **Brutal perception.** Domain randomization (randomized lighting, status-light occluders, sensor noise,
  motion blur) makes perception hard; a robust model **holds where the clean one crumbles**.
- **Path planning.** A* on an occupancy grid + a waypoint follower; the robot routes around static
  pillars, body clearance proven on every leg it drives.
- **Multi-robot fleet.** Up to 4 robots, a manager that assigns `done` machines without double-booking, a
  **symmetric contact backstop** + a **stuck-triggered priority-yield deadlock breaker** that keep bodies
  ≥ 2·radius apart *and* the busy floor live (verified collision-free + every-part-delivered over 40 seeded
  4×4/3×3/3×2 runs), and ≥2 scheduling strategies measured for throughput.
- **Learned policy.** Behavior cloning of the hand-coded controller; the learned brain drives the entire
  autonomy loop end-to-end. *(Offline loss didn't predict closed-loop success — a lesson logged.)*
- **Command center.** This dashboard: a **live cinematic 3D feed** of the cell + a tactical top-down with
  planned paths overlaid, per-robot perceived-vs-true state, live task assignments, a **hand-coded ↔
  learned** controller toggle, and a **perception-difficulty slider** that dials domain randomization
  up/down live — drag it and watch the clean model's accuracy drop while the robust model holds.

## v3 — make it look and behave like a real Feather sim

Five polish iterations layered on top, each through the full engineering loop (reviewer SHIP on all five):

1. **Cinematic world + live 3D feed.** Glossy materials, gradient skybox, reflective floor, shadows; the
   command center streams a live 3D overview. Lighting domain randomization is now *relative* to the
   authored key light (the feed stays correctly lit, with no train/serve gap), and the robust model is
   **mix-trained** → robust *without* sacrificing clean accuracy.
2. **Manipulator arms + physical part transport.** Every robot gets an arm + gripper; parts **ride the
   gripper** from machine to a **growing stack on the output table**.
3. **Animated reach / grasp / retract.** The arm **swings into the machine to grasp** and **extends over
   the table to place** — a kinematic, `gravcomp`-decoupled joint that never perturbs the base.
4. **Onboard cameras.** Each robot's **eye-view** — the machine it's tending, its own gripper, the part —
   composited into a live strip in the command center.
5. **A busier 4×4 floor + deadlock-free coordination.** Four robots tending four machines; a
   **stuck-triggered priority-yield deadlock breaker** keeps the contended floor live where the bare
   symmetric backstop would freeze — verified collision-free *and* every-part-delivered across 40 seeds.

## How the loop closes

```
   camera frame ─▶ Perception.perceive ─▶ which machine is *perceived* done?
                                                    │
                                                    ▼
        Robot.tend  ◀──  assign longest-waiting (fleet: no double-booking)
        │  move_to(machine, plan=A*) → pick → move_to(table) → place
        ▼
   part delivered ─▶ throughput / uptime ─▶ (repeat, unattended, collision-free)
```

Selection consumes only the perception model's predictions — the perceived-vs-ground-truth split is kept
honest end to end (a test makes perception *disagree* with the sim and asserts the loop follows
perception). A false positive surfaces as a `PreconditionError` and is shrugged off; a real navigation
failure fails loudly.

## Engineering rigor

Every phase ran the same loop: smallest vertical slice → `test-runner` (nothing advances on red) →
independent `reviewer` (address CRITICAL/HIGH before commit) → log `DECISIONS`/`LEARNINGS` → commit. The
reviewer caught bugs a green suite hid — a robot body silently clipping a pillar (planning protects the
*center*, the follower bows), a collision guarantee that held only on the lucky seed 0, and a 4-robot
deadlock the bare backstop masked. All are written up in [`LEARNINGS.md`](LEARNINGS.md). **176 tests**;
rendering-dependent tests skip without a GL backend (`MUJOCO_GL=egl`/`osmesa` to run them in CI).

## Docs

- [`PLAN.md`](PLAN.md) — phased roadmap + acceptance criteria (v1 phases 0–6, v2 phases A–E, v3 iterations 1–5)
- [`DECISIONS.md`](DECISIONS.md) — architecture decision log (why)
- [`LEARNINGS.md`](LEARNINGS.md) — sim/training gotchas, never hit twice
- [`CLAUDE.md`](CLAUDE.md) — conventions + the engineering loop

## Regenerating the GIFs

```bash
python3 scripts/record_fleet_gif.py --out docs/fleet.gif     # the command-center fleet view
python3 scripts/record_gif.py --parts 3 --out docs/autonomy.gif  # the single-robot loop
```
