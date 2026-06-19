# FeatherSim

A machine-tending autonomy stack **in simulation**, inspired by Feather Robotics: holonomic wheeled
mobile robots autonomously tend several CNC-style machines. It ships a developer **skill SDK**, a
**perception** model trained on **auto-labeled** sim data (hardened with **domain randomization**),
**A\* path planning**, an unattended **single- and multi-robot autonomy** stack, a **behavior-cloned**
controller, and a browser **command center**. Pure simulation — no real hardware.

![The multi-robot fleet, rendered live](docs/fleet.gif)

*Three robots — each with an **arm + gripper** — tend three machines unattended and collision-free
(status-light dome = machine state). Each perceives from its own camera, a fleet manager assigns work
without double-booking, A\* plans each path, and parts **physically ride a gripper from machine to a
growing stack on the table**. The command center streams this **live 3D view** plus a tactical top-down
with paths overlaid — with a hand-coded↔learned controller toggle and a perception-difficulty slider — at
`make dashboard`.*

## Quickstart

```bash
make install     # pip install -r requirements.txt  (Python 3.11+)
make test        # pytest — 162 tests
make demo        # headless single-robot autonomy loop (routes around obstacles)
make fleet       # headless 3-robot fleet, compares scheduling strategies, collision-free
make dashboard   # the command center at http://localhost:8000  (fleet + paths + toggle + slider)
make teleop      # single-robot dashboard with WASD override
make train       # retrain perception (clean + domain-randomized models)
make policy      # behavior-clone the controller into a learned policy
```

## The stack, layer by layer

A walking skeleton built in vertical slices — runnable and demoable at every phase:

| Layer | Package | What it does |
|---|---|---|
| **Sim** | `feathersim/sim/` | Headless MuJoCo world (1–3 robots, 2–3 machines, table, obstacles) + a pure timer FSM `idle→running→done`. Deterministic per seed. |
| **Kinematics** | `feathersim/kinematics/` | Holonomic **mecanum** drive math — pure inverse/forward kinematics, no sim import. |
| **Control** | `feathersim/control/` | Go-to-pose P-controller; the body twist is routed through the wheel IK→FK each tick. Pluggable `velocity_fn`. |
| **Skill SDK** | `feathersim/sdk/` | A `Robot` facade hiding joints/MJCF/kinematics: `move_to / pick / place / tend`. Preconditions raise `SkillError`. |
| **Perception** | `feathersim/perception/` | Renders per-machine crops, **auto-labels** from ground-truth configs, trains a 2-head CNN. With **domain randomization**: robust **84.4%** under DR vs a clean model's **74.4%** (both 1.0 clean; 0.39 baseline). |
| **Planning** | `feathersim/planning/` | Occupancy grid + 8-connected **A\*** + line-of-sight smoothing + a waypoint follower. Routes around obstacles. |
| **Autonomy** | `feathersim/autonomy/` | The single-robot loop: perceive → tend the longest-waiting perceived-`done` machine → repeat. Selects on **perception, never ground truth**. |
| **Fleet** | `feathersim/fleet/` | Multi-robot tick engine: task allocation (no double-booking), **symmetric collision avoidance**, pluggable scheduling. |
| **Policy** | `feathersim/policy/` | **Behavior-cloned** go-to-pose controller; a tiny MLP drop-in for the P-controller — runs the whole loop at 112% of the expert's throughput. |
| **Dashboard** | `feathersim/dashboard/` | FastAPI + single-file vanilla JS: the multi-robot **command center** and the single-robot **WASD-teleop** dashboard. |

## v2 — five hard systems on top of the walking skeleton

- **Brutal perception.** Domain randomization (randomized lighting, status-light occluders, sensor noise,
  motion blur) makes perception hard; a robust model **holds where the clean one crumbles**.
- **Path planning.** A* on an occupancy grid + a waypoint follower; the robot routes around static
  pillars, body clearance proven on every leg it drives.
- **Multi-robot fleet.** 2–3 robots, a manager that assigns `done` machines without double-booking, a
  **symmetric contact backstop** that keeps bodies ≥ 2·radius apart (verified collision-free over 160
  seeded runs), and ≥2 scheduling strategies measured for throughput.
- **Learned policy.** Behavior cloning of the hand-coded controller; the learned brain drives the entire
  autonomy loop end-to-end. *(Offline loss didn't predict closed-loop success — a lesson logged.)*
- **Command center.** This dashboard: a **live cinematic 3D feed** of the cell + a tactical top-down with
  planned paths overlaid, per-robot perceived-vs-true state, live task assignments, a **hand-coded ↔
  learned** controller toggle, and a **perception-difficulty slider** that dials domain randomization
  up/down live — drag it and watch the clean model's accuracy drop while the robust model holds.

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
*center*, the follower bows), and a collision guarantee that held only on the lucky seed 0. Both are
written up in [`LEARNINGS.md`](LEARNINGS.md). **162 tests**; rendering-dependent tests skip without a GL
backend (`MUJOCO_GL=egl`/`osmesa` to run them in CI).

## Docs

- [`PLAN.md`](PLAN.md) — phased roadmap + acceptance criteria (v1 phases 0–6, v2 phases A–E)
- [`DECISIONS.md`](DECISIONS.md) — architecture decision log (why)
- [`LEARNINGS.md`](LEARNINGS.md) — sim/training gotchas, never hit twice
- [`CLAUDE.md`](CLAUDE.md) — conventions + the engineering loop

## Regenerating the GIFs

```bash
python3 scripts/record_fleet_gif.py --out docs/fleet.gif     # the command-center fleet view
python3 scripts/record_gif.py --parts 3 --out docs/autonomy.gif  # the single-robot loop
```
