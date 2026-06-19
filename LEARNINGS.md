# FeatherSim — LEARNINGS.md

Append a dated entry whenever something surprises you (sim quirks, training failures, flaky
tests). This is the compounding mechanism: never get bitten by the same issue twice.

## 2026-06-18 — Dev environment baseline
Python 3.13.5 (anaconda), pip 25.1, pytest 8.3.4, git 2.50.1 on macOS (darwin 24.6).

## 2026-06-18 — PyBullet won't build here → switched sim to MuJoCo [RESOLVED]
The Phase-0 worry was real. `pip install pybullet` fails on this box (macOS, Apple clang 17,
framework Python 3.13): PyBullet ships no macOS wheels, and the source build dies compiling its
vendored zlib — the old zlib headers collide with the new macOS SDK `_stdio.h`, throwing
`error: expected identifier or '('` / `expected ')'` expanded from the `NULL` macro (with `ZEXPORT`
fragments in the trace). Not a Python-version issue — there's no wheel on Mac at any version, so a
venv wouldn't help; only conda-forge (env fragmentation) or a C patch (non-reproducible) would.
**Resolution:** switched the backend to MuJoCo (`pip install mujoco`, official prebuilt wheel,
instant). Verified: imports, steps headless deterministically, and `mujoco.Renderer` returns
offscreen RGB frames. See DECISIONS.md.

## 2026-06-18 — MuJoCo headless render needs a GL context, but it worked out of the box here
`mujoco.Renderer(model, h, w).render()` returned a real `(h, w, 3)` uint8 frame with no extra setup
on this macOS box. On headless Linux/CI you may need `MUJOCO_GL=egl` (or `osmesa`) in the env, since
the default GLFW backend wants a display. Note this if camera rendering (Phase 4) ever fails in CI.

## 2026-06-18 — pytest import path
Tests import the top-level `feathersim` package. Set `pythonpath = ["."]` in
`[tool.pytest.ini_options]` so `pytest` from repo root puts the root on `sys.path` (avoids
`ModuleNotFoundError` without needing an editable install or a root `conftest.py`).

## 2026-06-18 — Perception gotchas (Phase 4): four things that bit me
Building the auto-labeling + CNN pipeline surfaced four non-obvious failures, each fixed:
1. **Global avg-pool washed out the small status light** → the state head collapsed to a constant
   (acc 0.31, *below* the 0.38 baseline) while the larger bed-part head learned fine. Fix: a global
   **max-pool** branch (concat with avg) preserves the localized light signal; also enlarged the light.
   Lesson: for a small, localized cue, average pooling is the wrong inductive bias — use max.
2. **No contrast = no signal.** The first bed part was whitish (0.80) against a whitish floor (0.82);
   part acc stalled ~0.75. A vivid blue part (high contrast vs floor, dark door, and machine bodies)
   jumped it to 1.0. Design the visual cue for contrast against *everything* it can sit in front of.
3. **Train/serve mismatch.** Training rendered neighbors neutral but live `perceive`/`sync_visuals`
   lights *all* machines; edge machines whose camera grazes a lit neighbor then read at ~chance
   (conf 0.34). Fix: light every machine with an independent random config in the dataset and label
   the centered target — train distribution now matches serving.
4. **Visual-cue geoms are still collision geoms by default.** The new part geom protruded in front of
   the machine and physically blocked the robot's tending approach (SDK drive stalled ~0.16 m short).
   Fix: `contype="0" conaffinity="0"` on the light/part geoms — they're perception cues, not obstacles.

## 2026-06-18 — `python` ≠ `python3` here: deps live under `python3` only
On this machine `python` → conda base (`/opt/anaconda3`, no project deps) while `python3` → the
python.org framework build (`/Library/Frameworks/Python.framework/Versions/3.13`) where
`pip install` actually put mujoco/torch/numpy. Running `python scripts/print_state.py` died with
`ModuleNotFoundError: No module named 'mujoco'`; `python3 ...` works. The Makefile already uses
`python3` for every target — keep it that way and always invoke `python3`/`make`, never bare
`python`, for this project.

## 2026-06-18 — The demo caught a scheduler-fairness bug the unit tests didn't (Phase 5)
Running `make demo` on the real model delivered 6 parts but `machine_0: 0, machine_1: 3, machine_2: 3` —
one machine starved. Diagnosis: perception was *not* at fault (a quick probe with the robot parked at the
table read all three machines' `done` state with 100% agreement, 0 misreads). The cause was the loop's
tie-break: sorting perceived-done candidates by `(confidence, name)` descending, with confidence ~constant
at 0.98, made the order purely name-descending, so `machine_0` lost every tie and never got serviced within
the 6-part target. Fix: schedule **oldest-waiting-first** (track when each machine was first seen done) →
even `2/2/2`. Lessons: (1) a green unit suite doesn't prove the *emergent* behavior is right — always run
the actual demo and eyeball the distribution, not just the total; (2) when one component looks guilty
(perception), cheaply falsify that hypothesis before "fixing" it — the bug was in the scheduler.

## 2026-06-18 — MuJoCo GL contexts are thread-affine: create the renderer on the thread that renders (Phase 6)
The dashboard runs the sim on a background thread. I first built the `mujoco.Renderer`s in
`SimManager.__init__` (the main/caller thread) but called `.render()` from the sim thread — and
`manager.frame()` stayed `None` forever (the daemon thread silently died). On macOS the renderer's GL
context is bound to the thread that *created* it; rendering from another thread fails. Fix: create the
renderers at the top of `_run()` (on the sim thread) and close them there in a `finally` — never in
`__init__`/`stop()`. Corollary: a wedged sim thread can't have its renderers closed from `stop()` either
(same affinity), so `stop()` just warns if the join times out. On headless Linux/CI the same code needs
`MUJOCO_GL=egl`. This is the Phase-4 "headless render needs a GL context" note, now with a threading twist.

## 2026-06-18 — An endless MJPEG generator hangs FastAPI's TestClient (Phase 6)
Consuming the `multipart/x-mixed-replace` camera stream through `TestClient.stream(...)` and `break`-ing
out hung the whole test run (had to SIGKILL; exit 144). The server-side generator is `while True: … sleep`,
and the test transport doesn't simulate a browser disconnect, so it never stops pulling / tearing down.
Two fixes: (1) test the generator function directly (`next(gen)`; `gen.close()`) and the manager's
`frame()` rather than driving the infinite stream through TestClient; (2) make the generator loop on
`manager.is_running()` so real shutdown ends it cleanly (in production Starlette stops pulling it on
disconnect, but don't rely on that for the stop path). Lesson: never drive an unbounded streaming
response through TestClient — exercise the generator in isolation.

## 2026-06-18 — Domain randomization needs *data scale* to pay off — it backfires when small (v2 Phase A)
Building the robust-perception comparison, the unit-scale check (`n=300`) showed the **opposite** of the
full-scale result: the DR-trained model got **0.61** on the randomized set vs the clean model's **0.72** —
robust *lost*. At full scale (`n=720`) it flips to 0.844 vs 0.744 (+10 pts). With little data the DR model
underfits the much harder (occlusion + noise + blur) distribution while the clean model coasts on the easy
fraction of randomized samples. Lesson: don't assert a scale-sensitive ML *outcome* in a fast unit test —
it'll be flaky-by-construction. Instead the suite proves the **mechanism** that always holds (a clean model
*degrades* under randomization) with one small training run, and locks the **headline** (robust > clean) by
asserting the committed full-scale `metrics.json`. The real proof is the regenerated artifact + `make train`.

## 2026-06-18 — A 3D occluder's *projected* shadow is bigger than its physical size (v2 Phase A)
The status-light occluder is a small box placed ~0.12 m *in front* of the light (toward the camera), so on
the image it covers more than its half-extent would suggest — a box smaller than the light sphere can still
shadow most of it. My first justification ("half-extent < light radius ⇒ never fully hidden") was wrong
reasoning even though the bound happened to hold. The "never fully hidden / always labelable" invariant is
**empirical** (tuned so worst case leaves ~13 of 93 light px visible), not derivable from the half-extent.
Lesson: for anything between the camera and the cue, reason in *image/projected* space, not object space.

## 2026-06-18 — Grid inflation protects the *center*, but a P-controlled follower bows outside the corridor (v2 Phase B)
A* on a radius-inflated occupancy grid guarantees the robot **center** stays clear of obstacles, and
`segment_free` validates the straight segments between waypoints. But the waypoint follower drives a
*proportional curve* toward each waypoint and rounds corners (switching to the next leg within
`waypoint_tolerance`), so the actual trajectory **bows outside the validated segments** on turns. Result:
the body clipped a non-colliding pillar by 3 cm on the machine→table legs even though every planned cell
was free — and because the pillars are `contype=0`, there was no contact, no error, **no test failure**.
Two-part lesson: (1) inflation-by-radius is necessary but not sufficient when the follower can leave the
corridor — add a margin for the bow (tighter `waypoint_tolerance` + extra obstacle inflation), and reason
about the *driven* trajectory, not just the planned path; (2) the bug hid because the clearance test
sampled **one** leg (table→machine_2, the easy direction) — the loop also drives machine→table, which
bows worst. Fix: parametrize the safety test over **every** leg the system can drive. A green suite that
samples one favorable case is worse than no test — it certifies a property that doesn't hold.

## 2026-06-18 — An obstacle next to the goal binds clearance regardless of inflation (v2 Phase B)
First obstacle placement (0.55, 0.35) sat right beside machine_2's tending-pose corridor (x≈1.0), so the
robot's *final approach to the goal* grazed the pillar — and no amount of grid inflation helped, because
the robot must reach the fixed goal pose. Moving the pillars onto the table↔machine diagonals, clear of
every tending pose, made clearance governed by inflation (tunable) instead of by goal geometry (fixed).
Also: pillars too close together (±0.5) pinch the narrow central corridor shut at grid resolution,
trapping the start and making machine_1 unreachable. Placement is a real constraint, not cosmetic — sweep it.

## 2026-06-18 — Slide-joint qpos is relative to the body's home, not world (multi-robot, v2 Phase C)
Making the world multi-robot, I first placed each robot body at its start position in the MJCF
(`<body pos="sx sy 0.15">`) with planar slide joints. `robot_pose` reads `qpos` — which is the joint
*displacement from the body home* — so every robot reported (0,0) regardless of where it actually was
(in v1 the single body was at the origin, so qpos==world by luck). Two robots overlapped at distance 0.
Fix: home *all* robot bodies at the origin and write start positions into `qpos` in `__post_init__`, so
`qpos` is the true world position everywhere. Lesson: with free/slide joints, qpos is frame-relative —
don't conflate it with world coordinates unless the body frame is the world frame.

## 2026-06-18 — A green multi-robot test certified a collision-freedom that held only on seed 0 (v2 Phase C)
The collision-avoidance test parametrized over robot/obstacle *counts* but pinned `seed=0`. It passed —
but a seed sweep (review) showed bodies overlapped on ~half of all seeds; `seed=0` just landed in the
safe basin. Same trap as Phase B (sampling one favorable case). Two compounding lessons: (1) for a
*stochastic* safety property, parametrize the test over many seeds, not one — a single-seed safety test
is theatre. (2) The avoidance bug itself: a strict-priority scheme where the top robot never yields lets
it **rear-end** a lower robot that yielded into its (stale) path. Strict priority prevents *deadlock* but
not *collision* — collision avoidance must be **symmetric** at the contact layer even if task priority is
asymmetric. Fix: a symmetric predictive backstop (stop if the next step lands within a body-clearance of
*any* other robot), verified over 160 runs. It costs the structural no-deadlock guarantee (now bounded by
a time budget + surfaced via `completed`), an acceptable trade for an actual safety guarantee.

## 2026-06-18 — Under saturation, scheduling strategy barely moves *throughput* (v2 Phase C)
Expected `longest_waiting` vs `nearest_done` to show a throughput gap; they tied (≤1% apart). Reason:
when robots are kept busy (a backlog of done machines), total throughput is **robot-limited** — the robot
is always tending *something*, so which machine it picks doesn't change the rate, only the average wait
and travel. Throughput is the wrong discriminator for a scheduler under load; latency/fairness is where
the strategies differ. Worth measuring, but don't expect the headline metric to separate them.

## 2026-06-18 — Offline BC val-loss does NOT predict closed-loop success (v2 Phase D)
Behavior-cloning the controller, the validation MSE was ~constant at **1.5e-3** across dataset sizes —
but closed-loop goal-reach went **0.50 → 1.00** between 8k and 12k samples at the *same* val loss. Small
per-step action errors that the offline metric barely registers **compound over a trajectory** (the robot
drifts, visits states slightly off the training distribution, errs more). Lesson: never gate a BC policy
on offline loss — evaluate it **in closed loop** (reach rate / steps), which is the signal that actually
moves. The test fixture trains a policy big enough to drive (12k/40), and the suite asserts the closed-loop
*steps* bound, not val MSE. Mirrors the Phase-A "scale matters" lesson but for the *metric*, not the data.

## 2026-06-18 — A tanh head can't reach a saturated target (v2 Phase D)
The expert P-controller clamps to ±max speed, and with uniform goal sampling ~80% of targets sit *at* the
clamp. With `ACTION_SCALE` = the caps, those targets land at ±1.0 — the tanh asymptote the head can only
*approach*. So imitation error concentrates entirely in the saturated regime (a gentle under-shoot near
±max), while the unsaturated map is learned almost exactly. Benign here (the under-shoot makes the policy
slightly faster, not unstable) and the test asserts *mean* imitation error + a loose worst-case bound
rather than tight per-point equality. If exact saturation mattered, widen `ACTION_SCALE` a hair above the
caps so true ±max sits at ±0.95 (reachable). Lesson: don't put a bounded-asymptote head's target *on* its
asymptote.

## 2026-06-19 — The difficulty slider didn't bite until I applied the *scene* stage, not just the sensor (v2 Phase E)
The live difficulty slider initially moved nothing — both models read ~100% at full difficulty. The bug:
the dashboard's `_perceive` applied only `randomizer.corrupt_image` (the **sensor** stage: noise + blur),
not the **scene** stage (`apply_scene`: randomized lighting + the status-light **occluder**). But the
occluder — which physically blocks the cue the model reads — is the *dominant* difficulty lever; noise/blur
alone barely dent a big bright status light. Applying the full pipeline (sample_scene → apply_scene →
render → corrupt_image) made accuracy react sharply (clean 1.0→0.79). Lesson: when a DR "intensity" knob
doesn't move your metric, check you're applying the part of the pipeline that carries the signal — here the
hard corruption was 3D-scene-side, not pixel-side, and I'd wired up only the cheap half.

## 2026-06-19 — Train robust on a clean+randomized MIX, not only-randomized (v3)
Hardening perception with the cinematic visuals, training the robust model on *only* domain-randomized
data plateaued it at ~0.9 on clean (it traded ~10% clean accuracy for robustness — the tiny CNN can't be
perfectly both). Pushing the DR harder only widened the slider gap by *lowering* the deployed model's
clean accuracy — wrong tradeoff. Fix: train the robust model on a **50/50 clean+randomized mix** → 1.0 on
clean *and* 0.91 under DR (+19 pts over the clean model). Lesson: don't make the deployed model robust by
making it worse at the easy case; mix easy + hard so it's good at both. Tune DR for the *gap*, not by
sacrificing clean.

## 2026-06-19 — A high-res offscreen render needs `<global offwidth/offheight>` (v3)
Rendering the 560px 3D feed failed silently until I added `<visual><global offwidth/offheight 1280>` —
MuJoCo's default offscreen framebuffer is 640×480, so any `mujoco.Renderer(model, h, w)` with h or w > 480
errors with a message pointing at `<global offheight>`. Set the global once (≥ the largest renderer) and
all renderers (32/64/560) share it. Cheap (~one 1280² buffer per GL context).

## 2026-06-19 — A symmetric collision backstop deadlocks at 4 robots; break it with stuck-triggered priority yield (v3)
Scaling the fleet 3→4 robots, the symmetric contact backstop (each robot stops if its predicted step lands
too close to another) **deadlocked**: clusters of robots all freeze and never recover (each waits for the
others). It worked at 3 robots only by having less contention — a latent fragility, not a guarantee.
Widening the table/cell for 4 robots exposed it even at 3 robots. Fix that *kept* the proven behavior:
a **stuck-triggered priority yield** — a robot only yields after being backstop-blocked for `_STUCK_TIME`
(so normal passing is untouched), and yields by backing *away* from any *higher-priority* (lower-id) robot.
Strict id-priority = a total order ⇒ no yield cycles ⇒ the lowest-id robot in a cluster always progresses.
Gotchas found while tuning: (1) the yield radius must cover the backstop's *full* reach
(`_MIN_SEP + _LOOKAHEAD`), or robots freeze just *outside* the yield zone where the backstop still blocks;
(2) *continuous* (non-stuck-triggered) yielding starves low-priority robots and ~halves throughput — only
yield when actually stuck; (3) with 4 robots two can close *simultaneously* (each predicts only its own
step), so `_MIN_SEP` needs margin (0.44→0.48) to keep bodies clear. Residual: `_MIN_SEP` (0.48) is only
0.02 m under `_SLOT_SPACING` (0.50), so adjacent table-slot approaches occasionally trip the backstop and a
rare seed takes ~90 s (vs ~35 s) to unwind — completes, but the proper fix is ORCA/prioritized planning.
Verified collision-free + live across 40 seeds (4×4, 3×3, 3×2), worst body sep 0.45 m.

## 2026-06-19 — Decouple a kinematic visual joint from the base with `gravcomp`, not low mass (v3)
Animating the robot arms (a hinge sub-body, qpos slewed kinematically each `step`), the unactuated base
started drifting — gravity on the (massed) arm transmits a reaction through the shoulder joint to the free
planar base. Lowering the arm's density helped but never reached the Phase-1 "base is static to 1e-9"
invariant. Fix: `<body gravcomp="1">` on the arm — MuJoCo cancels gravity on that body, so it exerts zero
gravitational reaction, and combined with zeroing the arm DOF's `qvel` each step the arm is **dynamically
inert** (base drift measured exactly 0). Lesson: to add a visual, kinematically-driven appendage to a
dynamic body without perturbing it, use `gravcomp` + qvel-zeroing — don't fight it with tuned mass.

## 2026-06-19 — Two different "drifts" looked the same; only one was the arm (v3)
While fixing the arm-base drift I mis-attributed it: the reach animation's reaction is actually **zero**
post-gravcomp. The drift I'd seen was (a) the massed arm's gravity *before* gravcomp, and (b) **residual
drive velocity** — a leftover `qvel` from the prior drive leg integrating over the parked arm-wait. The
real fixes were `stop_base` on arrival/while parked (kills residual velocity) and gravcomp (kills arm
reaction); the reviewer caught that my *comments* blamed the arm for the residual-velocity drift. Lesson:
when a fix works, still verify *why* — a guard built on the wrong mechanism misleads the next editor.
Also: keep the SM's arrival tolerance strictly *below* the SDK's `_at` "parked" tolerance, so an "arrived"
robot reliably passes pick/place's precondition (boundary margin), independent of any drift.

## 2026-06-18 — Subagents load at session start
Files added to `.claude/agents/` are NOT available mid-session — they're read when the session
starts. After creating/editing them, restart Claude Code (or add via `/agents`) before trying to
delegate, or the Agent call won't find the custom agent. This is why Phase 0 ends with a
mandatory restart before Phase 1.
