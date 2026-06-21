"""ORCA reciprocal collision avoidance — pure, no sim.

Drives small multi-agent scenarios to completion and asserts the two properties that matter: agents never
overlap (discs stay ≥ combined radius apart) and they still reach their goals (no deadlock/livelock).
"""

import math

from feathersim.fleet.orca import ORCAAgent, new_velocity

R = 0.5            # agent radius
MAX_SPEED = 1.0
TAU = 2.0          # avoidance time horizon
DT = 0.05
BIAS = 0.03        # symmetry-breaking rotation (rad) — see _pref


def _pref(to_goal, dist):
    """Preferred velocity toward a goal, rotated by a tiny fixed angle. Pure ORCA stalls on perfectly
    symmetric/collinear encounters (it slows to a stop with no lateral preference); a small deterministic
    rotation of the *preferred* velocity breaks the tie so agents pass — and turns the antipodal circle
    into ORCA's signature collision-free swirl. Standard practice in deployed ORCA systems."""
    if dist <= 1e-9:
        return (0.0, 0.0)
    speed = min(MAX_SPEED, 2.0 * dist)                    # P-style slowdown near the goal
    ux, uy = to_goal[0] / dist, to_goal[1] / dist
    c, s = math.cos(BIAS), math.sin(BIAS)
    return ((ux * c - uy * s) * speed, (ux * s + uy * c) * speed)


def _simulate(starts, goals, *, radius=R, steps=1200, reciprocal=True):
    """Step every agent under ORCA toward its goal; return (positions_trace, min_pairwise_gap)."""
    pos = [tuple(p) for p in starts]
    vel = [(0.0, 0.0) for _ in starts]
    n = len(starts)
    min_gap = math.inf
    for _ in range(steps):
        new_vel = []
        for i in range(n):
            to_goal = (goals[i][0] - pos[i][0], goals[i][1] - pos[i][1])
            dist = math.hypot(*to_goal)
            pref = _pref(to_goal, dist)
            me = ORCAAgent(pos[i], vel[i], radius)
            others = [ORCAAgent(pos[j], vel[j], radius, reciprocal=reciprocal) for j in range(n) if j != i]
            new_vel.append(new_velocity(me, others, pref, MAX_SPEED, TAU, DT))
        vel = new_vel
        for i in range(n):
            pos[i] = (pos[i][0] + vel[i][0] * DT, pos[i][1] + vel[i][1] * DT)
        for i in range(n):
            for j in range(i + 1, n):
                min_gap = min(min_gap, math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]))
    reached = all(math.dist(pos[i], goals[i]) < 0.1 for i in range(n))
    return pos, min_gap, reached


def test_no_neighbors_returns_preferred_clamped():
    a = ORCAAgent((0.0, 0.0), (0.0, 0.0), R)
    assert new_velocity(a, [], (0.3, 0.0), MAX_SPEED, TAU, DT) == (0.3, 0.0)      # under cap → unchanged
    fast = new_velocity(a, [], (5.0, 0.0), MAX_SPEED, TAU, DT)                    # over cap → clamped
    assert math.isclose(math.hypot(*fast), MAX_SPEED, rel_tol=1e-6)


def test_head_on_agents_pass_without_colliding():
    # Two agents swapping positions along a line — the canonical reciprocal case.
    _, min_gap, reached = _simulate([(-3.0, 0.0), (3.0, 0.0)], [(3.0, 0.0), (-3.0, 0.0)])
    assert min_gap >= 2 * R - 1e-3       # discs never overlap
    assert reached                       # both still get across (deflect + pass, no deadlock)


def test_perpendicular_crossing():
    _, min_gap, reached = _simulate([(-3.0, 0.0), (0.0, -3.0)], [(3.0, 0.0), (0.0, 3.0)])
    assert min_gap >= 2 * R - 1e-3
    assert reached


def test_antipodal_circle_is_collision_free_and_resolves():
    # The classic ORCA stress test: agents on a circle each driving to the antipodal point. Pure VO would
    # gridlock at the centre; ORCA swirls them through collision-free.
    n = 6
    starts = [(3.0 * math.cos(2 * math.pi * i / n), 3.0 * math.sin(2 * math.pi * i / n)) for i in range(n)]
    goals = [(-x, -y) for (x, y) in starts]
    _, min_gap, reached = _simulate(starts, goals, steps=2000)
    assert min_gap >= 2 * R - 5e-3
    assert reached


def test_recovers_from_initial_overlap():
    # Start the pair already overlapping; the colliding branch must push them apart, not crash.
    pos, min_gap, _ = _simulate([(-0.3, 0.0), (0.3, 0.0)], [(-3.0, 0.0), (3.0, 0.0)], steps=400)
    assert math.dist(pos[0], pos[1]) > 2 * R          # ended separated
    assert all(math.isfinite(c) for p in pos for c in p)


def test_static_obstacle_is_fully_avoided():
    # One mover, one parked (non-reciprocal) blocker straddling the straight-line path: the mover must
    # still clear it (it takes full responsibility) and reach the goal.
    pos = [(-3.0, 0.0), (0.0, 0.0)]
    vel = [(0.0, 0.0), (0.0, 0.0)]
    goal = (3.0, 0.0)
    min_gap = math.inf
    for _ in range(1500):
        to_goal = (goal[0] - pos[0][0], goal[1] - pos[0][1])
        dist = math.hypot(*to_goal)
        pref = _pref(to_goal, dist)
        mover = ORCAAgent(pos[0], vel[0], R)
        blocker = ORCAAgent(pos[1], (0.0, 0.0), R, reciprocal=False)   # parked, won't move
        vel[0] = new_velocity(mover, [blocker], pref, MAX_SPEED, TAU, DT)
        pos[0] = (pos[0][0] + vel[0][0] * DT, pos[0][1] + vel[0][1] * DT)
        min_gap = min(min_gap, math.dist(pos[0], pos[1]))
    assert min_gap >= 2 * R - 1e-3        # cleared the blocker
    assert math.dist(pos[0], goal) < 0.1  # and reached the goal
    assert pos[1] == (0.0, 0.0)           # the parked agent never moved
