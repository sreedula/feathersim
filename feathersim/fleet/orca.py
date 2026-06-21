"""Optimal Reciprocal Collision Avoidance (ORCA) — reciprocal n-body avoidance (van den Berg et al. 2011).

A faithful port of the per-agent algorithm from the canonical RVO2 library (``Agent.cpp``): build one ORCA
half-plane per neighbour, then solve a 2D linear program for the velocity closest to the *preferred* one
that satisfies every half-plane and the max-speed circle; on infeasibility (dense packing) fall back to a
3D LP that minimises the maximum constraint violation.

Pure and sim-agnostic — it operates on positions/velocities only (no MuJoCo), so it's unit-tested directly.
The fleet uses it for **robot↔robot** avoidance on top of A* (which handles the static obstacles), which is
what makes the floor both collision-free *and* deadlock-free without the old priority-yield heuristic.

Reciprocity nuance: a moving robot and a *parked* one don't split avoidance 50/50 — the parked robot won't
budge, so the mover must take full responsibility. Each neighbour therefore carries a ``reciprocal`` flag
(True for a peer that is itself running ORCA, False for a stationary obstacle), selecting the ``0.5`` vs
``1.0`` share of the avoidance vector ``u`` — exactly as RVO2 distinguishes agent-vs-agent from
agent-vs-obstacle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec = tuple[float, float]
_EPS = 1e-10


def _sub(a: Vec, b: Vec) -> Vec: return (a[0] - b[0], a[1] - b[1])
def _add(a: Vec, b: Vec) -> Vec: return (a[0] + b[0], a[1] + b[1])
def _mul(a: Vec, s: float) -> Vec: return (a[0] * s, a[1] * s)
def _dot(a: Vec, b: Vec) -> float: return a[0] * b[0] + a[1] * b[1]
def _det(a: Vec, b: Vec) -> float: return a[0] * b[1] - a[1] * b[0]
def _abs_sq(a: Vec) -> float: return a[0] * a[0] + a[1] * a[1]
def _abs(a: Vec) -> float: return math.sqrt(_abs_sq(a))


def _normalize(a: Vec) -> Vec:
    n = _abs(a)
    return (a[0] / n, a[1] / n) if n > _EPS else (0.0, 0.0)


@dataclass(frozen=True)
class Line:
    """A directed half-plane constraint: feasible velocities lie to the *right* of ``direction`` through
    ``point`` (i.e. ``det(direction, v - point) <= 0``)."""
    point: Vec
    direction: Vec


@dataclass(frozen=True)
class ORCAAgent:
    """A disc agent: world position, current velocity, radius. ``reciprocal`` = this agent is itself running
    ORCA (a moving peer that splits avoidance 50/50); False = a static obstacle the other must fully avoid."""
    pos: Vec
    vel: Vec
    radius: float
    reciprocal: bool = True


def _orca_line(a: ORCAAgent, b: ORCAAgent, inv_tau: float, dt: float) -> Line:
    """The ORCA half-plane for agent ``a`` induced by neighbour ``b`` (RVO2 ``computeNewVelocity``)."""
    rel_pos = _sub(b.pos, a.pos)
    rel_vel = _sub(a.vel, b.vel)
    dist_sq = _abs_sq(rel_pos)
    r = a.radius + b.radius
    r_sq = r * r

    if dist_sq > r_sq:                                   # not yet colliding
        w = _sub(rel_vel, _mul(rel_pos, inv_tau))        # rel. vel − cutoff-circle centre
        w_len_sq = _abs_sq(w)
        dp1 = _dot(w, rel_pos)
        if dp1 < 0.0 and dp1 * dp1 > r_sq * w_len_sq:    # project on the cutoff circle
            w_len = math.sqrt(w_len_sq)
            unit_w = _mul(w, 1.0 / w_len)
            direction = (unit_w[1], -unit_w[0])
            u = _mul(unit_w, r * inv_tau - w_len)
        else:                                            # project on a cone leg
            leg = math.sqrt(dist_sq - r_sq)
            if _det(rel_pos, w) > 0.0:                   # left leg
                direction = ((rel_pos[0] * leg - rel_pos[1] * r) / dist_sq,
                             (rel_pos[0] * r + rel_pos[1] * leg) / dist_sq)
            else:                                        # right leg
                direction = (-(rel_pos[0] * leg + rel_pos[1] * r) / dist_sq,
                             -(-rel_pos[0] * r + rel_pos[1] * leg) / dist_sq)
            dp2 = _dot(rel_vel, direction)
            u = _sub(_mul(direction, dp2), rel_vel)
    else:                                                # already overlapping — react over one timestep
        inv_dt = 1.0 / dt
        w = _sub(rel_vel, _mul(rel_pos, inv_dt))
        w_len = _abs(w)
        unit_w = _mul(w, 1.0 / w_len) if w_len > _EPS else (0.0, 0.0)
        direction = (unit_w[1], -unit_w[0])
        u = _mul(unit_w, r * inv_dt - w_len)

    share = 0.5 if b.reciprocal else 1.0                 # full responsibility vs a non-avoiding obstacle
    return Line(_add(a.vel, _mul(u, share)), direction)


def _lp1(lines: list[Line], i: int, radius: float, opt_vel: Vec, direction_opt: bool) -> tuple[bool, Vec]:
    """Optimise along constraint ``i`` subject to the prior lines and the max-speed circle (RVO2)."""
    line = lines[i]
    dot = _dot(line.point, line.direction)
    disc = dot * dot + radius * radius - _abs_sq(line.point)
    if disc < 0.0:
        return False, (0.0, 0.0)                         # the circle fully invalidates this line
    sqrt_disc = math.sqrt(disc)
    t_left = -dot - sqrt_disc
    t_right = -dot + sqrt_disc
    for j in range(i):
        denom = _det(line.direction, lines[j].direction)
        numer = _det(lines[j].direction, _sub(line.point, lines[j].point))
        if abs(denom) <= _EPS:                           # near-parallel
            if numer < 0.0:
                return False, (0.0, 0.0)
            continue
        t = numer / denom
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, (0.0, 0.0)
    if direction_opt:
        t = t_right if _dot(opt_vel, line.direction) > 0.0 else t_left
    else:
        t = _dot(line.direction, _sub(opt_vel, line.point))
        t = max(t_left, min(t_right, t))
    return True, _add(line.point, _mul(line.direction, t))


def _lp2(lines: list[Line], radius: float, opt_vel: Vec, direction_opt: bool) -> tuple[int, Vec]:
    """Incremental 2D LP: velocity closest to ``opt_vel`` within all half-planes + the circle. Returns
    ``(len(lines), v)`` on success, or ``(i, v)`` where ``i`` is the first infeasible constraint."""
    if direction_opt:
        result = _mul(opt_vel, radius)
    elif _abs_sq(opt_vel) > radius * radius:
        result = _mul(_normalize(opt_vel), radius)
    else:
        result = opt_vel
    for i in range(len(lines)):
        if _det(lines[i].direction, _sub(lines[i].point, result)) > 0.0:
            temp = result
            ok, cand = _lp1(lines, i, radius, opt_vel, direction_opt)
            if not ok:
                return i, temp
            result = cand
    return len(lines), result


def _lp3(lines: list[Line], begin: int, radius: float, result: Vec) -> Vec:
    """Fallback when 2D LP is infeasible: minimise the maximum constraint violation (RVO2; no obstacle
    lines here, so all lines may be relaxed)."""
    distance = 0.0
    for i in range(begin, len(lines)):
        if _det(lines[i].direction, _sub(lines[i].point, result)) > distance:
            proj: list[Line] = []
            for j in range(i):
                determinant = _det(lines[i].direction, lines[j].direction)
                if abs(determinant) <= _EPS:
                    if _dot(lines[i].direction, lines[j].direction) > 0.0:
                        continue                         # same direction — already covered
                    point = _mul(_add(lines[i].point, lines[j].point), 0.5)
                else:
                    point = _add(lines[i].point, _mul(
                        lines[i].direction,
                        _det(lines[j].direction, _sub(lines[i].point, lines[j].point)) / determinant))
                proj.append(Line(point, _normalize(_sub(lines[j].direction, lines[i].direction))))
            temp = result
            opt = (-lines[i].direction[1], lines[i].direction[0])
            cnt, result = _lp2(proj, radius, opt, True)
            if cnt < len(proj):                          # floating-point edge case — keep current
                result = temp
            distance = _det(lines[i].direction, _sub(lines[i].point, result))
    return result


def new_velocity(agent: ORCAAgent, neighbors: list[ORCAAgent], pref_vel: Vec, max_speed: float,
                 time_horizon: float, dt: float) -> Vec:
    """The collision-free velocity for ``agent`` closest to ``pref_vel``, given ``neighbors``.

    ``time_horizon`` (s) is how far ahead avoidance looks; ``dt`` (s) is the sim timestep (used only in the
    already-overlapping recovery branch). With no neighbours this just returns ``pref_vel`` clamped to
    ``max_speed``."""
    if not neighbors:
        return _mul(_normalize(pref_vel), max_speed) if _abs_sq(pref_vel) > max_speed * max_speed else pref_vel
    inv_tau = 1.0 / time_horizon
    lines = [_orca_line(agent, nb, inv_tau, dt) for nb in neighbors]
    fail, result = _lp2(lines, max_speed, pref_vel, False)
    if fail < len(lines):
        result = _lp3(lines, fail, max_speed, result)
    return result
