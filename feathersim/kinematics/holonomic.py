"""Mecanum holonomic drive kinematics — pure functions, no sim import. [Phase 2]

A 4-wheel mecanum base maps a body-frame twist ``(vx, vy, omega)`` to four wheel angular
velocities and back. Wheels are ordered ``(front-left, front-right, rear-left, rear-right)``;
body frame is x-forward, y-left, omega counter-clockwise (right-hand rule).

The inverse map ``J`` (body → wheels, before the 1/r scale) has orthogonal columns, so the
forward map is its scaled transpose and ``body → wheels → body`` recovers the twist exactly
(the redundant 4th wheel only adds a null space the other way). That identity is what the
go-to-pose controller leans on, and what the round-trip tests pin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


@dataclass(frozen=True)
class MecanumGeometry:
    """Physical layout of the mecanum base (SI units).

    ``half_length`` is the center-to-wheel distance along x (forward), ``half_width`` along y
    (left). Only their sum enters the kinematics, exposed as :attr:`reach`.
    """

    wheel_radius: float = 0.05
    half_length: float = 0.15
    half_width: float = 0.15

    def __post_init__(self) -> None:
        if self.wheel_radius <= 0:
            raise ValueError(f"wheel_radius must be > 0, got {self.wheel_radius}")
        if self.half_length < 0 or self.half_width < 0:
            raise ValueError("half_length/half_width must be >= 0")

    @property
    def reach(self) -> float:
        """``lx + ly`` — the lever arm that converts yaw rate to wheel speed."""
        return self.half_length + self.half_width


def body_to_wheels(vx: float, vy: float, omega: float, geom: MecanumGeometry) -> np.ndarray:
    """Inverse kinematics: body twist → 4 wheel angular velocities (rad/s), in WHEEL_NAMES order."""
    L = geom.reach
    return np.array(
        [
            vx - vy - L * omega,  # front-left
            vx + vy + L * omega,  # front-right
            vx + vy - L * omega,  # rear-left
            vx - vy + L * omega,  # rear-right
        ]
    ) / geom.wheel_radius


def wheels_to_body(wheels, geom: MecanumGeometry) -> tuple[float, float, float]:
    """Forward kinematics: 4 wheel angular velocities → body twist ``(vx, vy, omega)``."""
    w = np.asarray(wheels, dtype=float)
    if w.shape != (4,):
        raise ValueError(f"expected 4 wheel speeds, got shape {w.shape}")
    r, L = geom.wheel_radius, geom.reach
    fl, fr, rl, rr = w
    vx = (r / 4.0) * (fl + fr + rl + rr)
    vy = (r / 4.0) * (-fl + fr + rl - rr)
    omega = (r / (4.0 * L)) * (-fl + fr - rl + rr)
    return (float(vx), float(vy), float(omega))


def inverse_matrix(geom: MecanumGeometry) -> np.ndarray:
    """The 4×3 inverse-kinematics matrix ``M`` such that ``wheels = M @ [vx, vy, omega]``."""
    L = geom.reach
    return np.array(
        [
            [1.0, -1.0, -L],
            [1.0, 1.0, L],
            [1.0, 1.0, -L],
            [1.0, -1.0, L],
        ]
    ) / geom.wheel_radius
