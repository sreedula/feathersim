"""Phase 2: pure mecanum kinematics (no MuJoCo)."""

import numpy as np
import pytest

from feathersim.kinematics.holonomic import (
    WHEEL_NAMES,
    MecanumGeometry,
    body_to_wheels,
    inverse_matrix,
    wheels_to_body,
)

GEOM = MecanumGeometry(wheel_radius=0.05, half_length=0.15, half_width=0.15)


def test_geometry_validation_and_reach():
    assert GEOM.reach == pytest.approx(0.30)
    with pytest.raises(ValueError):
        MecanumGeometry(wheel_radius=0.0)
    with pytest.raises(ValueError):
        MecanumGeometry(half_length=-0.1)


def test_zero_twist_zero_wheels():
    assert np.allclose(body_to_wheels(0.0, 0.0, 0.0, GEOM), np.zeros(4))


def test_pure_forward_all_wheels_equal():
    # Driving straight: every wheel spins the same way at vx / r.
    w = body_to_wheels(1.0, 0.0, 0.0, GEOM)
    assert np.allclose(w, np.full(4, 1.0 / GEOM.wheel_radius))


def test_pure_strafe_sign_pattern():
    # Strafing left (vy>0): FL,RR spin backward; FR,RL forward (the mecanum X-pattern).
    w = body_to_wheels(0.0, 1.0, 0.0, GEOM)
    s = 1.0 / GEOM.wheel_radius
    assert np.allclose(w, [-s, s, s, -s])


def test_pure_rotation_sign_pattern():
    # Rotating CCW (omega>0): left wheels backward, right wheels forward.
    w = body_to_wheels(0.0, 0.0, 1.0, GEOM)
    fl, fr, rl, rr = w
    assert fl < 0 and rl < 0
    assert fr > 0 and rr > 0


@pytest.mark.parametrize(
    "twist",
    [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.4, -0.7, 1.3), (-2.0, 0.5, -0.9)],
)
def test_body_wheels_body_round_trip_is_identity(twist):
    recovered = wheels_to_body(body_to_wheels(*twist, GEOM), GEOM)
    assert recovered == pytest.approx(twist)


def test_round_trip_random_geometries_and_twists():
    rng = np.random.default_rng(0)
    for _ in range(200):
        geom = MecanumGeometry(
            wheel_radius=float(rng.uniform(0.02, 0.1)),
            half_length=float(rng.uniform(0.05, 0.3)),
            half_width=float(rng.uniform(0.05, 0.3)),
        )
        twist = tuple(rng.uniform(-3, 3, size=3))
        assert wheels_to_body(body_to_wheels(*twist, geom), geom) == pytest.approx(twist)


def test_inverse_matrix_matches_body_to_wheels():
    twist = np.array([0.4, -0.7, 1.3])
    assert np.allclose(inverse_matrix(GEOM) @ twist, body_to_wheels(*twist, GEOM))


def test_wheels_to_body_rejects_bad_shape():
    with pytest.raises(ValueError):
        wheels_to_body([1.0, 2.0, 3.0], GEOM)


def test_wheel_names_order():
    assert WHEEL_NAMES == ("front_left", "front_right", "rear_left", "rear_right")
