"""Tests for numerical IK solvers."""

from __future__ import annotations

import numpy as np

from neural_ik.robotics.kinematics.numerical import (
    damped_least_squares_ik,
    jacobian_pseudoinverse_ik,
    jacobian_transpose_ik,
)
from neural_ik.robotics.models.planar_2r import Planar2R


def test_dls_converges():
    robot = Planar2R()
    target = np.array([1.0, 0.5])
    theta, success, iters = damped_least_squares_ik(robot, target, max_iters=200)
    assert success
    pos = robot.forward_kinematics(theta)
    np.testing.assert_allclose(pos, target, atol=1e-5)


def test_pseudoinverse_converges():
    robot = Planar2R()
    target = np.array([0.8, 0.3])
    theta, success, _ = jacobian_pseudoinverse_ik(robot, target, max_iters=200)
    assert success
    pos = robot.forward_kinematics(theta)
    np.testing.assert_allclose(pos, target, atol=1e-5)


def test_transpose_improves():
    robot = Planar2R()
    target = np.array([1.2, 0.2])
    theta0 = np.array([0.1, 0.1])
    pos0 = robot.forward_kinematics(theta0)
    err0 = np.linalg.norm(target - pos0)
    theta, _, _ = jacobian_transpose_ik(
        robot, target, theta0=theta0, max_iters=300, alpha=0.3
    )
    pos = robot.forward_kinematics(theta)
    err = np.linalg.norm(target - pos)
    assert err < err0
