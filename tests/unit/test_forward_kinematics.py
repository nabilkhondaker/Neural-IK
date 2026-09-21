"""Tests for forward kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from neural_ik.robotics.models.planar_2r import Planar2R


def test_fk_zero_pose():
    robot = Planar2R(link_1=1.0, link_2=0.75)
    pos = robot.forward_kinematics(np.array([0.0, 0.0]))
    assert pos.shape == (2,)
    np.testing.assert_allclose(pos, [1.75, 0.0], atol=1e-10)


def test_fk_batch():
    robot = Planar2R()
    theta = np.zeros((10, 2))
    pos = robot.forward_kinematics(theta)
    assert pos.shape == (10, 2)


def test_fk_right_angle():
    robot = Planar2R(link_1=1.0, link_2=1.0)
    # θ1=0, θ2=π/2 → end at (1, 1)
    pos = robot.forward_kinematics(np.array([0.0, np.pi / 2]))
    np.testing.assert_allclose(pos, [1.0, 1.0], atol=1e-10)
