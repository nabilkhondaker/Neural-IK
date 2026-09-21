"""Regression tests on known configurations."""

from __future__ import annotations

import numpy as np

from neural_ik.robotics.models.planar_2r import Planar2R


def test_stretched_configuration():
    robot = Planar2R(link_1=1.0, link_2=1.0)
    # Fully stretched along x
    pos = robot.forward_kinematics(np.array([0.0, 0.0]))
    np.testing.assert_allclose(pos, [2.0, 0.0], atol=1e-12)
    th, ok = robot.analytical_ik(np.array([2.0, 0.0]), branch="elbow_up")
    assert ok
    pos2 = robot.forward_kinematics(th)
    np.testing.assert_allclose(pos2, [2.0, 0.0], atol=1e-6)
