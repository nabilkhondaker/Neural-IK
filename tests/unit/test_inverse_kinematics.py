"""Tests for analytical inverse kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from neural_ik.robotics.models.planar_2r import Planar2R


def test_ik_fk_roundtrip():
    robot = Planar2R(link_1=1.0, link_2=0.75)
    rng = np.random.default_rng(0)
    joints = robot.sample_joints(100, rng=rng)
    # Prefer elbow-up region (θ2 < 0)
    joints[:, 1] = -np.abs(joints[:, 1])
    targets = robot.forward_kinematics(joints)
    recovered, reachable = robot.analytical_ik(targets, branch="elbow_up")
    assert np.all(reachable)
    pos2 = robot.forward_kinematics(recovered)
    np.testing.assert_allclose(pos2, targets, atol=1e-6)


def test_unreachable():
    robot = Planar2R(link_1=1.0, link_2=0.75)
    far = np.array([10.0, 10.0])
    theta, reachable = robot.analytical_ik(far)
    assert reachable is False
    assert np.isnan(theta).all()


def test_workspace_boundary():
    robot = Planar2R(link_1=1.0, link_2=0.75)
    r_min, r_max = robot.workspace_radius()
    assert abs(r_min - 0.25) < 1e-9
    assert abs(r_max - 1.75) < 1e-9
    assert robot.is_reachable(np.array([r_max - 1e-6, 0.0]))
    assert not robot.is_reachable(np.array([r_max + 0.1, 0.0]))
