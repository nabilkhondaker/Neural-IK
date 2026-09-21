"""Workspace tests."""

from __future__ import annotations

import numpy as np

from neural_ik.robotics.kinematics.workspace import workspace_mask
from neural_ik.robotics.models.planar_2r import Planar2R


def test_workspace_mask_shape():
    robot = Planar2R()
    xs = np.linspace(-2, 2, 20)
    ys = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(xs, ys)
    mask = workspace_mask(robot, X, Y)
    assert mask.shape == X.shape
    assert mask.dtype == bool
