"""Analytical inverse kinematics utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.models.robot import RobotModel


def analytical_inverse_kinematics(
    robot: RobotModel,
    target: NDArray[np.floating[Any]],
    branch: str = "elbow_up",
) -> tuple[NDArray[np.floating[Any]], NDArray[np.bool_] | bool]:
    """Dispatch to the robot's analytical IK solver.

    Parameters
    ----------
    robot : RobotModel
    target : (2,) or (batch, 2)
    branch : str
        Configuration branch identifier.

    Returns
    -------
    theta, reachable
    """
    return robot.analytical_ik(target, branch=branch)
