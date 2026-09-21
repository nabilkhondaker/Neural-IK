"""Forward kinematics utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.models.robot import RobotModel


def forward_kinematics(
    robot: RobotModel,
    theta: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Dispatch to the robot's forward kinematics.

    Parameters
    ----------
    robot : RobotModel
    theta : (n_joints,) or (batch, n_joints)

    Returns
    -------
    positions : (2,) or (batch, 2)
    """
    return robot.forward_kinematics(theta)
