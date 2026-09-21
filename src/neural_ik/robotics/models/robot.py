"""Abstract base class for robot models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RobotModel(ABC):
    """Abstract interface for a robotic manipulator.

    Subclasses must implement forward kinematics, analytical inverse
    kinematics (when available), workspace checks, and joint limits.
    """

    name: str

    @abstractmethod
    def forward_kinematics(
        self, theta: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Map joint angles to end-effector position(s).

        Parameters
        ----------
        theta : array-like
            Joint angles. Shape (n_joints,) or (batch, n_joints).

        Returns
        -------
        positions : ndarray
            End-effector Cartesian coordinates. Shape (2,) or (batch, 2).
        """

    @abstractmethod
    def analytical_ik(
        self,
        target: NDArray[np.floating[Any]],
        branch: str = "elbow_up",
    ) -> tuple[NDArray[np.floating[Any]], bool]:
        """Compute analytical inverse kinematics.

        Parameters
        ----------
        target : array-like
            Desired end-effector position (x, y). Shape (2,) or (batch, 2).
        branch : str
            Configuration branch, e.g. 'elbow_up' or 'elbow_down'.

        Returns
        -------
        theta : ndarray
            Joint angles. Same batching as target.
        reachable : bool or ndarray of bool
            Whether the target lies inside the reachable workspace.
        """

    @abstractmethod
    def is_reachable(self, target: NDArray[np.floating[Any]]) -> NDArray[np.bool_]:
        """Return True for targets inside the reachable workspace."""

    @abstractmethod
    def joint_limits(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Return (lower, upper) joint limit arrays."""

    @property
    @abstractmethod
    def n_joints(self) -> int:
        """Number of joints."""

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom (end-effector)."""
