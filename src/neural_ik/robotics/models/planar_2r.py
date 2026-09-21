"""2R planar manipulator model with analytical IK."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.models.robot import RobotModel


class Planar2R(RobotModel):
    """Two-link planar revolute-revolute manipulator.

    Forward kinematics
    ------------------
    .. math::

        x = L_1 \\cos\\theta_1 + L_2 \\cos(\\theta_1 + \\theta_2)

        y = L_1 \\sin\\theta_1 + L_2 \\sin(\\theta_1 + \\theta_2)

    Analytical inverse kinematics uses the standard geometric solution
    with elbow-up and elbow-down branches.
    """

    name = "planar_2r"

    def __init__(
        self,
        link_1: float = 1.0,
        link_2: float = 0.75,
        theta1_limits: tuple[float, float] = (-np.pi, np.pi),
        theta2_limits: tuple[float, float] = (-np.pi, np.pi),
    ) -> None:
        if link_1 <= 0 or link_2 <= 0:
            raise ValueError("Link lengths must be positive")
        self.L1 = float(link_1)
        self.L2 = float(link_2)
        self._theta1_min, self._theta1_max = theta1_limits
        self._theta2_min, self._theta2_max = theta2_limits

    @property
    def n_joints(self) -> int:
        return 2

    @property
    def n_dof(self) -> int:
        return 2

    def joint_limits(
        self,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        lower = np.array([self._theta1_min, self._theta2_min], dtype=np.float64)
        upper = np.array([self._theta1_max, self._theta2_max], dtype=np.float64)
        return lower, upper

    def forward_kinematics(
        self, theta: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Compute end-effector position from joint angles."""
        theta = np.asarray(theta, dtype=np.float64)
        single = theta.ndim == 1
        if single:
            theta = theta[None, :]
        if theta.shape[-1] != 2:
            raise ValueError(f"Expected last dim 2, got {theta.shape}")

        th1 = theta[..., 0]
        th2 = theta[..., 1]
        x = self.L1 * np.cos(th1) + self.L2 * np.cos(th1 + th2)
        y = self.L1 * np.sin(th1) + self.L2 * np.sin(th1 + th2)
        pos = np.stack([x, y], axis=-1)
        return pos[0] if single else pos

    def is_reachable(self, target: NDArray[np.floating[Any]]) -> NDArray[np.bool_]:
        """Check workspace membership: |L1-L2| <= r <= L1+L2."""
        target = np.asarray(target, dtype=np.float64)
        single = target.ndim == 1
        if single:
            target = target[None, :]
        r = np.sqrt(target[..., 0] ** 2 + target[..., 1] ** 2)
        r_min = abs(self.L1 - self.L2)
        r_max = self.L1 + self.L2
        # Small numerical tolerance
        ok = (r >= r_min - 1e-9) & (r <= r_max + 1e-9)
        return ok[0] if single else ok

    def analytical_ik(
        self,
        target: NDArray[np.floating[Any]],
        branch: str = "elbow_up",
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.bool_] | bool]:
        """Geometric analytical inverse kinematics.

        Parameters
        ----------
        target : (2,) or (N, 2)
            Desired (x, y).
        branch : {'elbow_up', 'elbow_down'}
            Configuration selection. Elbow-up corresponds to negative
            theta2 in the standard convention used here.

        Returns
        -------
        theta : (2,) or (N, 2)
            Joint angles. Unreachable targets return NaN.
        reachable : bool or (N,) bool
        """
        target = np.asarray(target, dtype=np.float64)
        single = target.ndim == 1
        if single:
            target = target[None, :]

        x = target[..., 0]
        y = target[..., 1]
        r2 = x**2 + y**2
        r = np.sqrt(r2)

        reachable = self.is_reachable(target)
        L1, L2 = self.L1, self.L2

        # Cosine of theta2 via law of cosines
        cos_th2 = (r2 - L1**2 - L2**2) / (2.0 * L1 * L2)
        # Clamp for numerical safety
        cos_th2 = np.clip(cos_th2, -1.0, 1.0)
        sin_th2 = np.sqrt(np.maximum(1.0 - cos_th2**2, 0.0))

        if branch == "elbow_up":
            th2 = -np.arctan2(sin_th2, cos_th2)
        elif branch == "elbow_down":
            th2 = np.arctan2(sin_th2, cos_th2)
        else:
            raise ValueError(f"Unknown branch: {branch}")

        # theta1 from atan2 of adjusted position
        k1 = L1 + L2 * cos_th2
        k2 = L2 * np.sin(th2)  # note: sin(th2) already signed via branch
        th1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        theta = np.stack([th1, th2], axis=-1)
        theta = np.where(reachable[:, None], theta, np.nan)

        if single:
            return theta[0], bool(reachable[0])
        return theta, reachable

    def both_branches(
        self, target: NDArray[np.floating[Any]]
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.bool_] | bool,
    ]:
        """Return both elbow-up and elbow-down solutions."""
        up, reach = self.analytical_ik(target, branch="elbow_up")
        down, _ = self.analytical_ik(target, branch="elbow_down")
        return up, down, reach

    def workspace_radius(self) -> tuple[float, float]:
        """Return (r_min, r_max) of the annular workspace."""
        return abs(self.L1 - self.L2), self.L1 + self.L2

    def sample_joints(
        self,
        n: int,
        rng: np.random.Generator | None = None,
        respect_limits: bool = True,
    ) -> NDArray[np.floating[Any]]:
        """Uniform random joint configurations within limits."""
        if rng is None:
            rng = np.random.default_rng()
        lower, upper = self.joint_limits()
        if not respect_limits:
            lower = np.array([-np.pi, -np.pi])
            upper = np.array([np.pi, np.pi])
        return rng.uniform(lower, upper, size=(n, 2))

    def __repr__(self) -> str:
        return (
            f"Planar2R(L1={self.L1}, L2={self.L2}, "
            f"limits=[{self._theta1_min:.2f},{self._theta1_max:.2f}]x"
            f"[{self._theta2_min:.2f},{self._theta2_max:.2f}])"
        )
