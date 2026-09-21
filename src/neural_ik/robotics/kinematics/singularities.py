"""Singularity and manipulability analysis for planar 2R."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.kinematics.numerical import _jacobian_planar_2r
from neural_ik.robotics.models.planar_2r import Planar2R


def manipulability(robot: Planar2R, theta: NDArray[np.floating[Any]]) -> float | NDArray[np.floating[Any]]:
    """Yoshikawa manipulability measure: sqrt(det(J J^T)).

    Near zero indicates proximity to a singularity.
    """
    theta = np.asarray(theta, dtype=np.float64)
    single = theta.ndim == 1
    if single:
        theta = theta[None, :]
    vals = []
    for th in theta:
        J = _jacobian_planar_2r(robot, th)
        m = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        vals.append(m)
    arr = np.array(vals)
    return float(arr[0]) if single else arr


def near_singularity(
    robot: Planar2R,
    theta: NDArray[np.floating[Any]],
    threshold: float = 0.05,
) -> bool | NDArray[np.bool_]:
    """Flag configurations whose manipulability falls below threshold."""
    m = manipulability(robot, theta)
    if isinstance(m, float):
        return m < threshold
    return m < threshold


def singularity_distance(
    robot: Planar2R,
    theta: NDArray[np.floating[Any]],
) -> float | NDArray[np.floating[Any]]:
    """Heuristic distance to singularity: |sin(theta2)| for 2R.

    For a planar 2R the Jacobian determinant is proportional to
    L1 L2 sin(theta2). Fully stretched or folded configurations
    (theta2 ≈ 0 or ±π) are singular.
    """
    theta = np.asarray(theta, dtype=np.float64)
    single = theta.ndim == 1
    if single:
        theta = theta[None, :]
    d = np.abs(np.sin(theta[..., 1]))
    return float(d[0]) if single else d
