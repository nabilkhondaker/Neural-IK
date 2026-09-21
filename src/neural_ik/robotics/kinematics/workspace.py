"""Workspace analysis utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.models.planar_2r import Planar2R


def workspace_mask(
    robot: Planar2R,
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
) -> NDArray[np.bool_]:
    """Boolean mask of reachable points on a grid.

    Parameters
    ----------
    robot : Planar2R
    x, y : meshgrid-compatible arrays

    Returns
    -------
    mask : bool array, same shape as x/y
    """
    targets = np.stack([x.ravel(), y.ravel()], axis=-1)
    reach = robot.is_reachable(targets)
    return reach.reshape(x.shape)


def sample_workspace(
    robot: Planar2R,
    n: int,
    rng: np.random.Generator | None = None,
    method: str = "uniform_disk",
) -> NDArray[np.floating[Any]]:
    """Sample points uniformly inside the reachable workspace.

    Methods
    -------
    uniform_disk : rejection sampling in the annular region
    joint_space : sample joints then FK (biased toward interior)
    """
    if rng is None:
        rng = np.random.default_rng()
    r_min, r_max = robot.workspace_radius()

    if method == "joint_space":
        joints = robot.sample_joints(n, rng=rng)
        return robot.forward_kinematics(joints)

    # Rejection sampling in annulus
    points = []
    while len(points) < n:
        # Sample in bounding square then reject
        cand = rng.uniform(-r_max, r_max, size=(n * 2, 2))
        r = np.linalg.norm(cand, axis=1)
        ok = (r >= r_min) & (r <= r_max)
        points.append(cand[ok])
    pts = np.concatenate(points, axis=0)[:n]
    return pts
