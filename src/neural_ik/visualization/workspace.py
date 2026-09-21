"""Workspace visualization."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.kinematics.workspace import workspace_mask
from neural_ik.robotics.models.planar_2r import Planar2R


def plot_workspace(
    robot: Planar2R,
    samples: NDArray[np.floating[Any]] | None = None,
    resolution: int = 200,
    ax: plt.Axes | None = None,
    title: str = "Reachable Workspace",
) -> plt.Axes:
    """Plot annular workspace and optional sample points."""
    r_max = robot.L1 + robot.L2
    xs = np.linspace(-r_max - 0.05, r_max + 0.05, resolution)
    ys = np.linspace(-r_max - 0.05, r_max + 0.05, resolution)
    X, Y = np.meshgrid(xs, ys)
    mask = workspace_mask(robot, X, Y)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=["#cce5ff"], alpha=0.6)
    ax.contour(X, Y, mask.astype(float), levels=[0.5], colors=["#004080"], linewidths=1.5)
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], s=2, c="k", alpha=0.3, label="samples")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
