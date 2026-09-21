"""2R planar robot visualization."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.models.planar_2r import Planar2R


def plot_robot(
    robot: Planar2R,
    theta: NDArray[np.floating[Any]],
    target: NDArray[np.floating[Any]] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw the 2R arm in a given configuration.

    Parameters
    ----------
    robot : Planar2R
    theta : (2,) joint angles
    target : optional (2,) desired position marker
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    th1, th2 = theta[0], theta[1]
    x0, y0 = 0.0, 0.0
    x1 = robot.L1 * np.cos(th1)
    y1 = robot.L1 * np.sin(th1)
    x2 = x1 + robot.L2 * np.cos(th1 + th2)
    y2 = y1 + robot.L2 * np.sin(th1 + th2)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot([x0, x1], [y0, y1], "b-", lw=3, label="Link 1")
    ax.plot([x1, x2], [y1, y2], "g-", lw=3, label="Link 2")
    ax.plot(x0, y0, "ko", ms=10, label="Base")
    ax.plot(x1, y1, "bo", ms=8, label="Joint 2")
    ax.plot(x2, y2, "rs", ms=8, label="End effector")
    if target is not None:
        t = np.asarray(target).ravel()
        ax.plot(t[0], t[1], "mx", ms=12, mew=2, label="Target")
    r_max = robot.L1 + robot.L2 + 0.1
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    if title:
        ax.set_title(title)
    return ax
