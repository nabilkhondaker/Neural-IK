"""Workspace generalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.models.planar_2r import Planar2R


def radial_split(
    positions: NDArray[np.floating[Any]],
    joints: NDArray[np.floating[Any]],
    robot: Planar2R,
    inner_frac: float = 0.7,
) -> dict[str, tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]]:
    """Split data into inner and outer workspace regions.

    Useful for testing whether a model trained on the dense interior
    generalizes to the near-boundary region.
    """
    r_min, r_max = robot.workspace_radius()
    r = np.linalg.norm(positions, axis=1)
    threshold = r_min + inner_frac * (r_max - r_min)
    inner = r <= threshold
    outer = ~inner
    return {
        "inner": (positions[inner], joints[inner]),
        "outer": (positions[outer], joints[outer]),
    }
