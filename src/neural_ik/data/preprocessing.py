"""Preprocessing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.geometry.transforms import wrap_angle


def wrap_joint_angles(joints: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Wrap all joint angles to (-π, π]."""
    return wrap_angle(joints)


def remove_nan_rows(
    positions: NDArray[np.floating[Any]],
    joints: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Drop samples containing NaN."""
    mask = ~(np.isnan(positions).any(axis=1) | np.isnan(joints).any(axis=1))
    return positions[mask], joints[mask]
