"""Angle utilities and simple transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def wrap_angle(theta: NDArray[np.floating[Any]] | float) -> NDArray[np.floating[Any]] | float:
    """Wrap angle(s) to (-π, π]."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def angular_distance(
    a: NDArray[np.floating[Any]] | float,
    b: NDArray[np.floating[Any]] | float,
) -> NDArray[np.floating[Any]] | float:
    """Smallest absolute angular difference between a and b."""
    return np.abs(wrap_angle(np.asarray(a) - np.asarray(b)))
