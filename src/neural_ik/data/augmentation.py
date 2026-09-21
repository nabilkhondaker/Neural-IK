"""Optional data augmentation for IK datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def add_position_noise(
    positions: NDArray[np.floating[Any]],
    std: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating[Any]]:
    """Additive isotropic Gaussian noise on Cartesian positions."""
    if rng is None:
        rng = np.random.default_rng()
    return positions + rng.normal(0.0, std, size=positions.shape)
