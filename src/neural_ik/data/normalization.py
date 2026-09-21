"""Input/output normalization for neural IK."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class Normalizer:
    """Zero-mean unit-variance normalizer for positions and joints.

    Fitted on training data only; applied to all splits and at inference.
    """

    def __init__(self) -> None:
        self.pos_mean: NDArray[np.floating[Any]] | None = None
        self.pos_std: NDArray[np.floating[Any]] | None = None
        self.jnt_mean: NDArray[np.floating[Any]] | None = None
        self.jnt_std: NDArray[np.floating[Any]] | None = None
        self._fitted = False

    def fit(
        self,
        positions: NDArray[np.floating[Any]],
        joints: NDArray[np.floating[Any]],
    ) -> Normalizer:
        self.pos_mean = positions.mean(axis=0)
        self.pos_std = positions.std(axis=0)
        self.pos_std = np.where(self.pos_std < 1e-8, 1.0, self.pos_std)
        self.jnt_mean = joints.mean(axis=0)
        self.jnt_std = joints.std(axis=0)
        self.jnt_std = np.where(self.jnt_std < 1e-8, 1.0, self.jnt_std)
        self._fitted = True
        return self

    def transform_positions(
        self, positions: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        assert self._fitted and self.pos_mean is not None and self.pos_std is not None
        return (positions - self.pos_mean) / self.pos_std

    def inverse_positions(
        self, positions: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        assert self._fitted and self.pos_mean is not None and self.pos_std is not None
        return positions * self.pos_std + self.pos_mean

    def transform_joints(
        self, joints: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        assert self._fitted and self.jnt_mean is not None and self.jnt_std is not None
        return (joints - self.jnt_mean) / self.jnt_std

    def inverse_joints(
        self, joints: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        assert self._fitted and self.jnt_mean is not None and self.jnt_std is not None
        return joints * self.jnt_std + self.jnt_mean

    def state_dict(self) -> dict[str, Any]:
        return {
            "pos_mean": self.pos_mean,
            "pos_std": self.pos_std,
            "jnt_mean": self.jnt_mean,
            "jnt_std": self.jnt_std,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.pos_mean = np.asarray(state["pos_mean"])
        self.pos_std = np.asarray(state["pos_std"])
        self.jnt_mean = np.asarray(state["jnt_mean"])
        self.jnt_std = np.asarray(state["jnt_std"])
        self._fitted = True
