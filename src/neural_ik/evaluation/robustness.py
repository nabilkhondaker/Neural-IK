"""Noise robustness evaluation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from neural_ik.evaluation.evaluator import evaluate_solver
from neural_ik.robotics.models.robot import RobotModel


def noise_sweep(
    robot: RobotModel,
    clean_targets: NDArray[np.floating[Any]],
    predict_fn: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    noise_levels: list[float],
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate a solver under increasing position noise."""
    rng = np.random.default_rng(seed)
    results = []
    for std in noise_levels:
        noisy = clean_targets + rng.normal(0, std, size=clean_targets.shape)
        report = evaluate_solver(robot, noisy, predict_fn, name=f"noise_{std}")
        report["noise_std"] = std
        results.append(report)
    return results
