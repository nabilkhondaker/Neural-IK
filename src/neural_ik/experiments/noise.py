"""Noise robustness experiment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from neural_ik.evaluation.robustness import noise_sweep
from neural_ik.models.neural_ik import NeuralIKModel
from neural_ik.robotics.models.planar_2r import Planar2R

logger = logging.getLogger(__name__)


def run_noise_experiment(
    model: NeuralIKModel,
    test_positions: np.ndarray,
    output_dir: str | Path = "artifacts/experiments/noise",
    noise_levels: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate neural IK under increasing Cartesian noise."""
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robot = Planar2R()

    def predict(t: np.ndarray) -> np.ndarray:
        return model.predict(t)

    results = noise_sweep(robot, test_positions, predict, noise_levels)
    import json

    serializable = []
    for r in results:
        serializable.append(
            {
                "noise_std": r["noise_std"],
                "error_stats": r["error_stats"],
                "n_valid": r["n_valid"],
            }
        )
    with open(output_dir / "noise_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Noise experiment written to %s", output_dir)
    return results
