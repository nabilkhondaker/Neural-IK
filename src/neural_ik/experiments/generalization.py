"""Workspace generalization experiment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from neural_ik.evaluation.evaluator import evaluate_solver
from neural_ik.evaluation.generalization import radial_split
from neural_ik.robotics.models.planar_2r import Planar2R

logger = logging.getLogger(__name__)


def run_generalization_experiment(
    positions: np.ndarray,
    joints: np.ndarray,
    predict_fn,
    output_dir: str | Path = "artifacts/experiments/generalization",
) -> dict[str, Any]:
    """Compare error on inner vs outer workspace regions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robot = Planar2R()
    parts = radial_split(positions, joints, robot)

    results = {}
    for region, (pos, _) in parts.items():
        if len(pos) == 0:
            continue
        report = evaluate_solver(robot, pos, predict_fn, name=region)
        results[region] = report["error_stats"]

    import json

    with open(output_dir / "generalization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Generalization experiment written to %s", output_dir)
    return results
