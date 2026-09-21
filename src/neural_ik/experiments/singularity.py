"""Singularity analysis experiment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from neural_ik.evaluation.metrics import fk_consistency_error, summarize_errors
from neural_ik.robotics.kinematics.singularities import manipulability, singularity_distance
from neural_ik.robotics.models.planar_2r import Planar2R

logger = logging.getLogger(__name__)


def run_singularity_analysis(
    predict_fn,
    n_samples: int = 5000,
    seed: int = 0,
    output_dir: str | Path = "artifacts/experiments/singularity",
) -> dict[str, Any]:
    """Correlate prediction error with proximity to singularity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    robot = Planar2R()

    joints = robot.sample_joints(n_samples, rng=rng)
    positions = robot.forward_kinematics(joints)
    pred = predict_fn(positions)
    errors = fk_consistency_error(robot, pred, positions)
    manip = manipulability(robot, joints)
    dist = singularity_distance(robot, joints)

    # Bin by singularity distance
    bins = np.linspace(0, 1, 11)
    bin_stats = []
    for i in range(len(bins) - 1):
        mask = (dist >= bins[i]) & (dist < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_stats.append(
            {
                "dist_lo": float(bins[i]),
                "dist_hi": float(bins[i + 1]),
                "n": int(mask.sum()),
                "mae": float(np.mean(errors[mask])),
            }
        )

    result = {
        "overall": summarize_errors(errors),
        "by_singularity_distance": bin_stats,
        "corr_error_manipulability": float(
            np.corrcoef(errors, np.asarray(manip))[0, 1]
        ),
    }
    import json

    with open(output_dir / "singularity_results.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Singularity analysis written to %s", output_dir)
    return result
