"""Dataset quality checks and statistics."""

from __future__ import annotations

from typing import Any

import numpy as np

from neural_ik.data.generation import IKData
from neural_ik.robotics.models.planar_2r import Planar2R


def validate_dataset(data: IKData, robot: Planar2R | None = None) -> dict[str, Any]:
    """Run basic integrity checks.

    Returns a report dictionary; raises ValueError on critical failures.
    """
    report: dict[str, Any] = {}
    pos, jnt = data.positions, data.joints

    report["n_samples"] = len(pos)
    report["has_nan_positions"] = bool(np.isnan(pos).any())
    report["has_nan_joints"] = bool(np.isnan(jnt).any())
    report["has_inf"] = bool(np.isinf(pos).any() or np.isinf(jnt).any())

    if report["has_nan_positions"] or report["has_nan_joints"] or report["has_inf"]:
        raise ValueError("Dataset contains NaN or Inf values")

    if robot is not None:
        reach = robot.is_reachable(pos)
        report["unreachable_count"] = int((~reach).sum())
        report["unreachable_frac"] = float((~reach).mean())
    else:
        report["unreachable_count"] = None
        report["unreachable_frac"] = None

    # Duplicate detection (exact)
    if len(pos) > 1:
        # Approximate uniqueness via rounding
        rounded = np.round(pos, decimals=8)
        unique = np.unique(rounded, axis=0)
        report["unique_positions"] = len(unique)
        report["duplicate_frac"] = 1.0 - len(unique) / len(pos)
    else:
        report["unique_positions"] = len(pos)
        report["duplicate_frac"] = 0.0

    return report


def dataset_statistics(
    splits: dict[str, IKData],
    robot: Planar2R | None = None,
) -> dict[str, Any]:
    """Aggregate statistics across splits."""
    stats: dict[str, Any] = {}
    total = 0
    for name, data in splits.items():
        total += len(data)
        stats[f"{name}_n"] = len(data)
        stats[f"{name}_pos_mean"] = data.positions.mean(axis=0).tolist()
        stats[f"{name}_pos_std"] = data.positions.std(axis=0).tolist()
        stats[f"{name}_jnt_mean"] = data.joints.mean(axis=0).tolist()
        stats[f"{name}_jnt_std"] = data.joints.std(axis=0).tolist()
        if robot is not None:
            r_min, r_max = robot.workspace_radius()
            r = np.linalg.norm(data.positions, axis=1)
            coverage = float(((r >= r_min) & (r <= r_max)).mean())
            stats[f"{name}_workspace_coverage"] = coverage
    stats["total_samples"] = total
    return stats
