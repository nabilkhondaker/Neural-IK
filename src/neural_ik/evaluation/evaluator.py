"""Unified evaluation of analytical, numerical, and neural solvers."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from neural_ik.evaluation.metrics import fk_consistency_error, summarize_errors
from neural_ik.robotics.models.robot import RobotModel


def evaluate_solver(
    robot: RobotModel,
    targets: NDArray[np.floating[Any]],
    predict_fn: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    name: str = "solver",
) -> dict[str, Any]:
    """Evaluate a joint-angle predictor on a set of Cartesian targets.

    Parameters
    ----------
    robot : RobotModel
        Used for forward kinematics consistency check.
    targets : (N, 2)
        Desired end-effector positions.
    predict_fn : callable
        Maps (N, 2) targets → (N, 2) joint predictions.
    name : str
        Label for the report.

    Returns
    -------
    report : dict
        Position-error statistics, latency, and raw errors.
    """
    targets = np.asarray(targets, dtype=np.float64)
    t0 = time.perf_counter()
    joints = predict_fn(targets)
    elapsed = time.perf_counter() - t0

    # Filter NaN predictions (e.g. analytical on unreachable)
    valid = np.isfinite(joints).all(axis=1)
    if valid.sum() == 0:
        return {
            "name": name,
            "n_valid": 0,
            "n_total": len(targets),
            "latency_s": elapsed,
            "throughput_hz": 0.0,
            "error_stats": summarize_errors(np.array([])),
        }

    errs = fk_consistency_error(robot, joints[valid], targets[valid])
    stats = summarize_errors(errs)
    return {
        "name": name,
        "n_valid": int(valid.sum()),
        "n_total": len(targets),
        "latency_s": elapsed,
        "throughput_hz": len(targets) / max(elapsed, 1e-12),
        "error_stats": stats,
        "errors": errs,
        "valid_mask": valid,
    }
