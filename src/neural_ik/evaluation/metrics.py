"""Error metrics for inverse kinematics evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.geometry.transforms import angular_distance
from neural_ik.robotics.models.robot import RobotModel


def position_error(
    pred_pos: NDArray[np.floating[Any]],
    target_pos: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Euclidean distance between predicted and target positions."""
    pred_pos = np.asarray(pred_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    return np.sqrt(((pred_pos - target_pos) ** 2).sum(axis=-1))


def angular_mae(
    pred_joints: NDArray[np.floating[Any]],
    true_joints: NDArray[np.floating[Any]],
) -> float:
    """Mean absolute angular error with wrapping."""
    pred_joints = np.asarray(pred_joints, dtype=np.float64)
    true_joints = np.asarray(true_joints, dtype=np.float64)
    d = angular_distance(pred_joints, true_joints)
    return float(np.mean(d))


def summarize_errors(
    errors: NDArray[np.floating[Any]],
) -> dict[str, float]:
    """Compute MAE, RMSE, max, and percentile statistics."""
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "max": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "n": 0,
        }
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max": float(np.max(errors)),
        "p50": float(np.percentile(errors, 50)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "n": int(len(errors)),
    }


def fk_consistency_error(
    robot: RobotModel,
    pred_joints: NDArray[np.floating[Any]],
    target_pos: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Position error after applying FK to predicted joints."""
    pred_pos = robot.forward_kinematics(pred_joints)
    return position_error(pred_pos, target_pos)
