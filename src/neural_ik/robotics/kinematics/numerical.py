"""Numerical inverse kinematics solvers for planar 2R.

Implements Jacobian transpose, pseudoinverse, and damped least-squares
(DLS / Levenberg-Marquardt style) iterative methods.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.models.planar_2r import Planar2R


def _jacobian_planar_2r(robot: Planar2R, theta: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Analytic Jacobian of the 2R planar arm (2x2)."""
    th1, th2 = float(theta[0]), float(theta[1])
    L1, L2 = robot.L1, robot.L2
    s1, c1 = np.sin(th1), np.cos(th1)
    s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
    J = np.array(
        [
            [-L1 * s1 - L2 * s12, -L2 * s12],
            [L1 * c1 + L2 * c12, L2 * c12],
        ],
        dtype=np.float64,
    )
    return J


def _iterate(
    robot: Planar2R,
    target: NDArray[np.floating[Any]],
    theta0: NDArray[np.floating[Any]],
    update_fn: Callable[
        [NDArray[np.floating[Any]], NDArray[np.floating[Any]]], NDArray[np.floating[Any]]
    ],
    max_iters: int = 100,
    tol: float = 1e-6,
    alpha: float = 1.0,
) -> tuple[NDArray[np.floating[Any]], bool, int]:
    """Generic iterative IK loop."""
    theta = np.asarray(theta0, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64)
    for i in range(max_iters):
        pos = robot.forward_kinematics(theta)
        err = target - pos
        if np.linalg.norm(err) < tol:
            return theta, True, i + 1
        delta = update_fn(theta, err)
        theta = theta + alpha * delta
    pos = robot.forward_kinematics(theta)
    success = bool(np.linalg.norm(target - pos) < tol * 10)
    return theta, success, max_iters


def jacobian_transpose_ik(
    robot: Planar2R,
    target: NDArray[np.floating[Any]],
    theta0: NDArray[np.floating[Any]] | None = None,
    max_iters: int = 100,
    tol: float = 1e-6,
    alpha: float = 0.5,
) -> tuple[NDArray[np.floating[Any]], bool, int]:
    """Jacobian-transpose method: Δθ = α J^T e."""

    def update(th: NDArray[np.floating[Any]], err: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        J = _jacobian_planar_2r(robot, th)
        return J.T @ err

    if theta0 is None:
        theta0 = np.zeros(2)
    return _iterate(robot, target, theta0, update, max_iters, tol, alpha)


def jacobian_pseudoinverse_ik(
    robot: Planar2R,
    target: NDArray[np.floating[Any]],
    theta0: NDArray[np.floating[Any]] | None = None,
    max_iters: int = 100,
    tol: float = 1e-6,
    alpha: float = 1.0,
) -> tuple[NDArray[np.floating[Any]], bool, int]:
    """Jacobian pseudoinverse: Δθ = α J^{+} e."""

    def update(th: NDArray[np.floating[Any]], err: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        J = _jacobian_planar_2r(robot, th)
        # Moore-Penrose via SVD with small damping for stability
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        S_inv = np.zeros_like(S)
        mask = S > 1e-10
        S_inv[mask] = 1.0 / S[mask]
        Jpinv = Vt.T @ np.diag(S_inv) @ U.T
        return Jpinv @ err

    if theta0 is None:
        theta0 = np.zeros(2)
    return _iterate(robot, target, theta0, update, max_iters, tol, alpha)


def damped_least_squares_ik(
    robot: Planar2R,
    target: NDArray[np.floating[Any]],
    theta0: NDArray[np.floating[Any]] | None = None,
    max_iters: int = 100,
    tol: float = 1e-6,
    lambda_: float = 0.1,
    alpha: float = 1.0,
) -> tuple[NDArray[np.floating[Any]], bool, int]:
    """Damped least-squares (Levenberg-Marquardt style).

    Solves (J J^T + λ² I) Δx = e, then Δθ = J^T Δx.
    """

    def update(th: NDArray[np.floating[Any]], err: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        J = _jacobian_planar_2r(robot, th)
        A = J @ J.T + (lambda_**2) * np.eye(2)
        try:
            dx = np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(A, err, rcond=None)[0]
        return J.T @ dx

    if theta0 is None:
        theta0 = np.zeros(2)
    return _iterate(robot, target, theta0, update, max_iters, tol, alpha)
