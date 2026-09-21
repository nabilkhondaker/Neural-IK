"""Evaluation pipeline."""

from __future__ import annotations

import numpy as np

from neural_ik.evaluation.evaluator import evaluate_solver
from neural_ik.robotics.models.planar_2r import Planar2R


def test_eval_analytical():
    robot = Planar2R()
    rng = np.random.default_rng(1)
    j = robot.sample_joints(200, rng=rng)
    t = robot.forward_kinematics(j)

    def pred(x):
        th, _ = robot.analytical_ik(x, branch="elbow_up")
        return th

    report = evaluate_solver(robot, t, pred, name="analytical")
    assert report["error_stats"]["mae"] < 1e-5
