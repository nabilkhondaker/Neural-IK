"""Compare analytical vs numerical IK."""

from __future__ import annotations

import numpy as np

from neural_ik.evaluation.evaluator import evaluate_solver
from neural_ik.robotics.kinematics.numerical import damped_least_squares_ik
from neural_ik.robotics.models.planar_2r import Planar2R

robot = Planar2R()
rng = np.random.default_rng(0)
joints = robot.sample_joints(500, rng=rng)
targets = robot.forward_kinematics(joints)

def analytical(t):
    th, _ = robot.analytical_ik(t, branch="elbow_up")
    return th

def numerical(t):
    out = []
    for ti in t:
        th, _, _ = damped_least_squares_ik(robot, ti)
        out.append(th)
    return np.array(out)

for name, fn in [("analytical", analytical), ("numerical_dls", numerical)]:
    r = evaluate_solver(robot, targets, fn, name=name)
    print(name, r["error_stats"])
