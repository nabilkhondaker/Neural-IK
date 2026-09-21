#!/usr/bin/env python3
"""Evaluate analytical IK round-trip accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_ik.evaluation.evaluator import evaluate_solver
from neural_ik.robotics.models.planar_2r import Planar2R
from neural_ik.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging()

    robot = Planar2R()
    rng = np.random.default_rng(args.seed)
    joints = robot.sample_joints(args.n, rng=rng)
    targets = robot.forward_kinematics(joints)

    def analytical(t: np.ndarray) -> np.ndarray:
        th, _ = robot.analytical_ik(t, branch="elbow_up")
        return th

    report = evaluate_solver(robot, targets, analytical, name="analytical")
    print("Analytical IK evaluation:")
    for k, v in report["error_stats"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
