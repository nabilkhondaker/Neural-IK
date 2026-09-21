#!/usr/bin/env python3
"""Benchmark analytical IK latency."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_ik.evaluation.benchmarks import latency_benchmark
from neural_ik.robotics.models.planar_2r import Planar2R
from neural_ik.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    robot = Planar2R()
    rng = np.random.default_rng(0)
    joints = robot.sample_joints(2000, rng=rng)
    targets = robot.forward_kinematics(joints)

    def analytical(t: np.ndarray) -> np.ndarray:
        th, _ = robot.analytical_ik(t, branch="elbow_up")
        return th

    result = latency_benchmark(analytical, targets)
    print(result)


if __name__ == "__main__":
    main()
