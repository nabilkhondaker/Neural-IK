"""Latency comparison experiment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from neural_ik.evaluation.benchmarks import latency_benchmark
from neural_ik.robotics.models.planar_2r import Planar2R

logger = logging.getLogger(__name__)


def run_latency_experiment(
    solvers: dict[str, Callable[[np.ndarray], np.ndarray]],
    n_targets: int = 2000,
    seed: int = 0,
    output_dir: str | Path = "artifacts/experiments/latency",
) -> dict[str, Any]:
    """Benchmark multiple solvers on the same target set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    robot = Planar2R()
    joints = robot.sample_joints(n_targets, rng=rng)
    targets = robot.forward_kinematics(joints)

    results = {}
    for name, fn in solvers.items():
        results[name] = latency_benchmark(fn, targets)

    import json

    with open(output_dir / "latency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Latency experiment written to %s", output_dir)
    return results
