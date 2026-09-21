"""Evaluation metrics, benchmarks, and robustness tests."""

from neural_ik.evaluation.metrics import (
    angular_mae,
    position_error,
    summarize_errors,
)
from neural_ik.evaluation.evaluator import evaluate_solver
from neural_ik.evaluation.benchmarks import latency_benchmark

__all__ = [
    "angular_mae",
    "position_error",
    "summarize_errors",
    "evaluate_solver",
    "latency_benchmark",
]
