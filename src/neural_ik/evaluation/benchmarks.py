"""Latency and throughput benchmarks."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


def latency_benchmark(
    predict_fn: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    targets: NDArray[np.floating[Any]],
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, float]:
    """Measure mean / std inference latency."""
    targets = np.asarray(targets, dtype=np.float64)
    for _ in range(warmup):
        predict_fn(targets)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(targets)
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    n = len(targets)
    return {
        "mean_latency_s": float(arr.mean()),
        "std_latency_s": float(arr.std()),
        "min_latency_s": float(arr.min()),
        "max_latency_s": float(arr.max()),
        "throughput_hz": float(n / arr.mean()),
        "n_samples": n,
        "repeats": repeats,
    }
