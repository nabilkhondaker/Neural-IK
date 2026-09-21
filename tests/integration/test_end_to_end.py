"""End-to-end smoke test: generate data, train tiny model, evaluate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_ik.experiments.base import run_baseline_experiment


@pytest.mark.slow
def test_baseline_smoke(tmp_path: Path):
    result = run_baseline_experiment(
        output_dir=tmp_path / "exp",
        n_samples=2000,
        epochs=3,
        seed=0,
        hidden_dims=[32, 16],
    )
    assert result.name == "baseline"
    assert "neural" in result.metrics
    assert result.metrics["neural"]["n"] > 0
    # Neural should produce finite errors
    assert np.isfinite(result.metrics["neural"]["mae"])
