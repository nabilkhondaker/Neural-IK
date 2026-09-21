#!/usr/bin/env python3
"""Train neural IK (baseline experiment)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_ik.experiments.base import run_baseline_experiment
from neural_ik.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural IK model")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/experiments/baseline")
    args = parser.parse_args()

    setup_logging()
    result = run_baseline_experiment(
        output_dir=args.output_dir,
        n_samples=args.samples,
        epochs=args.epochs,
        seed=args.seed,
    )
    print("Best val loss:", result.metrics.get("best_val_loss"))
    print("Neural MAE:", result.metrics.get("neural", {}).get("mae"))
    print("Analytical MAE:", result.metrics.get("analytical", {}).get("mae"))


if __name__ == "__main__":
    main()
