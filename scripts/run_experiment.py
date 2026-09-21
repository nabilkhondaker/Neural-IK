#!/usr/bin/env python3
"""Run a named experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_ik.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output-dir", default="artifacts/experiments/baseline")
    args = parser.parse_args()
    setup_logging()

    if args.name == "baseline":
        from neural_ik.experiments.base import run_baseline_experiment

        result = run_baseline_experiment(
            output_dir=args.output_dir,
            n_samples=args.samples,
            epochs=args.epochs,
        )
        print(result.metrics)
    else:
        print(f"Unknown experiment: {args.name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
