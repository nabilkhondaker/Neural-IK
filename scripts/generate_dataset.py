#!/usr/bin/env python3
"""Standalone dataset generation script."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_ik.data.datasets import save_dataset
from neural_ik.data.generation import DatasetConfig, generate_ik_dataset
from neural_ik.data.validation import dataset_statistics, validate_dataset
from neural_ik.robotics.models.planar_2r import Planar2R
from neural_ik.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IK dataset")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="artifacts/data/dataset.npz")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--link-1", type=float, default=1.0)
    parser.add_argument("--link-2", type=float, default=0.75)
    args = parser.parse_args()

    setup_logging()
    cfg = DatasetConfig(
        n_samples=args.samples,
        seed=args.seed,
        noise_std=args.noise_std,
        link_1=args.link_1,
        link_2=args.link_2,
    )
    logger.info("Generating %d samples (seed=%d)", args.samples, args.seed)
    splits = generate_ik_dataset(cfg)
    robot = Planar2R(link_1=args.link_1, link_2=args.link_2)
    for name, data in splits.items():
        report = validate_dataset(data, robot)
        logger.info("Split %s: %s", name, report)
    stats = dataset_statistics(splits, robot)
    logger.info("Statistics: %s", stats)
    save_dataset(splits, args.output)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
