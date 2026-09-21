"""Dataset generation tests."""

from __future__ import annotations

from neural_ik.data.generation import DatasetConfig, generate_ik_dataset
from neural_ik.data.validation import validate_dataset
from neural_ik.robotics.models.planar_2r import Planar2R


def test_generate_small_dataset():
    cfg = DatasetConfig(n_samples=500, seed=1)
    splits = generate_ik_dataset(cfg)
    assert "train" in splits and "val" in splits and "test" in splits
    assert len(splits["train"]) > 0
    robot = Planar2R()
    report = validate_dataset(splits["train"], robot)
    assert report["has_nan_positions"] is False
    assert report["n_samples"] == len(splits["train"])
