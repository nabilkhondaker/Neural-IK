"""Base experiment utilities and baseline runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_ik.data.datasets import IKDataset, load_dataset, save_dataset
from neural_ik.data.generation import DatasetConfig, generate_ik_dataset
from neural_ik.data.normalization import Normalizer
from neural_ik.evaluation.evaluator import evaluate_solver
from neural_ik.models.factory import build_model
from neural_ik.models.neural_ik import NeuralIKModel
from neural_ik.robotics.models.planar_2r import Planar2R
from neural_ik.training.reproducibility import set_seed
from neural_ik.training.trainer import TrainConfig, Trainer

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def run_baseline_experiment(
    output_dir: str | Path = "artifacts/experiments/baseline",
    n_samples: int = 20_000,
    epochs: int = 30,
    seed: int = 42,
    hidden_dims: list[int] | None = None,
) -> ExperimentResult:
    """Train a standard neural IK model and compare against analytical IK.

    This is a self-contained runnable experiment suitable for CI and
    quick validation. Results are written under output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    if hidden_dims is None:
        hidden_dims = [128, 128, 64]

    robot = Planar2R(link_1=1.0, link_2=0.75)
    cfg = DatasetConfig(n_samples=n_samples, seed=seed)
    splits = generate_ik_dataset(cfg)
    data_path = output_dir / "dataset.npz"
    save_dataset(splits, data_path)

    # Normalization
    normalizer = Normalizer().fit(splits["train"].positions, splits["train"].joints)

    train_ds = IKDataset(splits["train"].positions, splits["train"].joints, normalizer)
    val_ds = IKDataset(splits["val"].positions, splits["val"].joints, normalizer)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = build_model(
        {"hidden_dims": hidden_dims, "activation": "relu", "dropout": 0.0}
    )
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=256,
        learning_rate=1e-3,
        checkpoint_dir=str(output_dir / "checkpoints"),
        log_every=max(1, epochs // 5),
    )
    trainer = Trainer(model, train_cfg)
    history = trainer.fit(train_loader, val_loader)

    # Build inference wrapper
    nik = NeuralIKModel(network=model, normalizer=normalizer)
    nik.save(output_dir / "neural_ik.pt")

    # Evaluate on test set
    test_pos = splits["test"].positions

    def neural_predict(t: np.ndarray) -> np.ndarray:
        return nik.predict(t)

    def analytical_predict(t: np.ndarray) -> np.ndarray:
        th, _ = robot.analytical_ik(t, branch="elbow_up")
        return th

    neural_report = evaluate_solver(robot, test_pos, neural_predict, name="neural")
    analytical_report = evaluate_solver(
        robot, test_pos, analytical_predict, name="analytical"
    )

    result = ExperimentResult(
        name="baseline",
        metrics={
            "neural": neural_report["error_stats"],
            "analytical": analytical_report["error_stats"],
            "neural_latency_s": neural_report["latency_s"],
            "analytical_latency_s": analytical_report["latency_s"],
            "best_val_loss": history.best_val_loss,
            "best_epoch": history.best_epoch,
            "n_train": len(splits["train"]),
            "n_test": len(splits["test"]),
        },
        artifacts={
            "dataset": str(data_path),
            "checkpoint": str(output_dir / "neural_ik.pt"),
        },
        notes="Baseline neural IK vs analytical IK on held-out test set.",
    )

    # Persist summary
    import json

    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "name": result.name,
                "metrics": result.metrics,
                "artifacts": result.artifacts,
                "notes": result.notes,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Baseline experiment complete. Results in %s", output_dir)
    return result
