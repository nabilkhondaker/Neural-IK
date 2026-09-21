"""Neural IK Lab CLI."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from neural_ik.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Neural IK Lab — neural inverse kinematics research toolkit."""
    setup_logging("DEBUG" if verbose else "INFO")


@cli.command("generate-data")
@click.option("--samples", default=100_000, show_default=True, help="Number of samples.")
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default="artifacts/data/dataset.npz", show_default=True)
@click.option("--noise-std", default=0.0, show_default=True)
def generate_data(samples: int, seed: int, output: str, noise_std: float) -> None:
    """Generate an IK training dataset via forward kinematics."""
    from neural_ik.data.generation import DatasetConfig, generate_ik_dataset
    from neural_ik.data.datasets import save_dataset
    from neural_ik.data.validation import dataset_statistics
    from neural_ik.robotics.models.planar_2r import Planar2R

    cfg = DatasetConfig(n_samples=samples, seed=seed, noise_std=noise_std)
    logger.info("Generating dataset with %d samples (seed=%d)", samples, seed)
    splits = generate_ik_dataset(cfg)
    save_dataset(splits, output)
    robot = Planar2R()
    stats = dataset_statistics(splits, robot)
    logger.info("Saved dataset to %s", output)
    logger.info(
        "Total=%d  train=%d  val=%d  test=%d",
        stats["total_samples"],
        stats.get("train_n", 0),
        stats.get("val_n", 0),
        stats.get("test_n", 0),
    )


@cli.command("train")
@click.option("--config", default="configs/default.yaml", show_default=True)
@click.option("--output-dir", default="artifacts/experiments/baseline", show_default=True)
@click.option("--epochs", default=None, type=int, help="Override epochs.")
@click.option("--samples", default=None, type=int, help="Override sample count.")
def train(config: str, output_dir: str, epochs: int | None, samples: int | None) -> None:
    """Train a neural IK model (baseline experiment)."""
    from neural_ik.experiments.base import run_baseline_experiment
    from neural_ik.utils.config import load_config

    cfg = load_config(config) if Path(config).exists() else {}
    ep = epochs or cfg.get("training", {}).get("epochs", 30)
    ns = samples or cfg.get("dataset", {}).get("samples", 20_000)
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    logger.info("Starting baseline training (epochs=%d, samples=%d)", ep, ns)
    result = run_baseline_experiment(
        output_dir=output_dir, n_samples=ns, epochs=ep, seed=seed
    )
    logger.info("Training complete. Best val loss=%.6f", result.metrics.get("best_val_loss"))


@cli.command("evaluate")
@click.option("--checkpoint", required=True, help="Path to neural_ik.pt")
@click.option("--dataset", default="artifacts/data/dataset.npz")
def evaluate(checkpoint: str, dataset: str) -> None:
    """Evaluate a trained neural IK model on a test set."""
    from neural_ik.data.datasets import load_dataset
    from neural_ik.evaluation.evaluator import evaluate_solver
    from neural_ik.models.architectures import MLP
    from neural_ik.models.neural_ik import NeuralIKModel
    from neural_ik.robotics.models.planar_2r import Planar2R

    splits = load_dataset(dataset)
    # Rebuild a default architecture; real use should store arch in checkpoint
    net = MLP(hidden_dims=(128, 128, 64))
    model = NeuralIKModel.load(checkpoint, network=net)
    robot = Planar2R()
    report = evaluate_solver(robot, splits["test"].positions, model.predict, name="neural")
    logger.info("Evaluation: %s", report["error_stats"])


@cli.command("benchmark")
@click.option("--n", default=2000, show_default=True)
def benchmark(n: int) -> None:
    """Latency benchmark of analytical IK."""
    from neural_ik.evaluation.benchmarks import latency_benchmark
    from neural_ik.robotics.models.planar_2r import Planar2R
    import numpy as np

    robot = Planar2R()
    rng = np.random.default_rng(0)
    joints = robot.sample_joints(n, rng=rng)
    targets = robot.forward_kinematics(joints)

    def analytical(t: np.ndarray) -> np.ndarray:
        th, _ = robot.analytical_ik(t, branch="elbow_up")
        return th

    result = latency_benchmark(analytical, targets)
    logger.info("Analytical latency: %s", result)


@cli.command("experiment")
@click.option("--name", default="baseline", show_default=True)
@click.option("--output-dir", default=None)
def experiment(name: str, output_dir: str | None) -> None:
    """Run a named experiment."""
    if name == "baseline":
        from neural_ik.experiments.base import run_baseline_experiment

        out = output_dir or "artifacts/experiments/baseline"
        run_baseline_experiment(output_dir=out)
    else:
        raise click.ClickException(
            f"Unknown experiment '{name}'. Supported: baseline"
        )


@cli.command("report")
@click.option("--output", default="artifacts/reports/research_report.md", show_default=True)
def report(output: str) -> None:
    """Generate a research report skeleton."""
    from neural_ik.reporting.generator import generate_text_report

    text = generate_text_report(output_path=output)
    logger.info("Report written to %s (%d chars)", output, len(text))


if __name__ == "__main__":
    cli()
