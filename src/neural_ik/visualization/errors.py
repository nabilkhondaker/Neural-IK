"""Error distribution and heatmap visualizations."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_error_histogram(
    errors: NDArray[np.floating[Any]],
    ax: plt.Axes | None = None,
    title: str = "Position Error Distribution",
    bins: int = 50,
) -> plt.Axes:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=bins, color="#4472C4", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Position error")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(np.median(errors), color="r", ls="--", label=f"median={np.median(errors):.4f}")
    ax.legend()
    return ax


def plot_error_heatmap(
    positions: NDArray[np.floating[Any]],
    errors: NDArray[np.floating[Any]],
    resolution: int = 60,
    ax: plt.Axes | None = None,
    title: str = "Error Heatmap",
) -> plt.Axes:
    """Binned mean error over the workspace."""
    positions = np.asarray(positions)
    errors = np.asarray(errors)
    mask = np.isfinite(errors)
    positions, errors = positions[mask], errors[mask]

    x_edges = np.linspace(positions[:, 0].min(), positions[:, 0].max(), resolution)
    y_edges = np.linspace(positions[:, 1].min(), positions[:, 1].max(), resolution)
    H_sum, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=[x_edges, y_edges], weights=errors
    )
    H_cnt, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=[x_edges, y_edges]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        H_mean = np.where(H_cnt > 0, H_sum / H_cnt, np.nan)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        H_mean.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
        cmap="hot",
    )
    plt.colorbar(im, ax=ax, label="Mean position error")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return ax
